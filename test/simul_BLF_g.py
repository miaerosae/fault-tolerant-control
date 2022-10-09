from ray import tune
import os
import json
from ray.air import CheckpointConfig, RunConfig
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune import CLIReporter
import argparse
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging
from fym.utils.rot import angle2quat, quat2angle

import ftc.config
from ftc.models.multicopter import Multicopter
import ftc.agents.BLF_g as BLF
from ftc.agents.param import get_b0, get_faulty_input
from ftc.plotting import exp_plot
import ftc.plotting_comp as comp
import ftc.plotting_forpaper as pfp
from copy import deepcopy
from ftc.faults.actuator import LoE
from ftc.faults.manager import LoEManager

plt.rc("text", usetex=False)
plt.rc("lines", linewidth=1.5)
plt.rc("axes", grid=True, labelsize=15, titlesize=12)
plt.rc("grid", linestyle="--", alpha=0.8)
plt.rc("legend", fontsize=15)

cfg = ftc.config.load()


class Env(BaseEnv):
    def __init__(self, config):
        super().__init__(dt=cfg.simul_condi.dt, max_t=cfg.simul_condi.max_t)
        init = cfg.models.multicopter.init
        cond = cfg.simul_condi
        self.plant = Multicopter(init.pos, init.vel, init.quat, init.omega,
                                 cond.blade, cond.ext_unc, cond.int_unc, cond.hub,
                                 cond.gyro, cond.uncertainty,
                                 cond.groundEffect, cond.drygen
                                 )
        self.n = self.plant.mixer.B.shape[1]

        # Define actuator dynamics
        # self.act_dyn = ActuatorDynamcs(tau=0.01, shape=(n, 1))

        # Define faults
        self.fault = True
        self.delay = cfg.faults.manager.delay
        self.fault_time = cfg.faults.manager.fault_time
        self.fault_index = cfg.faults.manager.fault_index
        self.LoE = cfg.faults.manager.LoE

        # Define agents
        params = cfg.agents.BLF
        self.rho_pos = params.oL.rho[1]
        self.rho_ang = params.iL.rho
        k11 = config["k11"]
        k12 = config["k12"]
        k13 = config["k13"]
        k21 = config["k21"]
        k22 = config["k22"]
        k23 = config["k23"]
        Kxy = np.array([k11, k12, k13])
        self.pos_ref = np.vstack([-0, 0, 0])
        self.blf_x = BLF.outerLoop(params.oL.alp, config["eps11"], Kxy,
                                   params.oL.rho, params.oL.rho_k,
                                   params.theta,
                                   cond.noise, -self.pos_ref[0][0], cond.BLF)
        self.blf_y = BLF.outerLoop(params.oL.alp, config["eps12"], Kxy,
                                   params.oL.rho, params.oL.rho_k,
                                   params.theta,
                                   cond.noise, -self.pos_ref[1][0], cond.BLF)
        self.blf_z = BLF.outerLoop(params.oL.alp, config["eps13"], Kxy,
                                   params.oL.rho, params.oL.rho_k,
                                   params.theta,
                                   cond.noise, -self.pos_ref[2][0], cond.BLF)
        J = np.diag(self.plant.J)
        b = np.array([1/J[0], 1/J[1], 1/J[2]])
        Kang = np.array([k21, k22, k23])
        self.blf_phi = BLF.innerLoop(params.iL.alp, config["eps21"], Kang,
                                     params.iL.xi, params.iL.rho,
                                     params.iL.c, b[0], self.plant.g, params.theta,
                                     cond.noise, cond.BLF)
        self.blf_theta = BLF.innerLoop(params.iL.alp, config["eps22"], Kang,
                                       params.iL.xi, params.iL.rho,
                                       params.iL.c, b[1], self.plant.g, params.theta,
                                       cond.noise, cond.BLF)
        self.blf_psi = BLF.innerLoop(params.iL.alp, config["eps23"], Kang,
                                     params.iL.xi_psi, params.iL.rho_psi,
                                     params.iL.c, b[2], self.plant.g, params.theta,
                                     cond.noise, cond.BLF)

        self.prev_rotors = np.zeros((4, 1))

    def get_ref(self, t):
        pos_des = np.vstack([np.sin(t/2)*np.cos(np.pi*t/5)*np.cos(np.pi/4),
                             np.sin(t/2)*np.sin(np.pi*t/5)*np.cos(np.pi/4),
                             -t])
        return pos_des

    def get_dref(self, t):
        pi = np.pi
        dref = np.vstack([1/2*np.cos(t/2)*np.cos(pi*t/5)-pi/5*np.sin(t/2)*np.sin(pi*t/5),
                          1/2*np.cos(t/2)*np.sin(pi*t/5)+pi/5*np.sin(t/2)*np.cos(pi*t/5),
                          -1])
        ddref = np.vstack([(- pi/5*np.cos(t/2)*np.sin(pi*t/5)
                            - (1/4 + pi**2/25)*np.sin(t/2)*np.cos(pi*t/5)),
                           (pi/5*np.cos(t/2)*np.cos(pi*t/5)
                            - (1/4 + pi**2/25)*np.sin(t/2)*np.sin(pi*t/5)),
                           0])
        return dref, ddref

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def get_W(self, t):
        W = np.diag([1., 1., 1., 1.])
        if self.fault is True:
            index = self.fault_index
            for i in range(np.size(self.fault_time)):
                if t > self.fault_time[i]:
                    W[index[i], index[i]] = self.LoE[i]
        return W

    def set_dot(self, t):
        ref = self.get_ref(t)
        dref, ddref = self.get_dref(t)
        W = self.get_W(t)
        What = self.get_W(t-self.delay)
        # windvel = self.get_windvel(t)

        # Outer-Loop: virtual input
        q = np.zeros((3, 1))
        q[0] = self.blf_x.get_virtual(t, ref[0], dref[0], ddref[0])
        q[1] = self.blf_y.get_virtual(t, ref[1], dref[1], ddref[1])
        q[2] = self.blf_z.get_virtual(t, ref[2], dref[2], ddref[2])

        # Inverse solution
        u1_cmd = self.plant.m * (q[0]**2 + q[1]**2 + (q[2]-self.plant.g)**2)**(1/2)
        phid = np.clip(np.arcsin(q[1] * self.plant.m / u1_cmd),
                       - np.deg2rad(45), np.deg2rad(45))
        thetad = np.clip(np.arctan(q[0] / (q[2] - self.plant.g)),
                         - np.deg2rad(45), np.deg2rad(45))
        psid = 0

        # m = self.plant.m
        # u1_cmd = m * (q[0]**2 + q[1]**2 + (q[2]-self.plant.g)**2)**(1/2)
        # euler = quat2angle(self.plant.quat.state)[::-1]
        # psid = euler[2]
        # phid = np.clip(np.arcsin(- m / u1_cmd * (q[2]*np.sin(psid)-q[1]*np.cos(psid))),
        #                - np.deg2rad(45), np.deg2rad(45))
        # thetad = np.clip(np.arctan(1 / (q[2] - self.plant.g) * (q[0]*np.cos(psid)+q[1]*np.sin(psid))),
        #                  - np.deg2rad(45), np.deg2rad(45))

        eulerd = np.vstack([phid, thetad, psid])

        # Inner-Loop
        u2 = self.blf_phi.get_u(t, phid)
        u3 = self.blf_theta.get_u(t, thetad)
        u4 = self.blf_psi.get_u(t, psid)

        # Saturation u1
        u1 = np.clip(u1_cmd, 0,
                     self.plant.rotor_max**2*self.n*cfg.models.multicopter.physPropBy.OS4.b)

        # rotors
        forces = np.vstack([u1, u2, u3, u4])
        rotors_cmd = np.linalg.pinv(self.plant.mixer.B.dot(What)).dot(forces)
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        # Set actuator faults
        rotors = get_faulty_input(W, rotors)
        self.prev_rotors = rotors

        # Disturbance
        dist = np.zeros((6, 1))
        dist[0] = self.blf_x.get_dist()
        dist[1] = self.blf_y.get_dist()
        dist[2] = self.blf_z.get_dist()
        dist[3] = self.blf_phi.get_dist()
        dist[4] = self.blf_theta.get_dist()
        dist[5] = self.blf_psi.get_dist()

        # Observation
        obs_pos = np.zeros((3, 1))
        obs_pos[0] = self.blf_x.get_err()
        obs_pos[1] = self.blf_y.get_err()
        obs_pos[2] = self.blf_z.get_err()
        obs_ang = np.zeros((3, 1))
        obs_ang[0] = self.blf_phi.get_obs()
        obs_ang[1] = self.blf_theta.get_obs()
        obs_ang[2] = self.blf_psi.get_obs()

        # caculate f
        J = np.diag(self.plant.J)
        p_, q_, r_ = self.plant.omega.state
        f = np.array([(J[1]-J[2]) / J[0] * q_ * r_,
                      (J[2]-J[0]) / J[1] * p_ * r_,
                      (J[0]-J[1]) / J[2] * p_ * q_])

        # get model uncertainty disturbance value
        model_uncert_vel, model_uncert_omega = self.plant.get_model_uncertainty(rotors)

        # get gain
        kpos_x = self.blf_x.get_gain(t, ref[0])
        kpos_y = self.blf_y.get_gain(t, ref[1])
        kpos_z = self.blf_z.get_gain(t, ref[2])
        kang_phi = self.blf_phi.get_gain(t, eulerd[0])
        kang_theta = self.blf_theta.get_gain(t, eulerd[1])
        kang_psi = self.blf_psi.get_gain(t, eulerd[2])

        # set_dot
        self.plant.set_dot(t, rotors,
                           # windvel,
                           prev_rotors=self.prev_rotors
                           )
        x, y, z = self.plant.pos.state.ravel()
        euler = quat2angle(self.plant.quat.state)[::-1]
        self.blf_x.set_dot(t, x, ref[0], q[0], dref[0], ddref[0])
        self.blf_y.set_dot(t, y, ref[1], q[1], dref[1], ddref[1])
        self.blf_z.set_dot(t, z, ref[2], q[2], dref[2], ddref[2])
        self.blf_phi.set_dot(t, euler[0], phid)
        self.blf_theta.set_dot(t, euler[1], thetad)
        self.blf_psi.set_dot(t, euler[2], psid)

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref,
                    virtual_u=forces, dist=dist, q=q, f=f,
                    obs_pos=obs_pos, obs_ang=obs_ang, eulerd=eulerd,
                    model_uncert_vel=model_uncert_vel,
                    model_uncert_omega=model_uncert_omega,
                    kpos1=kpos_x, kpos2=kpos_y, kpos3=kpos_z,
                    kang1=kang_phi, kang2=kang_theta, kang3=kang_psi)


def run(loggerpath, params):
    env = Env(params)
    env.logger = fym.Logger(loggerpath)
    env.logger.set_info(cfg=ftc.config.load())

    env.reset()

    try:
        while True:
            env.render()
            done, env_info = env.step()
            # env.logger.record(env=env_info)
            env_info = {
                # "detection_time": env.detection_time,
                "rotor_min": env.plant.rotor_min,
                "rotor_max": env.plant.rotor_max,
            }
            env.logger.set_info(**env_info)

            if done:
                break

    finally:
        env.close()
        return


def main(args):
    if args.with_ray:
        def objective(config):
            np.seterr(all="raise")
            env = Env(config)

            env.reset()
            tf = 0
            try:
                while True:
                    done, env_info = env.step()
                    tf = env_info["t"]

                    if done:
                        break

            finally:
                return {"tf": tf}

        config = {
            "k11": tune.uniform(0.1, 40),
            "k12": tune.uniform(0.1, 40),
            "k13": tune.uniform(0.1, 40),
            "k21": tune.uniform(0.1, 40),
            "k22": tune.uniform(0.1, 40),
            "k23": tune.uniform(0.1, 40),
            "eps11": tune.uniform(1.1, 7),
            "eps12": tune.uniform(1.1, 7),
            "eps13": tune.uniform(1.1, 10),
            "eps21": tune.uniform(15, 35),
            "eps22": tune.uniform(15, 35),
            "eps23": tune.uniform(15, 35),
        }
        current_best_params = [{
            "k11": 2,
            "k12": 30,
            "k13": 5/30/(0.5)**2,
            "k21": 500/30,
            "k22": 30,
            "k23": 5/30/np.deg2rad(45)**2,
            "eps11": 5,
            "eps12": 5,
            "eps13": 10,
            "eps21": 25,
            "eps22": 25,
            "eps23": 25,
        }]
        search = HyperOptSearch(
            metric="tf",
            mode="max",
            points_to_evaluate=current_best_params,
        )
        tuner = tune.Tuner(
            tune.with_resources(
                objective,
                resources={"cpu": os.cpu_count()},
                # resources={"cpu": 12},
            ),
            param_space=config,
            tune_config=tune.TuneConfig(
                num_samples=1000,
                search_alg=search,
            ),
            run_config=RunConfig(
                name="train_run",
                local_dir="data/ray_results",
                verbose=1,
                progress_reporter=CLIReporter(
                    parameter_columns=list(config.keys())[:3],
                    max_progress_rows=3,
                    metric="tf",
                    mode="max",
                    sort_by_metric=True,
                ),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=5,
                    checkpoint_score_attribute="tf",
                    checkpoint_score_order="max",
                ),
            ),
        )
        results = tuner.fit()
        config = results.get_best_result(metric="tf", mode="max").config
        with open("data/ray_results/train_run/best_config.json", "w") as f:
            json.dump(config, f)  # json file은 cat cmd로 볼 수 있다
        return

    elif args.with_plot:
        loggerpath = "data.h5"
        exp_plot(loggerpath, False)

    else:
        loggerpath = "data.h5"
        params = {
            "k11": cfg.agents.BLF.Kxy[0],
            "k12": cfg.agents.BLF.Kxy[1],
            "k13": cfg.agents.BLF.Kxy[2],
            "k21": cfg.agents.BLF.Kang[0],
            "k22": cfg.agents.BLF.Kang[1],
            "k23": cfg.agents.BLF.Kang[2],
            "eps11": cfg.agents.BLF.oL.eps[0],
            "eps12": cfg.agents.BLF.oL.eps[1],
            "eps13": cfg.agents.BLF.oL.eps[2],
            "eps21": cfg.agents.BLF.iL.eps[0],
            "eps22": cfg.agents.BLF.iL.eps[1],
            "eps23": cfg.agents.BLF.iL.eps[2],
        }

        run(loggerpath, params)
        exp_plot(loggerpath, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--with-ray", action="store_true")
    parser.add_argument("-p", "--with-plot", action="store_true")
    args = parser.parse_args()
    main(args)
    # comp.exp_plot("data.h5", "data1.h5")
    # pfp.exp_plot("data.h5")
