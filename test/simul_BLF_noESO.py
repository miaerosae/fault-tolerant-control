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
import ftc.agents.BLF_noESO as BLF
from ftc.agents.param import get_b0, get_faulty_input, get_PID_gain
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
        self.blf_x = BLF.outerLoop(Kxy,
                                   params.oL.rho, params.oL.rho_k,
                                   -self.pos_ref[0][0], cond.BLF)
        self.blf_y = BLF.outerLoop(Kxy,
                                   params.oL.rho, params.oL.rho_k,
                                   -self.pos_ref[1][0], cond.BLF)
        self.blf_z = BLF.outerLoop(Kxy,
                                   params.oL.rho, params.oL.rho_k,
                                   -self.pos_ref[2][0], cond.BLF)
        J = np.diag(self.plant.J)
        b = np.array([1/J[0], 1/J[1], 1/J[2]])
        Kang = np.array([k21, k22, k23])
        self.blf_phi = BLF.innerLoop(Kang,
                                     params.iL.rho,
                                     b[0], cond.BLF)
        self.blf_theta = BLF.innerLoop(Kang,
                                       params.iL.rho,
                                       b[1], cond.BLF)
        self.blf_psi = BLF.innerLoop(Kang,
                                     params.iL.rho_psi,
                                     b[2], cond.BLF)

        self.prev_rotors = np.zeros((4, 1))

    def get_ref(self, t):
        pos_des = np.vstack([0, 0, -t])
        dref = np.vstack([0, 0, -1])
        ddref = np.vstack([0, 0, 0])
        return pos_des, dref, ddref

    def step(self):
        env_info, done = self.update()
        # pos = self.plant.pos.state
        # ang = quat2angle(self.plant.quat.state)
        # dang = self.plant.omega.state
        # for i in range(3):
        #     if abs(pos[i]) > 1:
        #         done = True
        #     if abs(ang[i]) > cfg.agents.BLF.iL.rho[0]:
        #         done = True
        #     if abs(dang[i]) > cfg.agents.BLF.iL.rho[1]:
        #         done = True
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
        ref, dref, ddref = self.get_ref(t)
        W = self.get_W(t)
        What = self.get_W(t-self.delay)
        # windvel = self.get_windvel(t)

        # Outer-Loop: virtual input
        epos = self.plant.pos.state - ref
        eposd = self.plant.vel.state - dref
        vel = self.plant.vel.state
        e_ = np.hstack((epos, eposd))
        q = np.zeros((3, 1))
        q[0] = self.blf_x.get_virtual(t, e_[0], vel[0], dref[0], ddref[0])
        q[1] = self.blf_y.get_virtual(t, e_[1], vel[1], dref[1], ddref[1])
        q[2] = self.blf_z.get_virtual(t, e_[2], vel[2], dref[2], ddref[2])

        # Inverse solution
        u1_cmd = self.plant.m * (q[0]**2 + q[1]**2 + (q[2]-self.plant.g)**2)**(1/2)
        phid = np.clip(np.arcsin(q[1] * self.plant.m / u1_cmd),
                       - np.deg2rad(45), np.deg2rad(45))
        thetad = np.clip(np.arctan(q[0] / (q[2] - self.plant.g)),
                         - np.deg2rad(45), np.deg2rad(45))
        psid = 0
        eulerd = np.vstack([phid, thetad, psid])

        # Inner-Loop
        ang = np.array(quat2angle(self.plant.quat.state))[::-1][:, None]
        dang = self.plant.omega.state
        x_ = np.hstack((ang, dang))
        # caculate f
        J = np.diag(self.plant.J)
        p_, q_, r_ = self.plant.omega.state
        f = np.array([(J[1]-J[2]) / J[0] * q_ * r_,
                      (J[2]-J[0]) / J[1] * p_ * r_,
                      (J[0]-J[1]) / J[2] * p_ * q_])
        u2 = self.blf_phi.get_u(t, x_[0], phid, f[0])
        u3 = self.blf_theta.get_u(t, x_[1], thetad, f[1])
        u4 = self.blf_psi.get_u(t, x_[2], psid, f[2])

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

        # get model uncertainty disturbance value
        model_uncert_vel, model_uncert_omega = self.plant.get_model_uncertainty(rotors, t)
        int_vel = self.plant.get_int_uncertainties(t, self.plant.vel.state)

        # set_dot
        self.plant.set_dot(t, rotors,
                           prev_rotors=self.prev_rotors
                           )
        x, y, z = self.plant.pos.state.ravel()
        euler = quat2angle(self.plant.quat.state)[::-1]
        self.blf_x.set_dot(e_[0][0])
        self.blf_y.set_dot(e_[1][0])
        self.blf_z.set_dot(e_[2][0])
        self.blf_phi.set_dot(t, euler[0], phid)
        self.blf_theta.set_dot(t, euler[1], thetad)
        self.blf_psi.set_dot(t, euler[2], psid)

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref,
                    virtual_u=forces, q=q, f=f,
                    model_uncert_vel=model_uncert_vel,
                    model_uncert_omega=model_uncert_omega,
                    int_uncert_vel=int_vel,
                    eulerd=eulerd,
                    obs_pos=np.zeros((3, 1)), obs_ang=np.zeros((3, 1)))


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
            "k11": tune.uniform(0.1, 500),
            "k12": tune.uniform(0.1, 500),
            "k13": 0,
            "k21": tune.uniform(0.1, 500),
            "k22": tune.uniform(0.1, 500),
            "k23": 0,
        }
        current_best_params = [{
            "k11": 2,
            "k12": 30,
            "k13": 0,
            "k21": 500/30,
            "k22": 30,
            "k23": 0,
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
