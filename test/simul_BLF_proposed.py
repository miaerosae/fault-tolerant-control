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
import ftc.agents.BLF_proposed2 as BLF
from ftc.agents.param import get_b0, get_sumOfDist
from ftc.plotting import exp_plot
import ftc.plotting_comp as comp
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
        params = cfg.agents.BLF.pf
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
        self.blf_x = BLF.outerLoop(config["l11"], params.oL.alp, params.oL.bet,
                                   params.oL.R[0, :], Kxy, params.oL.rho,
                                   params.oL.rho_k, cond.noise,
                                   -self.pos_ref[0][0], params.theta)
        self.blf_y = BLF.outerLoop(config["l12"], params.oL.alp, params.oL.bet,
                                   params.oL.R[1, :], Kxy, params.oL.rho,
                                   params.oL.rho_k, cond.noise,
                                   -self.pos_ref[1][0], params.theta)
        self.blf_z = BLF.outerLoop(config["l13"], params.oL.alp, params.oL.bet,
                                   params.oL.R[2, :], Kxy, params.oL.rho,
                                   params.oL.rho_k, cond.noise,
                                   -self.pos_ref[2][0], params.theta)
        J = np.diag(self.plant.J)
        b = np.array([1/J[0], 1/J[1], 1/J[2]])
        Kang = np.array([k21, k22, k23])
        self.blf_phi = BLF.innerLoop(config["l21"], params.iL.alp, params.iL.bet,
                                     params.iL.dist_range, Kang, params.iL.xi,
                                     params.iL.rho, params.iL.c, b[0],
                                     self.plant.g, params.theta, cond.noise)
        self.blf_theta = BLF.innerLoop(config["l22"], params.iL.alp, params.iL.bet,
                                       params.iL.dist_range, Kang, params.iL.xi,
                                       params.iL.rho, params.iL.c, b[1],
                                       self.plant.g, params.theta, cond.noise)
        self.blf_psi = BLF.innerLoop(config["l23"], params.iL.alp, params.iL.bet,
                                     params.iL.dist_range, Kang, params.iL.xi_psi,
                                     params.iL.rho_psi, params.iL.c, b[2],
                                     self.plant.g, params.theta, cond.noise)

        self.prev_rotors = np.zeros((4, 1))

    def get_ref(self, t):
        # pos_des = self.pos_ref
        pos_des = np.vstack([np.sin(t/2)*np.cos(np.pi*t/5)*np.cos(np.pi/4),
                             np.sin(t/2)*np.sin(np.pi*t/5)*np.cos(np.pi/4),
                             -t])
        return pos_des

    def step(self):
        env_info, done = self.update()

        if abs(self.blf_x.e.state[0]) > 0.5:
            done = True
        if abs(self.blf_y.e.state[0]) > 0.5:
            done = True
        if abs(self.blf_z.e.state[0]) > 0.5:
            done = True
        ang = quat2angle(self.plant.quat.state)
        for i in range(3):
            if abs(ang[i]) > cfg.agents.BLF.pf.iL.rho[0]:
                done = True
        dang = self.plant.omega.state
        for i in range(3):
            if abs(dang[i]) > cfg.agents.BLF.pf.iL.rho[1]:
                done = True

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
        W = self.get_W(t)
        What = np.diag([1., 1., 1., 1.])

        # Outer-Loop: virtual input
        q = np.zeros((3, 1))
        q[0] = self.blf_x.get_virtual(t)
        q[1] = self.blf_y.get_virtual(t)
        q[2] = self.blf_z.get_virtual(t)

        # Inverse solution
        # u1_cmd = self.plant.m * (q[0]**2 + q[1]**2 + (q[2]-self.plant.g)**2)**(1/2)
        # phid = np.arcsin(q[1] * self.plant.m / u1_cmd)
        # thetad = np.arctan(q[0] / (q[2] - self.plant.g))
        # psid = 0

        m = self.plant.m
        u1_cmd = m * (q[0]**2 + q[1]**2 + (q[2]-self.plant.g)**2)**(1/2)
        phid = np.clip(np.arcsin(q[1] * self.plant.m / u1_cmd),
                       - np.deg2rad(45), np.deg2rad(45))
        thetad = np.clip(np.arctan(q[0] / (q[2] - self.plant.g)),
                         - np.deg2rad(45), np.deg2rad(45))
        psid = 0
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
        rotors_cmd = np.linalg.pinv(self.plant.mixer.B).dot(forces)
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        # get model uncertainty disturbance value
        real_dist = get_sumOfDist(t, cfg.simul_condi.ext_unc)
        dist_vel, dist_omega = self.plant.get_dist(t, W, rotors)
        real_dist = real_dist + np.vstack([dist_vel, dist_omega])

        # Set actuator faults
        rotors = W.dot(rotors)
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

        # set_dot
        self.plant.set_dot(t, rotors,
                           # windvel
                           prev_rotors=self.prev_rotors
                           )
        x, y, z = self.plant.pos.state.ravel()
        euler = quat2angle(self.plant.quat.state)[::-1]
        self.blf_x.set_dot(t, x, ref[0])
        self.blf_y.set_dot(t, y, ref[1])
        self.blf_z.set_dot(t, z, ref[2])
        self.blf_phi.set_dot(t, euler[0], phid)
        self.blf_theta.set_dot(t, euler[1], thetad)
        self.blf_psi.set_dot(t, euler[2], psid)

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref,
                    virtual_u=forces, dist=dist, q=q,
                    obs_pos=obs_pos, obs_ang=obs_ang, eulerd=eulerd,
                    dist_vel=dist_vel, dist_omega=dist_omega,
                    disterr=sum(abs(dist-real_dist)))


def run(loggerpath, params):
    env = Env(params)
    env.logger = fym.Logger(loggerpath)
    env.logger.set_info(cfg=ftc.config.load())

    env.reset()

    sumDistErr = 0
    # try:
    while True:
        env.render()
        done, env_info = env.step()
        sumDistErr = sumDistErr + env_info["disterr"]

        env_info["rotor_min"] = env.plant.rotor_min
        env_info["rotor_max"] = env.plant.rotor_max
        env.logger.set_info(**env_info)
        if done:
            tf = env_info["t"]
            print(str(tf))
            print(str(sumDistErr))
            print(str(100*tf-sumDistErr[0]))
            break

    # finally:
    env.close()
    # return


def main(args):
    if args.with_ray:
        def objective(config):
            np.seterr(all="raise")
            env = Env(config)

            env.reset()
            tf = 0
            sumDistErr = 0
            try:
                while True:
                    done, env_info = env.step()
                    tf = env_info["t"]
                    sumDistErr = env_info["disterr"]

                    if done:
                        break

            finally:
                return {"cost": 100*tf-sumDistErr[0]}

        config = {
            "k11": tune.uniform(0.01, 5),
            "k12": tune.uniform(0.01, 10),
            "k13": tune.uniform(0.01, 1),
            "k21": tune.uniform(1, 30),
            "k22": tune.uniform(50, 200),
            "k23": tune.uniform(0.01, 1),
            "l11": tune.uniform(20, 100),
            "l12": tune.uniform(20, 100),
            "l13": tune.uniform(10, 50),
            "l21": tune.uniform(30, 250),
            "l22": tune.uniform(30, 250),
            "l23": tune.uniform(30, 150),
        }
        current_best_params = [{
            "k11": 1.1878,
            "k12": 5.1439,
            "k13": 0.2293,
            "k21": 18.2726,
            "k22": 77.03,
            "k23": 0.1892,
            "l11": 35,
            "l12": 35,
            "l13": 4,
            "l21": 40,
            "l22": 40,
            "l23": 8,
        }]
        search = HyperOptSearch(
            metric="cost",
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
                num_samples=4000,
                search_alg=search,
            ),
            run_config=RunConfig(
                name="train_run",
                local_dir="data/ray_results",
                verbose=1,
                progress_reporter=CLIReporter(
                    parameter_columns=list(config.keys())[:3],
                    max_progress_rows=3,
                    metric="cost",
                    mode="max",
                    sort_by_metric=True,
                ),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=5,
                    checkpoint_score_attribute="cost",
                    checkpoint_score_order="max",
                ),
            ),
        )
        results = tuner.fit()
        config = results.get_best_result(metric="cost", mode="max").config
        with open("data/ray_results/train_run/best_config.json", "w") as f:
            json.dump(config, f)  # json file은 cat cmd로 볼 수 있다
        return

    else:
        loggerpath = "data.h5"
        params = {
            "k11": cfg.agents.BLF.pf.Kxy[0],
            "k12": cfg.agents.BLF.pf.Kxy[1],
            "k13": cfg.agents.BLF.pf.Kxy[2],
            "k21": cfg.agents.BLF.pf.Kang[0],
            "k22": cfg.agents.BLF.pf.Kang[1],
            "k23": cfg.agents.BLF.pf.Kang[2],
            "l11": cfg.agents.BLF.pf.oL.l[0],
            "l12": cfg.agents.BLF.pf.oL.l[1],
            "l13": cfg.agents.BLF.pf.oL.l[2],
            "l21": cfg.agents.BLF.pf.iL.l[0],
            "l22": cfg.agents.BLF.pf.iL.l[1],
            "l23": cfg.agents.BLF.pf.iL.l[2],
        }
        run(loggerpath, params)
        exp_plot(loggerpath, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--with-ray", action="store_true")
    args = parser.parse_args()
    main(args)
    # comp.exp_plot4("result_comp_ESO_blf.h5", "result_comp_ESO_blf_g.h5", "result_comp_ESO_blf_pf.h5", "data.h5")
    # comp.exp_plot4("result_comp_ESO_blf.h5", "result_comp_ESO_blf_g.h5", "result_comp_ESO_blf_pf.h5", "result_comp_ESO_blf_proposed.h5")
    # exp_plot("result_blf_proposed.h5", True)
    # comp.exp_plot("data.h5", "data1.h5")
