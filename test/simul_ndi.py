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
import ftc.agents.ndi as ndi
import ftc.agents.ESO as ESO
from ftc.agents.param import get_b0, get_sumOfDist, get_PID_gain
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

        Ko1 = np.diag((config["ko11"], config["ko12"]))
        Ko2 = np.diag((config["ko21"], config["ko22"]))
        Ki1 = np.diag((config["ki11"], config["ki12"],
                       config["ki13"], config["ki14"]))
        Ki2 = np.diag((config["ki21"], config["ki22"],
                       config["ki23"], config["ki24"]))

        # Define faults
        self.fault = True
        self.delay = cfg.faults.manager.delay
        self.fault_time = cfg.faults.manager.fault_time
        self.fault_index = cfg.faults.manager.fault_index
        self.LoE = cfg.faults.manager.LoE

        # controller
        self.ndi = ndi.NDIController(self.plant.mixer.B, self.plant.g,
                                     self.plant.m, self.plant.J,
                                     np.linalg.inv(self.plant.J),
                                     Ko1, Ko2, Ki1, Ki2)

        params = cfg.agents.BLF
        self.eso_x = ESO.outerLoop(params.oL.alp, config["eps11"],
                                   params.theta, 0)
        self.eso_y = ESO.outerLoop(params.oL.alp, config["eps12"],
                                   params.theta, 0)
        self.eso_z = ESO.outerLoop(params.oL.alp, config["eps13"],
                                   params.theta, 0)
        J = np.diag(self.plant.J)
        b = np.array([1/J[0], 1/J[1], 1/J[2]])
        self.eso_phi = ESO.innerLoop(params.iL.alp, config["eps21"],
                                     params.iL.xi, params.iL.c, b[0],
                                     params.theta)
        self.eso_theta = ESO.innerLoop(params.iL.alp, config["eps22"],
                                       params.iL.xi, params.iL.c, b[1],
                                       params.theta)
        self.eso_psi = ESO.innerLoop(params.iL.alp, config["eps23"],
                                     params.iL.xi, params.iL.c, b[2],
                                     params.theta)
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
        if abs(self.eso_x.e.state[0]) > 5:
            done = True
        if abs(self.eso_y.e.state[0]) > 5:
            done = True
        if abs(self.eso_z.e.state[0]) > 5:
            done = True
        ang = quat2angle(self.plant.quat.state)
        for i in range(3):
            if abs(ang[i]) > 90:
                done = True
        dang = self.plant.omega.state
        for i in range(3):
            if abs(dang[i]) > 300:
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
        dref, ddref = self.get_dref(t)
        W = self.get_W(t)

        # control input
        obs_pos = np.vstack((self.eso_x.get_obs(),
                             self.eso_y.get_obs(),
                             self.eso_z.get_obs()))
        obs_posd = np.vstack((self.eso_x.get_obsdot(),
                              self.eso_y.get_obsdot(),
                              self.eso_z.get_obsdot()))
        obs_eul = np.vstack((self.eso_phi.get_obs(),
                             self.eso_theta.get_obs(),
                             self.eso_psi.get_obs()))
        obs_euld = np.vstack((self.eso_phi.get_obsdot(),
                              self.eso_theta.get_obsdot(),
                              self.eso_psi.get_obsdot()))

        angd, q, u = self.ndi.get_control(t, obs_pos, obs_posd, obs_eul,
                                          obs_euld, ref, dref)
        q = np.vstack((q, 0))

        # Saturation u1
        u[0] = np.clip(-u[0], 0,
                       self.plant.rotor_max**2*self.n*cfg.models.multicopter.physPropBy.OS4.b)

        # rotors
        forces = u
        rotors_cmd = np.linalg.pinv(self.plant.mixer.B).dot(forces)
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        # Set actuator faults
        rotors = W.dot(rotors)
        self.prev_rotors = rotors

        # Disturbance
        dist = np.zeros((6, 1))
        dist[0] = self.eso_x.get_dist()
        dist[1] = self.eso_y.get_dist()
        dist[2] = self.eso_z.get_dist()
        dist[3] = self.eso_phi.get_dist()
        dist[4] = self.eso_theta.get_dist()
        dist[5] = self.eso_psi.get_dist()

        # Observation
        obs_ang = obs_eul

        # set_dot
        self.plant.set_dot(t, rotors,
                           # windvel,
                           prev_rotors=self.prev_rotors
                           )
        pos = self.plant.pos.state
        x, y, z = self.plant.pos.state.ravel()
        euler = quat2angle(self.plant.quat.state)[::-1]
        self.eso_x.set_dot(t, x, ref[0], q[0], dref[0], ddref[0])
        self.eso_y.set_dot(t, y, ref[1], q[1], dref[1], ddref[1])
        self.eso_z.set_dot(t, z, ref[2], q[2], dref[2], ddref[2])
        self.eso_phi.set_dot(t, euler[0], angd[0], u[1])
        self.eso_theta.set_dot(t, euler[1], angd[1], u[2])
        self.eso_psi.set_dot(t, euler[2], angd[2], u[3])

        return dict(t=t, x=self.plant.observe_dict(), What=np.diag([1, 1, 1, 1]),
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref,
                    virtual_u=forces, dist=dist, q=q,
                    obs_pos=obs_pos, obs_ang=obs_ang, eulerd=angd,
                    err=sum(1e2*abs(pos-ref))+sum(abs(np.vstack((euler))-angd)))


def run(loggerpath, params):
    env = Env(params)
    env.logger = fym.Logger(loggerpath)
    env.logger.set_info(cfg=ftc.config.load())

    env.reset()

    sumDistErr = 0
    try:
        while True:
            env.render()
            done, env_info = env.step()
            sumDistErr = sumDistErr + env_info["err"]
            env_info["rotor_min"] = env.plant.rotor_min
            env_info["rotor_max"] = env.plant.rotor_max
            env.logger.set_info(**env_info)

            if done:
                tf = env_info["t"]
                print(str(tf))
                print(str(sumDistErr))
                if np.isnan(sumDistErr):
                    sumDistErr = [1e6]
                print(str(100*tf-sumDistErr[0]))
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
            sumErr = 0
            try:
                while True:
                    done, env_info = env.step()
                    tf = env_info["t"]
                    sumErr = sumErr + env_info["err"]

                    if done:
                        break

            finally:
                if np.isnan(sumErr):
                    sumErr = [1e6]
                return {"cost": 1e+5*tf - sumErr[0]}

        config = {
            "ko11": tune.uniform(3., 30.),
            "ko12": tune.uniform(1, 20.),
            "ko21": tune.uniform(3, 30.),
            "ko22": tune.uniform(1., 20.),
            "ki11": tune.uniform(25, 500.),
            "ki12": tune.uniform(50, 1000.),
            "ki13": tune.uniform(250, 5000.),
            "ki14": tune.uniform(50, 1000.),
            "ki21": tune.uniform(25, 500.),
            "ki22": tune.uniform(50, 1000.),
            "ki23": tune.uniform(250, 5000.),
            "ki24": tune.uniform(50, 1000.),
            "eps11": cfg.agents.BLF.oL.eps[0],
            "eps12": cfg.agents.BLF.oL.eps[1],
            "eps13": cfg.agents.BLF.oL.eps[2],
            "eps21": cfg.agents.BLF.iL.eps[0],
            "eps22": cfg.agents.BLF.iL.eps[1],
            "eps23": cfg.agents.BLF.iL.eps[2],
        }
        current_best_params = [{
            "ko11": 12,
            "ko12": 3,
            "ko21": 9,
            "ko22": 6,
            "ki11": 25,
            "ki12": 50,
            "ki13": 250,
            "ki14": 50,
            "ki21": 25,
            "ki22": 50,
            "ki23": 250,
            "ki24": 50,
            "eps11": cfg.agents.BLF.oL.eps[0],
            "eps12": cfg.agents.BLF.oL.eps[1],
            "eps13": cfg.agents.BLF.oL.eps[2],
            "eps21": cfg.agents.BLF.iL.eps[0],
            "eps22": cfg.agents.BLF.iL.eps[1],
            "eps23": cfg.agents.BLF.iL.eps[2],
        }]
        search = HyperOptSearch(
            metric="cost",
            mode="max",
            points_to_evaluate=current_best_params,
        )
        tuner = tune.Tuner(
            tune.with_resources(
                objective,
                # resources={"cpu": os.cpu_count()},
                resources={"cpu": 12},
            ),
            param_space=config,
            tune_config=tune.TuneConfig(
                num_samples=3000,
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

    elif args.with_plot:
        loggerpath = "data.h5"
        # loggerpath = "Scenario2_noFDI.h5"
        exp_plot(loggerpath, False)

    else:
        loggerpath = "data.h5"
        params = {
            "ko11": 10,
            "ko12": 3,
            "ko21": 3,
            "ko22": 2,
            "ki11": 25,
            "ki12": 50,
            "ki13": 250,
            "ki14": 50,
            "ki21": 5,
            "ki22": 10,
            "ki23": 50,
            "ki24": 10,
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
