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
import ftc.agents.PID as PID
from ftc.agents.param import (get_b0, get_faulty_input,
                              get_PID_gain,
                              get_PID_gain_reverse)
from ftc.plotting import exp_plot
import ftc.plotting_comp as comp
from copy import deepcopy
from ftc.faults.actuator import LoE
from ftc.faults.manager import LoEManager

plt.rc("text", usetex=False)
plt.rc("lines", linewidth=1)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=0.8)

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
        self.fault = False
        self.delay = cfg.faults.manager.delay
        self.fault_time = cfg.faults.manager.fault_time
        self.fault_index = cfg.faults.manager.fault_index
        self.LoE = cfg.faults.manager.LoE

        # Define agents
        self.pos_ref = np.vstack([-0, 0, 0])
        kpos = np.array([config["k11"], config["k12"], config["k13"]])
        kang = np.array([config["k21"], config["k22"], config["k23"]])
        self.kpos, self.kang = kpos, kang
        # kpos, kang = get_PID_gain(params)
        params = cfg.agents.BLF
        self.blf_x = PID.PIDController(params.oL.alp, params.oL.eps[0],
                                       params.theta, -self.pos_ref[0][0],
                                       kpos, "pos")
        self.blf_y = PID.PIDController(params.oL.alp, params.oL.eps[1],
                                       params.theta, -self.pos_ref[1][0],
                                       kpos, "pos")
        self.blf_z = PID.PIDController(params.oL.alp, params.oL.eps[2],
                                       params.theta, -self.pos_ref[2][0],
                                       kpos, "pos")
        J = np.diag(self.plant.J)
        b = np.array([1/J[0], 1/J[1], 1/J[2]])
        self.blf_phi = PID.PIDController(params.iL.alp, params.iL.eps[0],
                                         params.theta, 0,
                                         kpos, "ang", b[0])
        self.blf_theta = PID.PIDController(params.iL.alp, params.iL.eps[1],
                                           params.theta, 0,
                                           kpos, "ang", b[1])
        self.blf_psi = PID.PIDController(params.iL.alp, params.iL.eps[2],
                                         params.theta, 0,
                                         kpos, "ang", b[2])

        self.prev_rotors = np.zeros((4, 1))

    def get_ref(self, t):
        # pos_des = self.pos_ref
        # dref = np.zeros((3, 1))
        pos_des = np.vstack([np.sin(t/2)*np.cos(np.pi*t/5),
                             np.sin(t/2)*np.sin(np.pi*t/5),
                             -t])
        pi = np.pi
        dref = np.vstack([1/2*np.cos(t/2)*np.cos(pi*t/5)-pi/5*np.sin(t/2)*np.sin(pi*t/5),
                          1/2*np.cos(t/2)*np.sin(pi*t/5)+pi/5*np.sin(t/2)*np.cos(pi*t/5),
                          -1])
        return pos_des, dref

    def step(self):
        env_info, done = self.update()

        # # Stop condition
        # ang = np.array(quat2angle(self.plant.quat.state))
        # for ai in ang:
        #     if abs(ai) > np.deg2rad(45):
        #         done = True
        # omega = self.plant.omega.state
        # for dang in omega:
        #     if abs(dang) > np.deg2rad(150):
        #         done = True
        # pos = self.plant.pos.state
        # for p in pos:
        #     if abs(p) > 5:
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
        ref, dref = self.get_ref(t)
        W = self.get_W(t)
        What = self.get_W(t-self.delay)
        # windvel = self.get_windvel(t)

        # Outer-Loop: virtual input
        q = np.vstack([
            self.blf_x.get_control(ref[0], dref[0]),
            self.blf_y.get_control(ref[1], dref[1]),
            self.blf_z.get_control(ref[2], dref[2])
        ])

        # Inverse solution
        u1_cmd = self.plant.m * (q[0]**2 + q[1]**2 + (q[2]-self.plant.g)**2)**(1/2)
        phid = np.clip(np.arcsin(q[1] * self.plant.m / u1_cmd),
                       - np.deg2rad(45), np.deg2rad(45))
        thetad = np.clip(np.arctan(q[0] / (q[2] - self.plant.g)),
                         - np.deg2rad(45), np.deg2rad(45))
        # phid = thetad = 0
        psid = 0
        eulerd = np.vstack([phid, thetad, psid])

        pos = self.plant.pos.state
        ang = quat2angle(self.plant.quat.state)[::-1]

        # Inner-Loop
        u2 = self.blf_phi.get_control(eulerd[0], 0)
        u3 = self.blf_theta.get_control(eulerd[1], 0)
        u4 = self.blf_psi.get_control(eulerd[2], 0)

        # Saturation u1
        u1 = np.clip(u1_cmd, 0, self.plant.rotor_max*self.n)

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

        obs_pos = np.vstack([
            self.blf_x.get_obs(),
            self.blf_y.get_obs(),
            self.blf_z.get_obs(),
        ])
        obs_ang = np.vstack([
            self.blf_phi.get_obs(),
            self.blf_theta.get_obs(),
            self.blf_psi.get_obs()
        ])
        dist = np.vstack([
            self.blf_x.get_dist(),
            self.blf_y.get_dist(),
            self.blf_z.get_dist(),
            self.blf_phi.get_dist(),
            self.blf_theta.get_dist(),
            self.blf_psi.get_dist(),
        ])

        # caculate f
        J = np.diag(self.plant.J)
        p_, q_, r_ = self.plant.omega.state
        f = np.array([(J[1]-J[2]) / J[0] * q_ * r_,
                      (J[2]-J[0]) / J[1] * p_ * r_,
                      (J[0]-J[1]) / J[2] * p_ * q_])

        # set_dot
        self.plant.set_dot(t, rotors,
                           prev_rotors=self.prev_rotors
                           )
        self.blf_x.set_dot(t, pos[0], ref[0], dref[0])
        self.blf_y.set_dot(t, pos[1], ref[1], dref[1])
        self.blf_z.set_dot(t, pos[2], ref[2], dref[2])
        self.blf_phi.set_dot(t, ang[0], eulerd[0], 0)
        self.blf_theta.set_dot(t, ang[1], eulerd[1], 0)
        self.blf_psi.set_dot(t, ang[2], eulerd[2], 0)

        env_info = {
            "t": t,
            "x": self.plant.observe_dict(),
            "What": What,
            "rotors": rotors,
            "rotors_cmd": rotors_cmd,
            "W": W,
            "ref": ref,
            "virtual_u": forces,
            "q": q,
            "eulerd": eulerd,
            "model_uncert_vel": model_uncert_vel,
            "model_uncert_omega": model_uncert_omega,
            "int_uncert_vel": int_vel,
            "dist": dist,
            "f": f,
            "obs_pos": obs_pos,
            "obs_ang": obs_ang,
        }
        return env_info


def run_ray(Kxy, Kz, Kang):
    env = Env(Kxy, Kz, Kang)
    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            break

    time = env.clock.get()
    env.close()

    return time


def run(loggerpath, config):
    env = Env(config)
    env.logger = fym.Logger(loggerpath)
    env.logger.set_info(cfg=ftc.config.load())

    env.reset()

    try:
        while True:
            env.render()
            done, env_info = env.step()
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
        exp_plot(loggerpath, False)


def main(args):
    loggerpath = "data.h5"
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
            "k11": tune.uniform(1, 400),
            "k12": tune.uniform(1, 400),
            "k13": tune.uniform(1, 100),
            "k21": tune.uniform(1, 400),
            "k22": tune.uniform(1, 400),
            "k23": tune.uniform(1, 100),
        }
        current_best_params = [{
            "k11": 2,
            "k12": 30,
            "k13": 5/30/(0.5)**2,
            "k21": 500/30,
            "k22": 30,
            "k23": 5/30/np.deg2rad(45)**2,
        }]
        search = HyperOptSearch(
            metric="tf",
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
                num_samples=10000,
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
        exp_plot(loggerpath, False)
    else:
        kpos, kang = get_PID_gain(cfg.agents.BLF)
        params = {
            "k11": kpos[0],
            "k12": kpos[1],
            "k13": kpos[2],
            "k21": kang[0],
            "k22": kang[1],
            "k23": kang[2],
        }
        # params = {
        #     "k11": 1,
        #     "k12": 1,
        #     "k13": 0,
        #     "k21": 10,
        #     "k22": 10,
        #     "k23": 0,
        # }
        run(loggerpath, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--with-plot", action="store_true")
    parser.add_argument("-r", "--with-ray", action="store_true")
    args = parser.parse_args()
    main(args)
