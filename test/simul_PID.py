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
from ftc.agents.CA import CA
import ftc.agents.PID as PID
from ftc.agents.param import get_b0, get_W, get_faulty_input, get_PID_gain
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
    def __init__(self, kpos, kang):
        super().__init__(dt=0.01, max_t=20)
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

        # Define agents
        self.CA = CA(self.plant.mixer.B)
        params = cfg.agents.BLF
        self.pos_ref = np.vstack([-0, 0, 0])
        kP, kD, kI = kpos.ravel()
        self.blf_x = PID.PIDController(params.oL.alp, params.oL.eps[0], params.theta,
                                       -self.pos_ref[0][0], kP, kD, kI, "pos")
        self.blf_y = PID.PIDController(params.oL.alp, params.oL.eps[1], params.theta,
                                       -self.pos_ref[1][0], kP, kD, kI, "pos")
        self.blf_z = PID.PIDController(params.oL.alp, params.oL.eps[2], params.theta,
                                       -self.pos_ref[2][0], kP, kD, kI, "pos")
        kP, kD, kI = kang.ravel()
        self.blf_phi = PID.PIDController(params.oL.alp, params.iL.eps[0], params.theta,
                                         0, kP, kD, kI, "ang")
        self.blf_theta = PID.PIDController(params.oL.alp, params.iL.eps[1], params.theta,
                                           0, kP, kD, kI, "ang")
        self.blf_psi = PID.PIDController(params.oL.alp, params.iL.eps[2], params.theta,
                                         0, kP, kD, kI, "ang")

        self.prev_rotors = np.zeros((4, 1))

    def get_ref(self, t):
        pos_des = self.pos_ref
        # pos_des = np.vstack([np.sin(t/2)*np.cos(np.pi*t/5),
        #                      np.sin(t/2)*np.sin(np.pi*t/5),
        #                      -t])
        return pos_des

    def step(self):
        *_, done = self.update()

        # Stop condition
        # omega = self.plant.omega.state
        # for dang in omega:
        #     if abs(dang) > np.deg2rad(80):
        #         done = True
        # err_pos = np.array([self.blf_x.e.state[0],
        #                     self.blf_y.e.state[0],
        #                     self.blf_z.e.state[0]])
        # for err in err_pos:
        #     if abs(err) > 10:
        #         done = True

        for rotor in self.prev_rotors:
            if rotor > self.plant.rotor_max + 5:
                done = True

        return done

    def get_faultBias(self, t):
        if self.fault_bias is True:
            delta = np.diag([0.1*np.cos(np.pi*t),
                             0.2*np.sin(5/2*t),
                             0.3/(1+np.exp(-t)),
                             0.1*np.sin(0.5*t)])
        else:
            delta = np.zeros((4, 4))
        return delta

    def set_dot(self, t):
        ref = self.get_ref(t)
        W = get_W(t, self.fault)
        What = get_W(t-self.delay, self.fault)
        # windvel = self.get_windvel(t)

        # Outer-Loop: virtual input
        q = np.zeros((3, 1))
        q[0] = self.blf_x.get_control(0)
        q[1] = self.blf_y.get_control(0)
        q[2] = self.blf_z.get_control(0)

        # Inverse solution
        u1_cmd = self.plant.m * (q[0]**2 + q[1]**2 + (q[2]-self.plant.g)**2)**(1/2)
        phid = np.clip(np.arcsin(q[1] * self.plant.m / u1_cmd),
                       - np.deg2rad(45), np.deg2rad(45))
        thetad = np.clip(np.arctan(q[0] / (q[2] - self.plant.g)),
                         - np.deg2rad(45), np.deg2rad(45))
        psid = 0
        eulerd = np.vstack([phid, thetad, psid])

        # Inner-Loop
        u2 = self.blf_phi.get_control(phid)
        u3 = self.blf_theta.get_control(thetad)
        u4 = self.blf_psi.get_control(psid)

        # Saturation u1
        u1 = np.clip(u1_cmd, 0, self.plant.rotor_max*self.n)

        # rotors
        forces = np.vstack([u1, u2, u3, u4])
        # rotors_cmd = np.linalg.pinv(self.plant.mixer.B).dot(forces)
        rotors_cmd = self.CA.get(What).dot(forces)
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
        obs_pos[0] = self.blf_x.get_obs()
        obs_pos[1] = self.blf_y.get_obs()
        obs_pos[2] = self.blf_z.get_obs()
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
                    virtual_u=forces, dist=dist, q=q, f=f,
                    obs_pos=obs_pos, obs_ang=obs_ang, eulerd=eulerd,
                    model_uncert_vel=model_uncert_vel,
                    model_uncert_omega=model_uncert_omega)


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


def run(loggerpath, Kxy, Kz, Kang):
    env = Env(Kxy, Kz, Kang)
    env.logger = fym.Logger(loggerpath)
    env.logger.set_info(cfg=ftc.config.load())

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            env_info = {
                # "detection_time": env.detection_time,
                "rotor_min": env.plant.rotor_min,
                "rotor_max": env.plant.rotor_max,
            }
            env.logger.set_info(**env_info)
            break

    env.close()


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
                    tf = env.info["t"]

                    if done:
                        break

            finally:
                return {"tf": tf}

        config = {
            "k11": tune.uniform(1, 400),
            "k12": tune.uniform(1, 400),
            "k13": tune.uniform(1, 400),
            "k21": tune.uniform(1, 400),
            "k22": tune.uniform(1, 400),
            "k23": tune.uniform(1, 400),
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
        exp_plot(loggerpath, False)
    else:
        params = {
            "k11": cfg.agents.BLF.Kxy[0],
            "k12": cfg.agents.BLF.Kxy[1],
            "k13": cfg.agents.BLF.Kxy[2],
            "k21": cfg.agents.BLF.Kang[0],
            "k22": cfg.agents.BLF.Kang[1],
            "k23": cfg.agents.BLF.Kang[2],
        }
        kpos, kang = get_PID_gain(params)
        run(loggerpath, kpos, kang)
        exp_plot(loggerpath, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--with-plot", action="store_true")
    parser.add_argument("-r", "--with-ray", action="store_true")
    args = parser.parse_args()
    main(args)
