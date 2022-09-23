import ray
from ray import tune
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
import ftc.agents.BLF_proposed as BLF
from ftc.agents.param import get_b0, get_W, get_faulty_input
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
    def __init__(self, Kxy, Kz, Kang):
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
        self.fault = True
        self.delay = cfg.faults.manager.delay

        # Define agents
        self.CA = CA(self.plant.mixer.B)
        params = cfg.agents.BLF.pf
        self.pos_ref = np.vstack([-0, 0, 0])
        self.blf_x = BLF.outerLoop(params.oL.l, params.oL.alp, params.oL.bet,
                                   params.oL.R, Kxy, params.oL.rho,
                                   params.oL.rho_k, cond.noise,
                                   -self.pos_ref[0][0], params.theta)
        self.blf_y = BLF.outerLoop(params.oL.l, params.oL.alp, params.oL.bet,
                                   params.oL.R, Kxy, params.oL.rho,
                                   params.oL.rho_k, cond.noise,
                                   -self.pos_ref[1][0], params.theta)
        self.blf_z = BLF.outerLoop(params.oL.l, params.oL.alp, params.oL.bet,
                                   params.oL.R, Kz, params.oL.rho,
                                   params.oL.rho_k, cond.noise,
                                   -self.pos_ref[2][0], params.theta)
        J = np.diag(self.plant.J)
        b = np.array([1/J[0], 1/J[1], 1/J[2]])
        self.blf_phi = BLF.innerLoop(params.iL.l, params.iL.alp, params.iL.bet,
                                     params.iL.dist_range, Kang, params.iL.xi,
                                     params.iL.rho, params.iL.c, b[0],
                                     self.plant.g, params.theta, cond.noise)
        self.blf_theta = BLF.innerLoop(params.iL.l, params.iL.alp, params.iL.bet,
                                       params.iL.dist_range, Kang, params.iL.xi,
                                       params.iL.rho, params.iL.c, b[0],
                                       self.plant.g, params.theta, cond.noise)
        self.blf_psi = BLF.innerLoop(params.iL.l, params.iL.alp, params.iL.bet,
                                     params.iL.dist_range, Kang, params.iL.xi,
                                     params.iL.rho, params.iL.c, b[0],
                                     self.plant.g, params.theta, cond.noise)

        self.prev_rotors = np.zeros((4, 1))

    def get_ref(self, t):
        # pos_des = self.pos_ref
        pos_des = np.vstack([np.sin(t/2)*np.cos(np.pi*t/5),
                             np.sin(t/2)*np.sin(np.pi*t/5),
                             -t])
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

        # for rotor in self.rotors_cmd:
        #     if rotor < 0 or rotor > self.plant.rotor_max + 5:
        #         done = True

        return done

    def set_dot(self, t):
        ref = self.get_ref(t)
        W = get_W(t, self.fault)
        What = get_W(t-self.delay, self.fault)
        # windvel = self.get_windvel(t)

        # Outer-Loop: virtual input
        q = np.zeros((3, 1))
        q[0] = self.blf_x.get_virtual(t)
        q[1] = self.blf_y.get_virtual(t)
        q[2] = self.blf_z.get_virtual(t)

        # Inverse solution
        u1_cmd = self.plant.m * (q[0]**2 + q[1]**2 + (q[2]-self.plant.g)**2)**(1/2)
        phid = np.arcsin(q[1] * self.plant.m / u1_cmd)
        thetad = np.arctan(q[0] / (q[2] - self.plant.g))
        psid = 0
        eulerd = np.vstack([phid, thetad, psid])

        # Inner-Loop
        u2 = self.blf_phi.get_u(t, phid)
        u3 = self.blf_theta.get_u(t, thetad)
        u4 = self.blf_psi.get_u(t, psid)

        # Saturation u1
        # u1 = np.clip(u1_cmd, 0, self.plant.rotor_max*self.n)
        u1 = u1_cmd

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
                    obs_pos=obs_pos, obs_ang=obs_ang, eulerd=eulerd)


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
    if args.with_ray:
        configs = {
            "k11": tune.uniform(1, 10),
            "k12": tune.uniform(1, 10),
            "k21": tune.uniform(1, 20),
            "k22": tune.uniform(1, 20),
            "k31": tune.uniform(1, 20),
            "k32": tune.uniform(1, 20),
        }

        def trainable(configs):
            score = run_ray(configs["k11"], configs["k12"], configs["k21"],
                            configs["k22"], configs["k31"], configs["k32"])
            tune.report(score=score)
        tune.run(trainable, config=configs, num_samples=10)

    else:
        loggerpath = "data.h5"

        Kxy = cfg.agents.BLF.pf.Kxy.ravel()
        Kz = cfg.agents.BLF.pf.Kz.ravel()
        Kang = cfg.agents.BLF.pf.Kang.ravel()
        run(loggerpath, Kxy, Kz, Kang)
        exp_plot(loggerpath, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--with-ray", action="store_true")
    args = parser.parse_args()
    main(args)
    # comp.exp_plot4("result_blf.h5", "result_blf_g.h5", "result_blf_pf.h5", "result_blf_proposed.h5")
    # exp_plot("result_blf_proposed.h5", True)
