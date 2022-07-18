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
import ftc.agents.BLF_pf as BLF
from ftc.agents.param import get_b0
from ftc.plotting import exp_plot
from copy import deepcopy
from ftc.faults.actuator import LoE
from ftc.faults.manager import LoEManager

plt.rc("text", usetex=False)
plt.rc("lines", linewidth=1)
plt.rc("axes", grid=True)
plt.rc("grid", linestyle="--", alpha=0.8)

cfg = ftc.config.load()


class Env(BaseEnv):
    def __init__(self, k11, k12, k21, k22, k31, k32):
        super().__init__(dt=0.01, max_t=10)
        init = cfg.models.multicopter.init
        self.plant = Multicopter(init.pos, init.vel, init.quat, init.omega)
        self.n = self.plant.mixer.B.shape[1]

        # Define actuator dynamics
        # self.act_dyn = ActuatorDynamcs(tau=0.01, shape=(n, 1))

        # Define faults
        self.sensor_faults = []
        self.fault_manager = LoEManager([
            # LoE(time=5, index=0, level=0.9),  # scenario a
            # LoE(time=6, index=2, level=0.8),  # scenario b
        ], no_act=self.n)

        # Define FDI
        self.fdi = self.fault_manager.fdi

        # Define agents
        self.CA = CA(self.plant.mixer.B)
        l = 3
        alp = np.array([23, 23, 23])
        bet = np.array([80.3686, 30.0826])
        Kxy = np.array([k11, k12])
        Kz = np.array([k21, k22])
        Rxy = np.array([10, 100])
        Rz = np.array([10, 30])
        rho_0, rho_inf = 15, 1e-1
        k = 0.2
        self.blf_x = BLF.outerLoop(l, alp, bet, Rxy, Kxy, rho_0, rho_inf, k)
        self.blf_y = BLF.outerLoop(l, alp, bet, Rxy, Kxy, rho_0, rho_inf, k)
        self.blf_z = BLF.outerLoop(l, alp, bet, Rz, Kz, rho_0, rho_inf, k)
        xi = np.array([-1, 1])
        rho = np.deg2rad([30, 80])
        c = np.array([20, 20])
        J = np.diag(self.plant.J)
        b = np.array([1/J[0], 1/J[1], 1/J[2]])
        Rang = np.array([np.deg2rad(80), 100])
        # Kang = np.array([20, 15])  # for rotor failure case
        Kang = np.array([k31, k32])
        self.blf_phi = BLF.innerLoop(l, alp, bet, Rang, Kang, xi,
                                     rho, c, b[0], self.plant.g)
        self.blf_theta = BLF.innerLoop(l, alp, bet, Rang, Kang, xi,
                                       rho, c, b[1], self.plant.g)
        self.blf_psi = BLF.innerLoop(l, alp, bet, Rang, Kang, xi,
                                     rho, c, b[2], self.plant.g)

        self.detection_time = self.fault_manager.fault_times + self.fdi.delay
        self.rotors_cmd = np.zeros((6, 1))

    def get_ref(self, t):
        pos_des = np.vstack([-0, 1, 0])
        vel_des = np.vstack([0, 0, 0])
        # pos_des = np.vstack([np.sin(t), np.cos(t), -t])
        # vel_des = np.vstack([np.cos(t), -np.sin(t), -1])
        quat_des = np.vstack([1, 0, 0, 0])
        omega_des = np.vstack([0, 0, 0])
        ref = np.vstack([pos_des, vel_des, quat_des, omega_des])
        return ref

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
        W = self.fdi.get_true(t)
        What = self.fdi.get(t)
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

        # caculate f
        J = np.diag(self.plant.J)
        obs_p = self.blf_phi.get_obsdot()
        obs_q = self.blf_theta.get_obsdot()
        obs_r = self.blf_psi.get_obsdot()
        f = np.array([(J[1]-J[2]) / J[0] * obs_q * obs_r,
                      (J[2]-J[0]) / J[1] * obs_p * obs_r,
                      (J[0]-J[1]) / J[2] * obs_p * obs_q])

        # Inner-Loop
        u2 = self.blf_phi.get_u(phid, f[0])
        u3 = self.blf_theta.get_u(thetad, f[1])
        u4 = self.blf_psi.get_u(psid, f[2])

        # Saturation u1
        u1 = np.clip(u1_cmd, 0, self.plant.rotor_max*self.n)

        # rotors
        forces = np.vstack([u1, u2, u3, u4])
        rotors_cmd = np.linalg.pinv(self.plant.mixer.B).dot(forces)
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)
        self.rotors_cmd = rotors_cmd

        # Set actuator faults
        rotors = self.fault_manager.get_faulty_input(t, rotors)

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
                           )
        x, y, z = self.plant.pos.state.ravel()
        euler = quat2angle(self.plant.quat.state)[::-1]
        self.blf_x.set_dot(t, x, ref[0])
        self.blf_y.set_dot(t, y, ref[1])
        self.blf_z.set_dot(t, z, ref[2])
        self.blf_phi.set_dot(t, euler[0], phid, f[0])
        self.blf_theta.set_dot(t, euler[1], thetad, f[1])
        self.blf_psi.set_dot(t, euler[2], psid, f[2])

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref,
                    virtual_u=forces, dist=dist,
                    obs_pos=obs_pos, obs_ang=obs_ang, eulerd=eulerd)


def run_ray(k11, k12, k21, k22, k31, k32):
    env = Env(k11, k12, k21, k22, k31, k32)
    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            break

    time = env.clock.get()
    env.close()

    return time


def run(loggerpath, k11, k12, k21, k22, k31, k32):
    env = Env(k11, k12, k21, k22, k31, k32)
    env.logger = fym.Logger(loggerpath)
    env.logger.set_info(cfg=ftc.config.load())

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            env_info = {
                "detection_time": env.detection_time,
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

        k11, k12, k21, k22, k31, k32 = 1, 1, 3, 2, 10, 10
        run(loggerpath, k11, k12, k21, k22, k31, k32)
        exp_plot(loggerpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--with-ray", action="store_true")
    args = parser.parse_args()
    main(args)
