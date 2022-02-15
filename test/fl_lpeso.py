import numpy as np
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging
from fym.utils.rot import angle2quat, quat2angle

import ftc.config
from ftc.models.multicopter import Multicopter
from ftc.agents.CA import CA
import ftc.agents.fl_miae as fl
import ftc.agents.lpeso as lpeso
from ftc.agents.param import get_b0
from ftc.plotting import exp_plot
from copy import deepcopy
from ftc.faults.actuator import LoE
from ftc.faults.manager import LoEManager
cfg = ftc.config.load()


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.1, max_t=10)
        init = cfg.models.multicopter.init
        self.plant = Multicopter(init.pos, init.vel, init.quat, init.omega)
        n = self.plant.mixer.B.shape[1]

        # Define actuator dynamics
        # self.act_dyn = ActuatorDynamcs(tau=0.01, shape=(n, 1))

        # Define faults
        self.sensor_faults = []
        self.fault_manager = LoEManager([
            # LoE(time=3, index=0, level=0.),  # scenario a
            # LoE(time=6, index=2, level=0.),  # scenario b
        ], no_act=n)

        # Define FDI
        self.fdi = self.fault_manager.fdi

        # Define agents
        self.CA = CA(self.plant.mixer.B)
        self.controller = fl.FLController(self.plant.m,
                                          self.plant.g,
                                          self.plant.J)
        b0 = get_b0(self.plant.m, self.plant.g, self.plant.J)
        K = np.array([[24, 492],
                      [24, 183.1341],
                      [24, 80.7314],
                      [24, 30.0029]])
        Kpsi = np.array([[8.5600, 31.7536],
                         [8.4400, 7.5582]])
        Lx, Ly, Lz, Lpsi = 1.025, 1.025, 46., 1.
        # Lx, Ly, Lz, Lpsi = 100, 100, 100, 100
        self.lpeso_x = lpeso.lowPowerESO(4, 1.2, K, b0[0], self.controller.F[0, :], Lx)
        self.lpeso_y = lpeso.lowPowerESO(4, 1.2, K, b0[1], self.controller.F[1, :], Ly)
        self.lpeso_z = lpeso.lowPowerESO(4, 1.2, K, b0[2], self.controller.F[2, :], Lz)
        self.lpeso_psi = lpeso.lowPowerESO(2, 4, Kpsi, b0[3], self.controller.F[3, 0:2], Lpsi)

        self.detection_time = self.fault_manager.fault_times + self.fdi.delay

        # Set references
        pos_des = np.vstack([-1, 1, 2])
        vel_des = np.vstack([0, 0, 0])
        quat_des = np.vstack([1, 0, 0, 0])
        omega_des = np.vstack([0, 0, 0])
        self.ref = np.vstack([pos_des, vel_des, quat_des, omega_des])

    def step(self):
        *_, done = self.update()
        return done

    def get_ref(self, t):
        return self.ref

    def set_dot(self, t):
        W = self.fdi.get_true(t)
        What = self.fdi.get(t)
        ref = self.get_ref(t)

        # Observer
        obs_ctrl = np.zeros((4, 1))
        obs_ctrl[2] = self.lpeso_x.get_virtual(t, self.plant.observe_list()[0][0])
        obs_ctrl[1] = self.lpeso_y.get_virtual(t, self.plant.observe_list()[0][1])
        obs_ctrl[0] = self.lpeso_z.get_virtual(t, self.plant.observe_list()[0][2])
        c_psi = quat2angle(self.plant.observe_list()[2])[::-1][2]  # current psi
        obs_ctrl[3] = self.lpeso_psi.get_virtual(t, c_psi)

        dist = np.zeros((4, 1))
        observation = np.zeros((4, 1))
        dist[0], observation[0] = self.lpeso_x.get_dist_obs(t, self.plant.observe_list()[0][0])
        dist[1], observation[1] = self.lpeso_y.get_dist_obs(t, self.plant.observe_list()[0][1])
        dist[2], observation[2] = self.lpeso_z.get_dist_obs(t, self.plant.observe_list()[0][2])
        dist[3], observation[3] = self.lpeso_psi.get_dist_obs(t, c_psi)

        # Controller
        virtual_ctrl = self.controller.get_virtual(t,
                                                   self.plant,
                                                   ref,
                                                   # disturbance=dist,
                                                   obs_u=obs_ctrl
                                                   )

        forces = self.controller.get_FM(virtual_ctrl)
        rotors_cmd = self.CA.get(What).dot(forces)

        # actuator saturation
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        # Set actuator faults
        rotors = self.fault_manager.get_faulty_input(t, rotors)

        self.plant.set_dot(t, rotors)
        self.controller.set_dot(virtual_ctrl)
        self.lpeso_x.set_dot(t, self.plant.observe_list()[0][0])
        self.lpeso_y.set_dot(t, self.plant.observe_list()[0][1])
        self.lpeso_z.set_dot(t, self.plant.observe_list()[0][2])
        self.lpeso_psi.set_dot(t, c_psi)

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref,
                    obs_u=obs_ctrl, virtual_u=forces, observation=observation,
                    dist=dist)


def run(loggerpath):
    env = Env()
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


def exp1(loggerpath):
    run(loggerpath)


if __name__ == "__main__":
    loggerpath = "data.h5"
    exp1(loggerpath)
    exp_plot(loggerpath)
