import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging
from fym.utils.rot import angle2quat, quat2angle

import ftc.config
from ftc.models.multicopter import Multicopter
from ftc.agents.CA import CA
import ftc.agents.fl_miae as fl
import ftc.agents.geso as geso
from ftc.agents.param import get_b0
from ftc.plotting import exp_plot
from copy import deepcopy
from ftc.faults.actuator import LoE
from ftc.faults.manager import LoEManager
cfg = ftc.config.load()


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.05, max_t=20)
        # super().__init__(solver="odeint", max_t=20, dt=10, ode_step_len=100)
        init = cfg.models.multicopter.init
        self.plant = Multicopter(init.pos, init.vel, init.quat, init.omega)
        n = self.plant.mixer.B.shape[1]

        # Define actuator dynamics
        # self.act_dyn = ActuatorDynamcs(tau=0.01, shape=(n, 1))

        # Define faults
        self.sensor_faults = []
        self.fault_manager = LoEManager([
            LoE(time=5, index=0, level=0.1),  # scenario a
            # LoE(time=6, index=2, level=0.8),  # scenario b
        ], no_act=n)

        # Define FDI
        self.fdi = self.fault_manager.fdi

        # Define agents
        self.CA = CA(self.plant.mixer.B)
        self.controller = fl.FLController(self.plant.m,
                                          self.plant.g,
                                          self.plant.J)
        L = np.array([37, 568.25, 4639.5, 21249.8, 51792.5, 52500])[:, None]
        Lpsi = np.array([26, 251, 1066, 1680])[:, None]
        # L = np.array([120, 6000, 160000, 2400000, 19200000, 64000000])[:, None]
        # Lpsi = np.array([80, 2400, 32000, 160000])[:, None]
        self.geso_x = geso.GESO_pos(L)
        self.geso_y = geso.GESO_pos(L)
        self.geso_z = geso.GESO_pos(L)
        self.geso_psi = geso.GESO_psi(Lpsi)
        self.detection_time = self.fault_manager.fault_times + self.fdi.delay

        # Set references
        pos_des = np.vstack([-5, 4, 5])
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
        dist = np.zeros((4, 1))
        observation = np.zeros((4, 1))
        dist[0], observation[0] = self.geso_x.get_dist_obs(self.plant.pos.state[0])
        dist[1], observation[1] = self.geso_y.get_dist_obs(self.plant.pos.state[1])
        dist[2], observation[2] = self.geso_z.get_dist_obs(self.plant.pos.state[2])
        c_psi = quat2angle(self.plant.quat.state)[::-1][2]  # current psi
        dist[3], observation[3] = self.geso_psi.get_dist_obs(c_psi)

        # Controller
        virtual_ctrl = self.controller.get_virtual(t,
                                                   self.plant,
                                                   ref,
                                                   # disturbance=dist
                                                   )

        forces = self.controller.get_FM(virtual_ctrl)
        rotors_cmd = self.CA.get(What).dot(forces)
        # rotors_cmd = np.linalg.pinv(self.plant.mixer.B).dot(forces)

        # actuator saturation
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        # Set actuator faults
        rotors = self.fault_manager.get_faulty_input(t, rotors)

        self.plant.set_dot(t, rotors)
        self.controller.set_dot(virtual_ctrl)
        v_eso = np.vstack([virtual_ctrl[0], virtual_ctrl[1]])
        self.geso_x.set_dot(t, self.plant.pos.state[0], v_eso[0][0])
        self.geso_y.set_dot(t, self.plant.pos.state[1], v_eso[1][0])
        self.geso_z.set_dot(t, self.plant.pos.state[2], v_eso[2][0])
        self.geso_psi.set_dot(t, c_psi, v_eso[3][0])

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref,
                    virtual_u=forces, observation=observation, dist=dist)


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
