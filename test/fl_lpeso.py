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
import ftc.agents.lpeso as lpeso
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
    def __init__(self):
        super().__init__(dt=0.05, max_t=40)
        init_pos = np.vstack((0, 0, 0))
        # init_ang = np.deg2rad([20, 30, 10])*(np.random.rand(3) - 0.5)
        # init_quat = (angle2quat(init_ang[2], init_ang[1], init_ang[0]))
        self.plant = Multicopter(
            pos=init_pos,
            vel=np.zeros((3, 1)),
            quat=np.vstack([1, 0, 0, 0]),
            omega=np.zeros((3, 1)),
        )
        # init = cfg.models.multicopter.init
        # self.plant = Multicopter(init.pos, init.vel, init.quat, init.omega)
        n = self.plant.mixer.B.shape[1]

        # Define actuator dynamics
        # self.act_dyn = ActuatorDynamcs(tau=0.01, shape=(n, 1))

        # Define faults
        self.sensor_faults = []
        self.fault_manager = LoEManager([
            LoE(time=3, index=0, level=0.5),  # scenario a
            LoE(time=6, index=2, level=0.8),  # scenario b
        ], no_act=n)

        # Define FDI
        self.fdi = self.fault_manager.fdi

        # Define agents
        self.CA = CA(self.plant.mixer.B)
        self.controller = fl.FLController(self.plant.m,
                                          self.plant.g,
                                          self.plant.J)
        b0 = get_b0(self.plant.m, self.plant.g, self.plant.J)
        self.b0 = b0
        K = np.array([[23, 484],
                      [23, 183.1653],
                      [23, 80.3686],
                      [23, 30.0826]])
        Lx, Ly, Lz, Lpsi = 1., 1., 55., 1.
        self.lpeso_x = lpeso.lowPowerESO(4, 2, K, b0[0], self.controller.F[0, :], Lx)
        self.lpeso_y = lpeso.lowPowerESO(4, 2, K, b0[1], self.controller.F[1, :], Ly)
        self.lpeso_z = lpeso.lowPowerESO(4, 2, K, b0[2], self.controller.F[2, :], Lz)
        H = np.array([[3, 3, 1]])
        # H = np.array([[30, 291, 910]])
        self.hgeso_psi = lpeso.highGainESO(0.5, H, b0[3], self.controller.F[3, 0:2], Lpsi)

        self.detection_time = self.fault_manager.fault_times + self.fdi.delay

        # Set references
        pos_des = np.vstack([-2, 5, 4])
        vel_des = np.vstack([0, 0, 0])
        quat_des = np.vstack([1, 0, 0, 0])
        omega_des = np.vstack([0, 0, 0])
        self.ref = np.vstack([pos_des, vel_des, quat_des, omega_des])

    def step(self):
        *_, done = self.update()
        return done

    def get_obs_ref(self):
        posd = self.ref[0:3]
        veld = np.zeros((3, 1))
        dveld = np.zeros((3, 1))
        ddveld = np.zeros((3, 1))
        psid = quat2angle(self.ref[6:10])[::-1][2]
        dpsid = 0

        obs_ref = np.zeros((4, 4, 1))
        for i in range(3):
            obs_ref[i, :, :] = np.array([posd[i], veld[i],
                                         dveld[i], ddveld[i]])
        obs_ref[3, 0:2, :] = np.array([psid, dpsid])[:, None]
        return obs_ref

    def set_dot(self, t):
        W = self.fdi.get_true(t)
        What = self.fdi.get(t)

        # Observer
        obs_ctrl = np.zeros((4, 1))
        obs_ref = self.get_obs_ref()
        obs_ctrl[2] = self.lpeso_x.get_virtual(t, obs_ref[0, :, :])
        obs_ctrl[1] = self.lpeso_y.get_virtual(t, obs_ref[1, :, :])
        obs_ctrl[0] = self.lpeso_z.get_virtual(t, obs_ref[2, :, :])
        obs_ctrl[3] = self.hgeso_psi.get_virtual(obs_ref[3, 0:2, :])

        # dist = np.zeros((4, 1))
        observation = np.zeros((4, 1))
        # dist[2], observation[0] = self.lpeso_x.get_dist_obs(t, y[0])
        # dist[1], observation[1] = self.lpeso_y.get_dist_obs(t, y[1])
        # dist[0], observation[2] = self.lpeso_z.get_dist_obs(t, y[2])
        observation[2], observation[3] = self.hgeso_psi.get_obs()
        # dist[3], observation[3] = self.geso_psi.get_dist_obs()

        # Controller
        virtual_ctrl = self.controller.get_virtual(t,
                                                   self.plant,
                                                   self.ref,
                                                   obs_u=obs_ctrl
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

        y = np.vstack([self.plant.pos.state, quat2angle(self.plant.quat.state)[::-1][2]])
        self.lpeso_x.set_dot(t, y[0], obs_ref[0, :, :])
        self.lpeso_y.set_dot(t, y[1], obs_ref[1, :, :])
        self.lpeso_z.set_dot(t, y[2], obs_ref[2, :, :])
        self.hgeso_psi.set_dot(t, y[3], obs_ref[3, 0:2, :])

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=self.ref,
                    obs_u=obs_ctrl, virtual_u=forces, obs=observation)


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
