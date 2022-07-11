import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging
from fym.utils.rot import angle2quat, quat2angle

import ftc.config
from ftc.models.multicopter import Multicopter
from ftc.agents.CA import CA
import ftc.agents.leso as leso
import ftc.agents.mismatch as mm
from ftc.agents.param import get_b0, get_B_hat
from ftc.plotting_error import exp_plot
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
        super().__init__(dt=0.001, max_t=10)
        init_pos = np.vstack((0, 0, 0))
        # init_ang = np.deg2rad([20, 30, 10])*(np.random.rand(3) - 0.5)
        # init_quat = (angle2quat(init_ang[2], init_ang[1], init_ang[0]))
        self.vel_base = np.random.randn(3, 1)
        self.plant = Multicopter(
            pos=init_pos,
            vel=np.zeros((3, 1)),
            quat=np.vstack([1, 0, 0, 0]),
            omega=np.zeros((3, 1)),
            # uncertainty=0.1,
        )
        # init = cfg.models.multicopter.init
        # self.plant = Multicopter(init.pos, init.vel, init.quat, init.omega)
        n = self.plant.mixer.B.shape[1]

        # Define actuator dynamics
        # self.act_dyn = ActuatorDynamcs(tau=0.01, shape=(n, 1))

        # Define faults
        self.sensor_faults = []
        self.fault_manager = LoEManager([
            # LoE(time=3, index=0, level=0.5),  # scenario a
            # LoE(time=6, index=2, level=0.8),  # scenario b
        ], no_act=n)

        # Define FDI
        self.fdi = self.fault_manager.fdi

        # Define agents
        self.CA = CA(self.plant.mixer.B)
        self.controller = leso.Controller(self.plant.m,
                                          self.plant.g)
        self.Bhat = get_B_hat(self.plant.m, self.plant.g, self.plant.J)
        b = get_b0(self.plant.m, self.plant.g, self.plant.J)
        # r = 5
        K = np.array([5, 10, 10, 5, 1])
        K_psi = np.array([3, 3, 1])
        # K = np.array([-35, -470, -3010, -9129, -10395])
        # K_psi = np.array([-15, -71, -105])
        # alp = np.array([-1, -4, -6, -4])
        alp = np.array([-16, -32, -24, -8])
        alp_psi = np.vstack([-4, -2])
        self.ux = mm.mismatchESO(4, b[0], 80, K, 1, alp, 1, 0.9)
        self.uy = mm.mismatchESO(4, b[1], 80, K, 1, alp, 1, 0.9)
        self.uz = mm.mismatchESO(4, b[2], 80, K, 55, alp, 1, 0.9)
        self.upsi = mm.mismatchESO(2, b[3], 80, K_psi, 1, alp_psi, 1, 0.7)

        self.detection_time = self.fault_manager.fault_times + self.fdi.delay
        self.u = np.zeros((4, 1))

        # nu = np.array([self.plant.m*self.plant.g, 0, 0, 0])[:, None]
        # self.u = np.linalg.pinv(self.plant.mixer.B).dot(nu)

    def get_ref(self, t):
        # Set references
        pos_des = np.vstack([1, 1, 1])
        vel_des = np.vstack([0, 0, 0])
        # pos_des = np.vstack([np.cos(t), np.sin(t), t])
        # vel_des = np.vstack([-np.sin(t), np.cos(t), 1])
        quat_des = np.vstack([1, 0, 0, 0])
        omega_des = np.vstack([0, 0, 0])
        ref = np.vstack([pos_des, vel_des, quat_des, omega_des])
        return ref

    def step(self):
        *_, done = self.update()
        return done

    def get_obs_ref(self, t):
        ref = self.get_ref(t)
        posd = ref[0:3]
        psid = quat2angle(ref[6:10])[::-1][2]
        obs_ref = np.vstack([posd, psid])
        return obs_ref

    def get_Bp(self, What):
        # ind = np.nonzero(What < 1)[0][0]
        eta = np.diag(What)
        d = self.plant.d
        B = np.array(
            [[eta[0], eta[1], eta[2], eta[3], eta[4], eta[5]],
             [-eta[0]*d, eta[1]*d, eta[2]*d/2, -eta[3]*d/2, -eta[4]*d/2,
              eta[5]*d/2],
             [0, 0, eta[2]*d*np.sqrt(3)/2, -eta[3]*d*np.sqrt(3)/2,
              eta[4]*d*np.sqrt(3)/2, -eta[5]*d*np.sqrt(3)/2]]
        )
        return B

    def set_dot(self, t):
        ref = self.get_ref(t)
        W = self.fdi.get_true(t)
        What = self.fdi.get(t)
        # windvel = self.get_windvel(t)

        # Observer
        obs_ctrl = np.zeros((4, 1))  # u_star
        obs_ref = self.get_obs_ref(t)
        obs_ctrl[2] = self.ux.get_virtual()
        obs_ctrl[1] = self.uy.get_virtual()
        obs_ctrl[0] = self.uz.get_virtual()
        obs_ctrl[3] = self.upsi.get_virtual()
        # TODO: can I use B0 instead of B, the paper says no, but if can,
        # I can use this for my paper
        # obs_ctrl = np.linalg.inv(self.Bhat).dot(obs_ctrl)  # u
        self.u = obs_ctrl

        # error observation
        err_obs = np.zeros((4, 1))
        err_obs[0] = self.ux.get_obs()
        err_obs[1] = self.uy.get_obs()
        err_obs[2] = self.uz.get_obs()
        err_obs[3] = self.upsi.get_obs()

        # total disturbance
        dist = np.zeros((4, 1))
        dist[0] = self.ux.get_dist()
        dist[1] = self.uy.get_dist()
        dist[2] = self.uz.get_dist()
        dist[3] = self.upsi.get_dist()

        # Controller
        virtual_ctrl = self.controller.get_virtual(t, obs_ctrl)

        forces = self.controller.get_FM(virtual_ctrl)
        # rotors_cmd = self.CA.get(What).dot(forces)
        rotors_cmd = np.linalg.pinv(self.plant.mixer.B).dot(forces)
        # if t < self.detection_time[1]:
        #     Bp = self.plant.mixer.B
        #     fm = forces
        # else:
        #     Bp = self.plant.mixer.B[0:3, :]
        #     fm = forces[0:3, :]
        # W1, W2 = np.eye(6) - What, np.eye(6)
        # W = W1 + W2
        # H = Bp.dot(np.linalg.inv(W).dot(Bp.T))
        # G = Bp.T.dot(np.linalg.inv(H))
        # F_ = np.eye(6) - Bp.T.dot(np.linalg.inv(H).dot(Bp.dot(np.linalg.inv(W).dot(W2))))
        # F = np.linalg.inv(W).dot(F_)
        # rotors_cmd = F.dot(self.u) + G.dot(fm)

        # actuator saturation
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        # Set actuator faults
        rotors = self.fault_manager.get_faulty_input(t, rotors)
        # self.u = rotors

        self.plant.set_dot(t, rotors,
                           # windvel
                           )
        self.controller.set_dot(virtual_ctrl)

        y = np.vstack([self.plant.pos.state, quat2angle(self.plant.quat.state)[::-1][2]])
        err_real = y - obs_ref
        self.ux.set_dot(t, y[0], obs_ref[0], self.u)
        self.uy.set_dot(t, y[1], obs_ref[1], self.u)
        self.uz.set_dot(t, y[2], obs_ref[2], self.u)
        self.upsi.set_dot(t, y[3], obs_ref[3], self.u)

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref,
                    obs_u=obs_ctrl, virtual_u=forces, dist=dist,
                    err_obs=err_obs, err=err_real)

    def get_windvel(self, t):
        windvel = self.vel_base * (
            np.sin(1.5*np.pi*t - 3)
            + 1.5 * np.sin(np.pi*t + 7)
            + 0.5 * np.sin(0.4*np.pi*t - 9.5)
        )
        return windvel


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
