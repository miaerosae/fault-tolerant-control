'''
Testing Reduced Dynamics, when complete failure occurs or severe faults occurs
Assume only 1 total failure occurs
'''
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging
from fym.utils.rot import angle2quat, quat2angle

import ftc.config
from ftc.models.multicopter import Multicopter
from ftc.agents.CA import CA, ConstrainedCA
import ftc.agents.lpeso as lpeso
from ftc.agents.param import get_b0, get_B_hat, get_B, get_K
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
        super().__init__(dt=0.01, max_t=20)
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
            LoE(time=3, index=0, level=0.0),  # scenario a
            # LoE(time=6, index=2, level=0.8),  # scenario b
        ], no_act=n)

        # Define FDI
        self.fdi = self.fault_manager.fdi

        # Define agents
        self.CA = CA(self.plant.mixer.B)
        self.controller = lpeso.Controller(self.plant.m,
                                           self.plant.g)
        b0 = get_b0(self.plant.m, self.plant.g, self.plant.J)
        self.Bhat = get_B_hat(self.plant.m, self.plant.g, self.plant.J)
        K = get_K()
        Lx, Ly, Lz, Lpsi = 1., 1., 55., 1.
        Rxy = np.array([10, 20, 50, 50])
        Rz = np.array([10, 20, 50, 5])
        # TODO: 이거 좀 더 작게 줘도 될듯

        self.lpeso_x = lpeso.lowPowerESO(4, 2, K, b0[0],
                                         self.controller.F[0, :], Lx, Rxy)
        self.lpeso_y = lpeso.lowPowerESO(4, 2, K, b0[1],
                                         self.controller.F[1, :], Ly, Rxy)
        self.lpeso_z = lpeso.lowPowerESO(4, 2, K, b0[2],
                                         self.controller.F[2, :], Lz, Rz)
        H = np.array([[3, 3, 1]])
        self.hgeso_psi = lpeso.highGainESO(0.5, H, b0[3],
                                           self.controller.F[3, 0:2], Lpsi)

        self.detection_time = self.fault_manager.fault_times + self.fdi.delay

        # calculate total fa (for integrating fa[0])
        # self.fa_F = BaseSystem(np.zeros((1)))
        # self.dfa_F = BaseSystem(np.zeros((1)))

        nu = np.array([self.plant.m*self.plant.g, 0, 0, 0])[:, None]
        self.u = np.linalg.pinv(self.plant.mixer.B).dot(nu)

    def get_ref(self, t):
        # Set references
        # if t < 20:
        #     pos_des = np.vstack([-4, 3, 4])
        # else:
        #     pos_des = np.vstack([-6, 1, 7])
        pos_des = np.vstack([-4, 3, 4])
        vel_des = np.vstack([0, 0, 0])
        # pos_des = np.vstack([sin(t/2), cos(t/2), -t])
        # vel_des = np.vstack([cos(t/2)/2, -sin(t/2)/2, -1])
        quat_des = np.vstack([1, 0, 0, 0])
        omega_des = np.vstack([0, 0, 0])
        return np.vstack([pos_des, vel_des, quat_des, omega_des])

    def step(self):
        *_, done = self.update()
        return done

    def get_obs_ref(self, t):
        ref = self.get_ref(t)
        posd = ref[0:3]
        veld = np.zeros((3, 1))
        dveld = np.zeros((3, 1))
        ddveld = np.zeros((3, 1))
        psid = quat2angle(ref[6:10])[::-1][2]
        dpsid = 0
        psid2 = quat2angle(self.plant.quat.state)[::-1][2]
        dpsid2 = self.plant.omega.state[2][0]

        obs_ref = np.zeros((4, 4, 1))
        for i in range(3):
            obs_ref[i, :, :] = np.array([posd[i], veld[i],
                                         dveld[i], ddveld[i]])
        if t < self.detection_time[1]:
            obs_ref[3, 0:2, :] = np.array([psid, dpsid])[:, None]
        else:
            obs_ref[3, 0:2, :] = np.array([psid2, dpsid2])[:, None]
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

        # Observer
        obs_ctrl = np.zeros((4, 1))
        obs_ref = self.get_obs_ref(t)
        obs_ctrl[2] = self.lpeso_x.get_virtual(t, obs_ref[0, :, :])
        obs_ctrl[1] = self.lpeso_y.get_virtual(t, obs_ref[1, :, :])
        obs_ctrl[0] = self.lpeso_z.get_virtual(t, obs_ref[2, :, :])
        obs_ctrl[3] = self.hgeso_psi.get_virtual(obs_ref[3, 0:2, :])

        observation = np.zeros((4, 1))
        observation[0] = self.lpeso_x.get_obs()
        observation[1] = self.lpeso_y.get_obs()
        observation[2] = self.lpeso_z.get_obs()
        observation[3] = self.hgeso_psi.get_obs()

        fa = np.zeros((4, 1))
        fa[2] = self.lpeso_x.get_dist()
        fa[1] = self.lpeso_y.get_dist()
        fa[0] = self.lpeso_z.get_dist()
        fa[3] = self.hgeso_psi.get_dist()

        # Controller
        virtual_ctrl = self.controller.get_virtual(t, obs_ctrl)

        forces = self.controller.get_FM(virtual_ctrl)
        if t < self.detection_time[1]:
            Bp = self.plant.mixer.B
            fm = forces
            # rotors_cmd = np.linalg.pinv(self.plant.mixer.B).dot(forces)
        else:
            Bp = self.plant.mixer.B[0:3, :]
            fm = forces[0:3, :]
            # rotors_cmd = np.linalg.pinv(self.get_Bp(What)).dot(forces[0:3, :])
        W1, W2 = np.eye(6) - What, np.eye(6)
        W = W1 + W2
        H = Bp.dot(np.linalg.inv(W).dot(Bp.T))
        G = Bp.T.dot(np.linalg.inv(H))
        F_ = np.eye(6) - Bp.T.dot(np.linalg.inv(H).dot(Bp.dot(np.linalg.inv(W).dot(W2))))
        F = np.linalg.inv(W).dot(F_)
        rotors_cmd = F.dot(self.u) + G.dot(fm)

        # get true fault value
        rotors_cmd_noinfo = np.linalg.pinv(self.plant.mixer.B).dot(forces)
        tfa = - self.plant.mixer.B.dot((np.eye(6)-W).dot(rotors_cmd_noinfo))

        # actuator saturation
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        # Set actuator faults
        rotors = self.fault_manager.get_faulty_input(t, rotors)
        self.u = rotors

        self.plant.set_dot(t, rotors)
        self.controller.set_dot(virtual_ctrl)

        y = np.vstack([self.plant.pos.state, quat2angle(self.plant.quat.state)[::-1][2]])
        self.lpeso_x.set_dot(t, y[0], obs_ref[0, :, :])
        self.lpeso_y.set_dot(t, y[1], obs_ref[1, :, :])
        self.lpeso_z.set_dot(t, y[2], obs_ref[2, :, :])

        # self.fa_F.dot = self.dfa_F.state
        # self.dfa_F.dot = fa[0]
        # fa = np.vstack((self.fa_F.state, fa[1:4, :]))

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref,
                    obs_u=obs_ctrl, virtual_u=forces, obs=observation,
                    tfa=tfa, fa=fa
                    )


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
