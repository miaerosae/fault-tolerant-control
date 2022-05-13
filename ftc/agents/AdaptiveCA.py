from fym.core import BaseEnv, BaseSystem

import numpy as np
from numpy import cos, sin
import scipy

'''
Based on backstepping method,
G. P. Falconi, F. Holzapfel, "Adaptive Fault Tolerant Control Allocation for
a Hexacopter System," American Control Conference, 2016
'''


def Proj_R(C, Y):
    proj_R = np.zeros((np.shape(Y)))
    for i in range(np.shape(Y)[0]):
        ci = C[i, :]
        yi = Y[i, :]
        proj_R[i, :] = Proj(ci, yi)
    return proj_R


def Proj(ci, yi, eps=1e-2, theta_max=1e5):
    f = ((1+eps) * np.linalg.norm(ci)**2 - theta_max**2) / (eps + theta_max**2)
    delf = (2*(1+eps) / (eps*theta_max**2)) * ci
    if f >= 0 and delf.T.dot(yi) > 0:
        out = yi - delf.dot(yi.dot(delf))/np.linalg.norm(delf)
    else:
        out = yi
    return out


def euler2rot(omega):
    phi, theta, psi = omega.ravel()
    c1 = sin(phi) * sin(theta)
    c2 = cos(phi) * sin(theta)
    R = np.array([[cos(theta)*cos(psi), c1*cos(psi)-cos(phi)*sin(psi), c2*cos(psi)+sin(phi)*sin(psi)],
                  [cos(theta)*sin(psi), c1*sin(psi)+cos(phi)*cos(psi), c2*sin(psi)-sin(phi)*cos(psi)],
                  [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)]])
    return R


class AdaptiveCA(BaseEnv):
    def __init__(self, B_A, m, J, gamma=1e-2):
        super().__init__()
        self.theta_hat = BaseSystem(np.zeros((6, 4)))
        self.B_A = B_A
        self.m = m
        self.J = J
        self.gamma = gamma

        self.zB = np.array([0, 0, 1])[:, None]
        self.Kx = m * np.eye(3)
        self.Kv = 1.82 * m * np.eye(3)
        self.Kp = np.hstack([self.Kx, self.Kv])
        self.Kt = 4 * np.eye(3)
        self.Ap = np.block([[np.zeros((3, 3)), np.eye(3)],
                            [-1/m*self.Kx, -1/m*self.Kv]])
        self.Bp = np.vstack([np.zeros((3, 3)), 1/m*np.eye(3)])
        self.P = scipy.linalg.solve_continuous_lyapunov(self.Ap, np.eye(6))
        self.P_bar = np.block([[self.P, np.zeros((6, 6))],
                               [np.zeros((6, 6)), 1/2*np.eye(6)]])

    def get_Tu_inv(self, T):
        Tu_inv = np.array([[0, 1/T, 0],
                           [-1/T, 0, 0],
                           [0, 0, -1]])
        return Tu_inv

    def get_ctrl(self, forces):
        ctrl = (np.linalg.pinv(self.B_A) + self.theta_hat.state).dot(forces)
        return ctrl

    def deriv(self, theta_hat, forces, des, state):
        B_A, gamma, zB = self.B_A, self.gamma, self.zB
        Ap, Bp, Kp, Kt = self.Ap, self.Bp, self.Kp, self.Kt
        P, P_bar = self.P, self.P_bar

        xd, vd, td, wd = des
        pos, vel, u1, omega = state
        e = np.vstack([xd-pos, vd-vel, u1-td, wd-omega])
        R = euler2rot(omega)

        theta_dot_ep = Bp
        theta_dot_u1 = Kp.dot(theta_dot_ep)
        theta_dot_et = theta_dot_u1
        theta_ddot_ep = Ap.dot(theta_dot_ep) + Bp.dot(theta_dot_et)
        theta_ddot_u1 = Kp.dot(theta_ddot_ep)
        Tu_inv = self.get_Tu_inv(forces[0][0])
        theta_dot_u2 = Tu_inv.dot(R.dot(
            2*Bp.T.dot(P.dot(theta_dot_ep)) + theta_ddot_u1
            + Kt.dot(theta_dot_et)))
        theta_dot_omegad = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]]).dot(theta_dot_u2)

        B_bar = np.block([[-theta_dot_ep.dot(zB), np.zeros((6, 3))],
                          [-theta_dot_u1.dot(zB), np.zeros((3, 3))],
                          [-theta_dot_omegad.dot(zB), np.zeros((3, 3))]])

        func = - (forces.dot(e.T.dot(P_bar.dot(B_bar.dot(B_A))))).T

        dot_theta_hat = gamma * Proj_R(theta_hat, func)

        return dot_theta_hat

    def set_dot(self, forces, des, state):
        theta_hat = self.theta_hat.state
        self.theta_hat.dot = self.deriv(theta_hat, forces,
                                        des, state)
