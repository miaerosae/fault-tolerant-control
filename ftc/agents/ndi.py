from fym.core import BaseEnv, BaseSystem
import numpy as np
from fym.utils.rot import quat2angle, quat2dcm, angle2dcm
from numpy import cos, sin, tan


class NDIController(BaseEnv):
    def __init__(self, B, g, m, J, Jinv, Ko1, Ko2, Ki1, Ki2):
        super().__init__()
        self.B_r2f = B
        self.g, self.m = g, m
        self.J, self.Jinv = J, Jinv
        self.Ko1, self.Ko2 = Ko1, Ko2
        self.Ki1, self.Ki2 = Ki1, Ki2
        # Ko1 = 3 * np.diag((4, 1))
        # Ko2 = 3 * np.diag((3, 2))
        # Ki1 = 5 * np.diag((5, 10, 50, 10))
        # Ki2 = 1 * np.diag((5, 10, 50, 10))

    def get_control(self, t, pos, vel, ang, omega, posd, posd_dot):
        Ko1, Ko2 = self.Ko1, self.Ko2
        Ki1, Ki2 = self.Ki1, self.Ki2

        """ outer-loop control
        Objective: horizontal position (x, y) tracking control
        States:
            pos[0:2]: horizontal position
            posd[0:2]: desired horizontal position
        """
        xo, xod = pos[0:2], posd[0:2]
        xo_dot, xod_dot = vel[0:2], posd_dot[0:2]
        eo, eo_dot = xo - xod, xo_dot - xod_dot

        # outer-loop virtual control input
        nuo = (-Ko1 @ eo - Ko2 @ eo_dot) / self.g
        angd = np.vstack((nuo[1], -nuo[0], 0))
        # angd = np.deg2rad(np.vstack((0, 0, 0)))  # to regulate Euler angle

        """ inner-loop control
        Objective: vertical position (z) and angle (phi, theta, psi) tracking control
        States:
            pos[2]: vertical position
            posd[2]: desired vertical position
            ang: Euler angle
            angd: desired Euler angle
        """
        xi = np.vstack((pos[2], ang))
        xid = np.vstack((posd[2], angd))
        xi_dot = np.vstack((vel[2], omega))
        xid_dot = np.vstack((posd_dot[2], 0, 0, 0))
        ei = xi - xid
        ei_dot = xi_dot - xid_dot
        f = np.vstack(
            (
                self.g,
                -self.Jinv @ np.cross(omega, self.J @ omega, axis=0),
            )
        )
        g = np.zeros((4, 4))
        g[0, 0] = angle2dcm(ang[2][0], ang[1][0], ang[0][0]).T[2, 2] / self.m
        g[1:4, 1:4] = self.Jinv

        # inner-loop virtual control input
        nui = np.linalg.inv(g) @ (-f - Ki1 @ ei - Ki2 @ ei_dot)

        return angd, nuo, nui


if __name__ == "__main__":
    pass
