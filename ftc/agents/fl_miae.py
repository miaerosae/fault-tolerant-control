import numpy as np
from numpy import sin, cos

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2angle


class FLController(BaseEnv):
    def __init__(self, m, g, J):
        super().__init__()
        self.m, self.g, self.J = m, g, J
        self.Jinv = np.linalg.inv(J)

        self.angle = BaseSystem(np.zeros((3, 1)))
        self.dangle = BaseSystem(np.zeros((3, 1)))
        self.u1 = BaseSystem(np.array((m*g)))
        self.du1 = BaseSystem(np.zeros((1)))

        self.trim_forces = np.vstack([self.m * self.g, 0, 0, 0])

        self.F1 = - np.array([[1, 1, 1]])*23.95
        self.F2 = - np.array([[1, 1, 1]])*48.11
        self.F3 = - np.array([[1, 1, 1]])*32.95
        self.F4 = - np.array([[1, 1, 1]])*9.79
        self.F_psi = - np.array([[4, 5, 0, 0]])
        F = np.hstack([self.F1.T, self.F2.T, self.F3.T, self.F4.T])
        self.F = np.vstack([F, self.F_psi])

    def get_alpbet(self, plant, reF):
        m, J = self.m, self.J
        quat = plant.quat.state
        # current state
        phi, theta, psi = quat2angle(quat)[::-1]
        dphi, dtheta, dpsi = self.dangle.state.ravel()
        ddphi = (J[1, 1]-J[2, 2])/J[0, 0]*dtheta*dpsi
        ddtheta = (J[2, 2]-J[0, 0])/J[1, 1]*dphi*dpsi
        ddpsi = (J[0, 0]-J[1, 1])/J[2, 2]*dphi*dtheta
        u1 = self.u1.state[0]
        du1 = self.du1.state[0]

        # bet(x)
        bet = np.zeros((4, 4))
        bet[0, 0] = - (sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta))/m
        bet[0, 1] = (-cos(phi)*sin(psi) + cos(psi)*sin(phi)*sin(theta))*u1/J[0, 0]/m
        bet[0, 2] = -cos(phi)*cos(psi)*cos(theta)*u1/J[1, 1]/m
        bet[0, 3] = (-cos(psi)*sin(phi) + cos(phi)*sin(psi)*sin(theta))*u1/J[2, 2]/m
        bet[1, 0] = (cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta))/m
        bet[1, 1] = (cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta))*u1/J[0, 0]/m
        bet[1, 2] = - cos(phi)*cos(theta)*sin(psi)*u1/J[1, 1]/m
        bet[1, 3] = - (sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta))*u1/J[2, 2]/m
        bet[2, 0] = - cos(phi)*cos(theta)/m
        bet[2, 1] = cos(theta)*sin(phi)*u1/J[0, 0]/m
        bet[2, 2] = cos(phi)*sin(theta)*u1/J[1, 1]/m
        bet[3, 3] = 1/J[2, 2]
        # print(repr(np.linalg.matrix_rank(bet)))

        # alp(x)
        alp = np.zeros((4, 1))
        alp[0, 0] = (u1*(sin(phi)*sin(psi)*dphi**2 + sin(phi)*sin(psi)*dpsi**2
                         - cos(phi)*sin(psi)*ddphi - cos(psi)*sin(phi)*ddpsi
                         + cos(phi)*cos(psi)*sin(theta)*dphi**2
                         + cos(phi)*cos(psi)*sin(theta)*dpsi**2
                         - 2*cos(phi)*cos(psi)*dphi*dpsi
                         + cos(phi)*cos(psi)*sin(theta)*dtheta**2
                         - cos(phi)*cos(psi)*cos(theta)*ddtheta
                         + cos(psi)*sin(phi)*sin(theta)*ddphi
                         + cos(phi)*sin(psi)*sin(theta)*ddpsi
                         + 2*cos(psi)*cos(theta)*sin(phi)*dphi*dtheta
                         + 2*cos(phi)*cos(theta)*sin(psi)*dpsi*dtheta
                         - 2*sin(phi)*sin(psi)*sin(theta)*dphi*dpsi))/m \
            - (2*du1*(cos(phi)*sin(psi)*dphi + cos(psi)*sin(phi)*dpsi
                      - cos(psi)*sin(phi)*sin(theta)*dphi
                      - cos(phi)*sin(psi)*sin(theta)*dpsi
                      + cos(phi)*cos(psi)*cos(theta)*dtheta))/m
        alp[1, 0] = -(2*du1*(sin(phi)*sin(psi)*dpsi - cos(phi)*cos(psi)*dphi
                             + cos(phi)*cos(psi)*sin(theta)*dpsi
                             + cos(phi)*cos(theta)*sin(psi)*dtheta
                             - sin(phi)*sin(psi)*sin(theta)*dphi))/m \
            - (u1*(cos(psi)*sin(phi)*dphi**2 + cos(psi)*sin(phi)*dpsi**2
                   - cos(phi)*cos(psi)*ddphi + sin(phi)*sin(psi)*ddpsi
                   - cos(phi)*sin(psi)*sin(theta)*dphi**2
                   - cos(phi)*sin(psi)*sin(theta)*dpsi**2
                   + 2*cos(phi)*sin(psi)*dphi*dpsi
                   - cos(phi)*sin(psi)*sin(theta)*dtheta**2
                   + cos(phi)*cos(psi)*sin(theta)*ddpsi
                   + cos(phi)*cos(theta)*sin(psi)*ddtheta
                   - sin(phi)*sin(psi)*sin(theta)*ddphi
                   + 2*cos(phi)*cos(psi)*cos(theta)*dpsi*dtheta
                   - 2*cos(psi)*sin(phi)*sin(theta)*dphi*dpsi
                   - 2*cos(theta)*sin(phi)*sin(psi)*dphi*dtheta))/m
        alp[2, 0] = (2*cos(theta)*sin(phi)*dphi*du1)/m \
            + (2*cos(phi)*sin(theta)*dtheta*du1)/m \
            + (cos(phi)*cos(theta)*u1*dphi**2)/m \
            + (cos(phi)*cos(theta)*u1*dtheta**2)/m \
            + (cos(theta)*sin(phi)*u1*ddphi)/m \
            + (cos(phi)*sin(theta)*u1*ddtheta)/m \
            - (2*sin(phi)*sin(theta)*u1*dphi*dtheta)/m
        alp[3, 0] = ddpsi

        return alp, bet

    def get_virtual(self, t, plant, ref,
                    disturbance=np.zeros((4, 1)), obs_u=np.zeros((4, 1))):
        m, g = self.m, self.g

        # desired value
        posd = ref[0:3]
        veld = np.zeros((3, 1))
        dveld = np.zeros((3, 1))
        ddveld = np.zeros((3, 1))
        psid = quat2angle(ref[6:10])[::-1][2]
        dpsid = 0

        # current state
        pos = plant.pos.state
        vel = plant.vel.state
        quat = plant.quat.state
        phi, theta, psi = quat2angle(quat)[::-1]
        dphi, dtheta, dpsi = self.dangle.state.ravel()
        u1 = self.u1.state[0]
        du1 = self.du1.state[0]
        dvel = np.array([
            -(u1*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)))/m,
            (u1*(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta)))/m,
            g - (cos(phi)*cos(theta)*u1)/m
        ])[:, None]
        ddvel = np.array([
            - (u1*(cos(phi)*sin(psi)*dphi + cos(psi)*sin(phi)*dpsi
                   - cos(psi)*sin(phi)*sin(theta)*dphi
                   - cos(phi)*sin(psi)*sin(theta)*dpsi
                   + cos(phi)*cos(psi)*cos(theta)*dtheta))/m
            - ((sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta))*du1)/m,
            ((cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta))*du1)/m
            - (u1*(sin(phi)*sin(psi)*dpsi - cos(phi)*cos(psi)*dphi
                   + cos(phi)*cos(psi)*sin(theta)*dpsi
                   + cos(phi)*cos(theta)*sin(psi)*dtheta
                   - sin(phi)*sin(psi)*sin(theta)*dphi))/m,
            (cos(theta)*sin(phi)*u1*dphi)/m
            - (cos(phi)*cos(theta)*du1)/m + (cos(phi)*sin(theta)*u1*dtheta)/m
        ])[:, None]

        alp, bet = self.get_alpbet(plant, ref)

        # define new control input vector v
        k1 = np.diag(self.F1[0, :])
        k2 = np.diag(self.F2[0, :])
        k3 = np.diag(self.F3[0, :])
        k4 = np.diag(self.F4[0, :])
        v = np.zeros((4, 1))
        v[0:3, :] = (k1.dot(pos-posd)
                     + k2.dot(vel-veld)
                     + k3.dot(dvel-dveld)
                     + k4.dot(ddvel-ddveld))
        v[3, :] = self.F_psi[:, 0:2].dot(np.vstack([psi-psid, dpsi-dpsid]))

        # fm = np.linalg.inv(bet).dot(-alp + v - disturbance) + obs_u
        fm = obs_u
        d2u1, u2, u3, u4 = fm.ravel()

        return d2u1, np.array([u2, u3, u4])[:, None]

    def get_FM(self, ctrl):
        return np.vstack((self.u1.state, ctrl[1]))

    def set_dot(self, ctrl):
        J = self.J
        dangle = self.dangle.state
        self.angle.dot = self.dangle.state
        self.dangle.dot = self.Jinv.dot(ctrl[1]) \
            + np.vstack([
                (J[1, 1]-J[2, 2])/J[0, 0]*dangle[1]*dangle[2],
                (J[2, 2]-J[0, 0])/J[1, 1]*dangle[0]*dangle[2],
                (J[0, 0]-J[1, 1])/J[2, 2]*dangle[0]*dangle[1]
            ])

        self.u1.dot = self.du1.state
        self.du1.dot = ctrl[0]


if __name__ == "__main__":
    pass
