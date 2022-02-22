from fym.core import BaseEnv, BaseSystem

import numpy as np


class GESO_pos(BaseEnv):
    '''
    basic generalized extended state observer (for position tracking)

    *Set disturbance poly as d = p0 + p1t (n=2)

    Parameters:
        L: observer gain matrix (5 by 1)
    '''
    def __init__(self, L):
        super().__init__()
        self.xi = BaseSystem(np.zeros((4, 1)))
        self.y0 = BaseSystem(np.zeros((1, 1)))

        self.A = np.array([[0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        self.B = np.array([0, 0, 1, 0])[:, None]
        self.C = np.array([[1, 0, 0, 0]])
        self.L = L
        # self.Kx = Kx

    def deriv(self, xi, y0, y, v):
        L, A, B, C = self.L, self.A, self.B, self.C

        yhat = C.dot(self.xi.state)  # ydot
        xidot = A.dot(xi) + B.dot(v) + L*(y - yhat)
        return xidot

    def get_dist_obs(self):
        # c0 = self.C[0, 0:4]
        # A = self.A[0:4, 0:4]
        # bu = self.B[0:4, :]
        # bd = self.A[0:4, 4][:, None]
        # Kx = np.reshape(self.Kx, (1, 4))
        # K1 = np.linalg.inv(A+bu.dot(Kx))
        # Kd = 1 / (c0.dot(K1.dot(bu))) * (c0.dot(K1.dot(bd)))
        disturbance = self.xi.state[3]
        observation = self.y0.state
        return disturbance, observation

    def set_dot(self, t, y, v):
        '''
        y: desired observer's real state
        v: input made from controller
        '''
        states = self.observe_list()
        self.xi.dot = self.deriv(*states, y, v)
        self.y0.dot = self.C.dot(self.xi.state)


class GESO_psi(BaseEnv):
    '''
    L: (4 by 1)
    '''
    def __init__(self, L):
        super().__init__()
        self.xi = BaseSystem(np.zeros((2, 1)))
        self.y0 = BaseSystem(np.zeros((1, 1)))

        self.A = np.array([[0, 1],
                           [0, 0]])
        self.B = np.array([1, 0])[:, None]
        self.C = np.array([[1, 0]])
        # self.A = np.array([[0, 1, 0, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1],
        #                    [0, 0, 0, 0]])
        # self.B = np.array([0, 1, 0, 0])[:, None]
        # self.C = np.array([[1, 0, 0, 0]])
        self.L = L
        # self.Kx = Kx

    def deriv(self, xi, y0, y, v):
        L, A, B, C = self.L, self.A, self.B, self.C

        yhat = C.dot(self.xi.state)
        xidot = A.dot(xi) + B.dot(v) + L*(y - yhat)
        return xidot

    def get_dist_obs(self):
        # c0 = self.C[0, 0:2]
        # A = self.A[0:2, 0:2]
        # bu = self.B[0:2, :]
        # bd = self.A[0:2, 2][:, None]
        # Kx = np.reshape(self.Kx, (1, 2))
        # K1 = np.linalg.inv(A+bu.dot(Kx))
        # Kd = 1 / (c0.dot(K1.dot(bu))) * (c0.dot(K1.dot(bd)))
        disturbance = self.xi.state[1]
        observation = self.y0.state
        return disturbance, observation

    def set_dot(self, t, y, v):
        '''
        y: desired observer's real state
        v: input made from controller
        '''
        states = self.observe_list()
        dots = self.deriv(*states, y, v)
        self.xi.dot = dots
        self.y0.dot = self.C.dot(self.xi.state)


if __name__ == "__main__":
    system = GESO_psi(np.vstack([1, 1, 1, 1]))
    system.set_dot(t=0, y=np.zeros((1, 1)), v=np.zeros((1, 1)))
    print(repr(system))
