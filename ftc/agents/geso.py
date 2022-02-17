from fym.core import BaseEnv, BaseSystem

import numpy as np


class GESO_pos(BaseEnv):
    '''
    basic generalized extended state observer (for position tracking)

    *Set disturbance poly as d = p0 + p1t (n=2)

    Parameters:
        L: observer gain matrix (6 by 1)
    '''
    def __init__(self, L):
        super().__init__()
        self.xi = BaseSystem(np.zeros((6, 1)))

        self.A = np.array([[0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0]])
        self.B = np.array([0, 0, 0, 1, 0, 0])[:, None]
        self.C = np.array([[1, 0, 0, 0, 0, 0]])
        self.L = L

    def deriv(self, xi, y, v):
        L, A, B, C = self.L, self.A, self.B, self.C

        y1 = C.dot(self.xi.state)
        xidot = A.dot(xi) + B.dot(v) + L.dot(y - y1)
        return xidot

    def get_dist_obs(self, y):
        disturbance = self.xi.state[4]
        observation = self.C.dot(self.xi.state)
        return disturbance, observation

    def set_dot(self, t, y, v):
        '''
        y: desired observer's real state
        v: input made from controller
        '''
        states = self.observe_list()
        self.xi.dot = self.deriv(*states, y, v)


class GESO_psi(BaseEnv):
    '''
    L: (4 by 1)
    '''
    def __init__(self, L):
        super().__init__()
        self.xi = BaseSystem(np.zeros((4, 1)))

        self.A = np.array([[0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0]])
        self.B = np.array([0, 1, 0, 0])[:, None]
        self.C = np.array([[1, 0, 0, 0]])
        self.L = L

    def deriv(self, xi, y, v):
        L, A, B, C = self.L, self.A, self.B, self.C

        y1 = C.dot(self.xi.state)
        xidot = A.dot(xi) + B.dot(v) + L.dot(y - y1)
        return xidot

    def get_dist_obs(self, y):
        disturbance = self.xi.state[2]
        observation = self.C.dot(self.xi.state)
        return disturbance, observation

    def set_dot(self, t, y, v):
        '''
        y: desired observer's real state
        v: input made from controller
        '''
        states = self.observe_list()
        self.xi.dot = self.deriv(*states, y, v)


if __name__ == "__main__":
    system = GESO_psi(np.vstack([1, 1, 1, 1]))
    system.set_dot(t=0, y=np.zeros((1, 1)), v=np.zeros((1, 1)))
    print(repr(system))
