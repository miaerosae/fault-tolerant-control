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
        self.xi = BaseSystem(np.zeros((5, 1)))
        self.y0 = BaseSystem(np.zeros((1, 1)))
        self.y1 = BaseSystem(np.zeros((1, 1)))
        self.y2 = BaseSystem(np.zeros((1, 1)))

        self.A = np.array([[0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0]])
        self.B = np.array([1, 0, 0, 0, 0])[:, None]
        self.C = np.array([1, 0, 0, 0, 0])
        # self.A = np.array([[0, 1, 0, 0, 0, 0],
        #                    [0, 0, 1, 0, 0, 0],
        #                    [0, 0, 0, 1, 0, 0],
        #                    [0, 0, 0, 0, 1, 0],
        #                    [0, 0, 0, 0, 0, 1],
        #                    [0, 0, 0, 0, 0, 0]])
        # self.B = np.array([0, 0, 0, 1, 0, 0])[:, None]
        # self.C = np.array([[1, 0, 0, 0, 0, 0]])
        self.L = L

    def deriv(self, xi, y0, y1, y2, y, v):
        L, A, B, C = self.L, self.A, self.B, self.C

        yhat = C.dot(self.xi.state)  # y 3dot
        xidot = A.dot(xi) + B.dot(v) + L*(y - yhat)
        y2dot = self.xi.state[0]
        y1dot = y2
        y0dot = y1
        return xidot, y0dot, y1dot, y2dot

    def get_dist_obs(self):
        disturbance = self.xi.state[1]
        observation = self.y0.state
        return disturbance, observation

    def set_dot(self, t, y, v):
        '''
        y: desired observer's real state
        v: input made from controller
        '''
        states = self.observe_list()
        self.xi.dot, self.y0.dot, self.y1.dot, self.y2.dot = self.deriv(*states, y, v)


class GESO_psi(BaseEnv):
    '''
    L: (4 by 1)
    '''
    def __init__(self, L):
        super().__init__()
        self.xi = BaseSystem(np.zeros((3, 1)))
        self.psi = BaseSystem(np.zeros((1, 1)))

        self.A = np.array([[0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]])
        self.B = np.array([1, 0, 0])[:, None]
        self.C = self.B.T
        # self.A = np.array([[0, 1, 0, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1],
        #                    [0, 0, 0, 0]])
        # self.B = np.array([0, 1, 0, 0])[:, None]
        # self.C = np.array([[1, 0, 0, 0]])
        self.L = L

    def deriv(self, xi, psi, y, v):
        L, A, B, C = self.L, self.A, self.B, self.C

        yhat = C.dot(self.xi.state)
        xidot = A.dot(xi) + B.dot(v) + L*(y - yhat)
        psidot = self.xi.state[0]
        return xidot, psidot

    def get_dist_obs(self):
        disturbance = self.xi.state[1]
        observation = self.psi.state
        return disturbance, observation

    def set_dot(self, t, y, v):
        '''
        y: desired observer's real state
        v: input made from controller
        '''
        states = self.observe_list()
        dots = self.deriv(*states, y, v)
        self.xi.dot, self.psi.dot = dots


if __name__ == "__main__":
    system = GESO_psi(np.vstack([1, 1, 1, 1]))
    system.set_dot(t=0, y=np.zeros((1, 1)), v=np.zeros((1, 1)))
    print(repr(system))
