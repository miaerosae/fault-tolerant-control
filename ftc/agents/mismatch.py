from fym.core import BaseEnv, BaseSystem

import numpy as np


def fal(tau, delta, theta):
    if abs(tau) > delta:
        return np.sign * (abs(tau)**theta)
    else:
        return tau / delta**(1-theta)


def sat(L, x):
    if x > L:
        return L
    elif x < -L:
        return -L
    else:
        return x


class mismatchESO(BaseEnv):
    def __init__(self, n, r, K, M1, alp1):
        self.e = BaseSystem(np.zeros((n+1, 1)))
        self.n, self.r, self.K = n, r, K
        self.M1, self.alp1 = M1, alp1

        self.A = np.eye(n+1, n+1, 1)

    def deriv(self, e, y, ref):
        n, r, K = self.n, self.r, self.K
        e_real = y - ref

        edot = self.A.dot(e)
        for i in range(n+1):
            edot[i] = (edot[i] + K[i]/(r**(n-(i+1)))*fal(r**(n)*(e_real-e)))
        return edot

    def get_virtual(self):
        e = self.e.state
        return sat(self.M1, self.alp1.T.dot(e[0:self.n-2])-e[self.n-1])

    def get_dist(self):
        return self.e.state[self.r-1][0]

    def set_dot(self, t, y, ref):
        state = self.observe_list()
        self.e.dot = self.deriv(*state, y, ref)


if __name__ == "__main__":
    pass
