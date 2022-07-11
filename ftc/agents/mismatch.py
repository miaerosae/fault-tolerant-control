from fym.core import BaseEnv, BaseSystem

import numpy as np


def fal(tau, delta, theta):
    if abs(tau) > delta:
        return np.sign(tau) * (abs(tau)**theta)
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
    def __init__(self, n, b, r, K, M1, alp, delta, theta):
        super().__init__()
        self.e = BaseSystem(np.zeros((n+1, 1)))
        self.n, self.r, self.K = n, r, K
        self.M1, self.alp, self.delta = M1, alp, delta
        self.theta = np.zeros(n+1, )
        for i in range(n+1):
            self.theta[i] = np.divmod(theta * (i+1), 1)[1]

        self.b = b
        self.A = np.eye(n+1, n+1, 1)
        B = np.zeros((n+1, 1))
        B[n-1, :] = 1
        self.B = B

    def deriv(self, e, y, ref, u):
        n, r, K = self.n, self.r, self.K
        e_real = y - ref

        u = self.get_virtual()
        edot = self.A.dot(e) #+ self.B*(self.b*(u))
        for i in range(n+1):
            edot[i] = edot[i] + \
                K[i]/(r**(n-(i+1)))*fal(r**(n)*(e_real-e[0]), self.delta, self.theta[i])
        return edot

    def get_virtual(self):
        e = self.e.state
        return sat(self.M1, (self.alp.T.dot(e[0:self.n])-e[self.n])/self.b)

    def get_dist(self):
        return self.e.state[self.n][0]

    def get_obs(self):
        return self.e.state[0][0]

    def set_dot(self, t, y, ref, u):
        state = self.observe_list()
        self.e.dot = self.deriv(*state, y, ref, u)


if __name__ == "__main__":
    pass
