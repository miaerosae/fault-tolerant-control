from fym.core import BaseEnv, BaseSystem

import numpy as np


def func_g(x, theta):
    # return np.sign(x) * abs(x)**theta
    delta = 1
    if abs(x) < delta:
        return x / delta**(1-theta)
    else:
        return np.sign(x) * abs(x)**theta


class PIDController(BaseEnv):
    def __init__(self, alp, eps, theta, init, kP, kD, kI, ctype):
        super().__init__()
        self.e = BaseSystem(np.vstack([init, 0, 0]))
        self.ei = BaseSystem(np.zeros((1,)))
        self.alp, self.eps = alp, eps
        self.theta = np.array([theta, 2*theta-1, 3*theta-2])
        self.kP, self.kD, self.kI = kP, kD, kI
        self.ctype = ctype
        self.e_record = 0

    def set_dot(self, t, y, ref, dref):
        alp, eps, theta = self.alp, self.eps, self.theta
        if self.ctype == "pos":
            real = y - ref
        elif self.ctype == "ang":
            real = y
        q = self.get_control(ref, dref)
        e, _ = self.observe_list()
        edot = np.zeros((3, 1))
        edot[0, :] = e[1] + (alp[0]/eps) * func_g(eps**2 * (real - e[0]), theta[0])
        edot[1, :] = e[2] + q + alp[1] * func_g(eps**2 * (real - e[0]), theta[1])
        edot[2, :] = alp[2] * eps * func_g(eps**2 * (real - e[0]), theta[2])
        self.e.dot = edot
        if self.ctype == "pos":
            self.ei.dot = e[0]
        elif self.ctype == "ang":
            self.ei.dot = e[0] - ref

    def get_control(self, ref, dref):
        if self.ctype == "pos":
            e = self.e.state[0]
            ed = self.e.state[1]
            self.e_record = e
        elif self.ctype == "ang":
            e = self.e.state[0] - ref
            ed = self.e.state[1] - dref
            self.e_record = e
        ei = self.ei.state
        return - (self.kP*e + self.kD*ed + self.kI*ei) - self.e.state[2]

    def get_obs(self):
        return self.e.state[0]

    def get_dist(self):
        return self.e.state[2]


if __name__ == "__main__":
    pass
