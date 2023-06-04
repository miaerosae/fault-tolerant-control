from fym.core import BaseEnv, BaseSystem

import numpy as np


def func_g(x, theta):
    # return np.sign(x) * abs(x)**theta
    delta = 1
    if abs(x) < delta:
        return x / delta**(1-theta)
    else:
        return np.sign(x) * abs(x)**theta


def q(x):
    if x > 0:
        return 1
    else:
        return 0


class outerLoop(BaseEnv):
    def __init__(self, alp, eps, theta, init):
        super().__init__()
        self.e = BaseSystem(np.vstack([init, 0, 0]))
        self.lamb = BaseSystem(np.zeros((2, 1)))
        self.integ_e = BaseSystem(np.zeros((1,)))

        self.alp, self.eps = alp, eps
        self.theta = np.array([theta, 2*theta-1, 3*theta-2])

    def deriv(self, e, lamb, integ_e, y, ref, t, q, *args):
        alp, eps, theta = self.alp, self.eps, self.theta
        e_real = y  # for state-subsystem estimation

        # q = self.get_virtual(t, ref, *args)
        # q_sat = np.clip(q, self.xi[0], self.xi[1])
        edot = np.zeros((3, 1))
        edot[0, :] = e[1] + (alp[0]/eps) * func_g(eps**2 * (e_real - e[0]), theta[0])
        edot[1, :] = e[2] + q + alp[1] * func_g(eps**2 * (e_real - e[0]), theta[1])
        edot[2, :] = alp[2] * eps * func_g(eps**2 * (e_real - e[0]), theta[2])
        lambdot = np.zeros((2, 1))
        # lambdot[0] = - self.c[0]*lamb[0] + lamb[1]
        # lambdot[1] = - self.c[1]*lamb[1] + (q_sat - q)
        integ_edot = y - ref
        return edot, lambdot, integ_edot

    def set_dot(self, t, y, ref, q, *args):
        states = self.observe_list()
        dots = self.deriv(*states, y, ref, t, q, *args)
        self.e.dot, self.lamb.dot, self.integ_e.dot = dots

    def get_obs(self):
        return self.e.state[0]

    def get_obsdot(self):
        return self.e.state[1]

    def get_dist(self):
        return self.e.state[2]


class innerLoop(BaseEnv):
    '''
    xi: lower and upper bound of u (moments for my case), [lower, upper]
    rho: bound of state x, dx
    virtual input nu = b*u
    '''
    def __init__(self, alp, eps, xi, c, b, theta):
        super().__init__()
        self.x = BaseSystem(shape=(3, 1))
        self.lamb = BaseSystem(np.zeros((2, 1)))
        self.integ_e = BaseSystem(np.zeros((1,)))

        self.alp, self.eps = alp, eps
        self.theta = np.array([theta, 2*theta-1, 3*theta-2])
        self.xi, self.c, self.b = xi, c, b

    def deriv(self, x, lamb, integ_e, t, y, ref, nu):
        alp, eps, theta = self.alp, self.eps, self.theta
        bound = self.b*self.xi
        nu_sat = np.clip(nu, bound[0], bound[1])

        xdot = np.zeros((3, 1))
        xdot[0, :] = x[1] + (alp[0]/eps) * func_g(eps**2 * (y - x[0]), theta[0])
        xdot[1, :] = x[2] + nu_sat + alp[1] * func_g(eps**2 * (y - x[0]), theta[1])
        xdot[2, :] = alp[2] * eps * func_g(eps**2 * (y - x[0]), theta[2])
        lambdot = np.zeros((2, 1))
        lambdot[0] = - self.c[0]*lamb[0] + lamb[1]
        lambdot[1] = - self.c[1]*lamb[1] + (nu_sat - nu)
        integ_edot = y - ref
        return xdot, lambdot, integ_edot

    def set_dot(self, t, y, ref, nu):
        states = self.observe_list()
        dots = self.deriv(*states, t, y, ref, nu)
        self.x.dot, self.lamb.dot, self.integ_e.dot = dots

    def get_obs(self):
        return self.x.state[0]

    def get_obsdot(self):
        return self.x.state[1]

    def get_dist(self):
        return self.x.state[2]


if __name__ == "__main__":
    pass
