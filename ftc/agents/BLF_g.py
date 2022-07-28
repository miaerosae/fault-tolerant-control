from fym.core import BaseEnv, BaseSystem

import numpy as np


def func_g(x, theta):
    return np.sign(x) * abs(x)**theta


class outerLoop(BaseEnv):
    def __init__(self, alp, eps, K, rho, k, init, theta):
        super().__init__()
        self.e = BaseSystem(np.vstack([init, 0, 0]))

        self.alp, self.eps, self.K, self.k = alp, eps, K, k
        self.rho_0, self.rho_inf = rho.ravel()
        self.theta = theta

    def deriv(self, e, y, ref, t):
        alp, eps, theta = self.alp, self.eps, self.theta
        e_real = y - ref

        q = self.get_virtual(t)
        edot = np.zeros((3, 1))
        edot[0, :] = e[1] + (alp[0]/eps) * func_g(eps**2 * (e_real - e[0]), theta[0])
        edot[1, :] = e[2] + q + alp[1] * func_g(eps**2 * (e_real - e[0]), theta[1])
        edot[2, :] = alp[2] * eps * func_g(eps**2 * (e_real - e[0]), theta[2])
        return edot

    def get_virtual(self, t):
        rho_0, rho_inf, k, K = self.rho_0, self.rho_inf, self.k, self.K
        e = self.e.state
        rho = (rho_0-rho_inf) * np.exp(-k*t) + rho_inf
        drho = - k * (rho_0-rho_inf) * np.exp(-k*t)
        ddrho = k**2 * (rho_0-rho_inf) * np.exp(-k*t)

        z1 = e[0] / rho
        dz1 = e[1]/rho - e[0]*drho/rho**2
        alpha = - (1-z1**2)*rho*K[0]*z1 + drho*z1
        z2 = e[1] - alpha
        dalpha = 2*rho*K[0]*dz1*z1**2 - (1-z1**2)*drho*K[0]*z1 \
            - (1-z1**2)*rho*K[0]*dz1 + ddrho*z1 + drho*dz1
        q = - e[2] + dalpha - K[1]*z2 - z1/(1-z1**2)/rho
        return q

    def set_dot(self, t, y, ref):
        states = self.observe_list()
        self.e.dot = self.deriv(*states, y, ref, t)

    def get_err(self):
        return self.e.state[0]

    def get_dist(self):
        return self.e.state[2]


class innerLoop(BaseEnv):
    '''
    xi: lower and upper bound of u (moments for my case), [lower, upper]
    rho: bound of state x, dx
    virtual input nu = f + b*u
    '''
    def __init__(self, alp, eps, K, xi, rho, c, b, g, theta):
        super().__init__()
        self.x = BaseSystem(np.zeros((3, 1)))
        self.lamb = BaseSystem(np.zeros((2, 1)))

        self.alp, self.eps, self.K = alp, eps, K
        self.xi, self.rho = xi, rho
        self.c, self.b, self.g = c, b, g
        self.theta = theta

    def deriv(self, x, lamb, t, y, ref, f):
        alp, eps, theta = self.alp, self.eps, self.theta
        nu = self.get_virtual(t, ref)
        bound = f + self.b*self.xi
        nu_sat = np.clip(nu, bound[0], bound[1])
        xdot = np.zeros((3, 1))
        xdot[0, :] = x[1] + (alp[0]/eps) * func_g(eps**2 * (y - x[0]), theta[0])
        xdot[1, :] = x[2] + nu_sat + alp[1] * func_g(eps**2 * (y - x[0]), theta[1])
        xdot[2, :] = alp[2] * eps * func_g(eps**2 * (y - x[0]), theta[2])
        lambdot = np.zeros((2, 1))
        lambdot[0] = - self.c[0]*lamb[0] + lamb[1]
        lambdot[1] = - self.c[1]*lamb[1] + (nu_sat - nu)
        return xdot, lambdot

    def get_virtual(self, t, ref):
        K, c, rho = self.K, self.c, self.rho
        x = self.x.state
        lamb = self.lamb.state
        lamb = np.vstack([0, 0])
        dref, ddref = 0, 0

        z1 = x[0] - ref - lamb[0]
        alpha = - K[0]*z1 - c[0]*lamb[0]
        z2 = x[1] - dref - alpha - lamb[1]
        dalpha = - K[0]*(x[1] - dref + c[0]*lamb[0] - lamb[1]) \
            - c[0]*(-c[0]*lamb[0] + lamb[1])
        nu = - c[1]*lamb[1] + dalpha + ddref - K[1]*z2 \
            - (rho[1]**2 - z2**2)/(rho[0]**2 - z1**2)*z1 - x[2]
        return nu

    def get_u(self, t, ref, f):
        nu = self.get_virtual(t, ref)
        bound = f + self.b*self.xi
        nu_sat = np.clip(nu, bound[0], bound[1])
        u = (nu_sat - f) / self.b
        return u

    def set_dot(self, t, y, ref, f):
        states = self.observe_list()
        self.x.dot, self.lamb.dot = self.deriv(*states, t, y, ref, f)

    def get_obs(self):
        return self.x.state[0]

    def get_obsdot(self):
        return self.x.state[1]

    def get_dist(self):
        return self.x.state[2]


if __name__ == "__main__":
    pass
