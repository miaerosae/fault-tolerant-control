from fym.core import BaseEnv, BaseSystem

import numpy as np


def sat(x, L):
    if x < - L:
        return - L
    elif x > L:
        return L
    else:
        return x


class outerLoop(BaseEnv):
    def __init__(self, l, alp, bet, r, K, rho_0, rho_inf, k):
        super().__init__()
        self.e = BaseSystem(np.zeros((3, 1)))
        self.eta = BaseSystem(np.zeros((2, 1)))

        self.alp, self.K = alp, K
        self.l, self.bet, self.r = l, bet, r
        self.rho_0, self.rho_inf, self.k = rho_0, rho_inf, k

    def deriv(self, e, eta, y, ref, t):
        l, alp, bet, r = self.l, self.alp, self.bet, self.r
        e_real = y - ref

        q = self.get_virtual(t)
        edot = np.zeros((3, 1))
        edot[0, :] = eta[0] + l * alp[0] * (e_real - e[0])
        edot[1, :] = eta[1] + q + l * alp[1] * (sat(eta[0], r[0]) - e[1])
        edot[2, :] = l * alp[2] * (sat(eta[1], r[1]) - e[2])

        etadot = np.zeros((2, 1))
        etadot[0, :] = sat(eta[1], r[1]) + q + l**2 * bet[0] * (e_real - e[0])
        etadot[1, :] = l**2 * bet[1] * (sat(eta[0], r[0]) - e[1])

        return edot, etadot

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
        dalpha = 2*rho*K[0]*dz1*z1**2 - (1-z1**2)*drho*K[0]*(z1+dz1) \
            + ddrho*z1 + drho*dz1
        q = - e[2] + dalpha - K[1]*z2 - z1/(1-z1**2)/rho
        return q

    def set_dot(self, t, y, ref):
        states = self.observe_list()
        self.e.dot, self.eta.dot = self.deriv(*states, y, ref, t)

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
    def __init__(self, l, alp, bet, r, K, xi, rho, c, b, g):
        super().__init__()
        self.x = BaseSystem(np.zeros((3, 1)))
        self.eta = BaseSystem(np.zeros((2, 1)))
        self.lamb = BaseSystem(np.zeros((2, 1)))

        self.alp, self.K = alp, K
        self.l, self.bet, self.r = l, bet, r
        self.xi, self.rho = xi, rho
        self.c, self.b, self.g = c, b, g

    def deriv(self, x, eta, lamb, t, y, ref, f):
        l, alp, bet, r = self.l, self.alp, self.bet, self.r
        nu = self.get_virtual(t, ref)
        bound = f + self.b*self.xi
        nu_sat = np.clip(nu, bound[0], bound[1])

        xdot = np.zeros((3, 1))
        xdot[0, :] = eta[0] + l * alp[0] * (y - x[0])
        xdot[1, :] = eta[1] + nu_sat + l * alp[1] * (sat(eta[0], r[0]) - x[1])
        xdot[2, :] = l * alp[2] * (sat(eta[1], r[1]) - x[2])

        etadot = np.zeros((2, 1))
        etadot[0, :] = sat(eta[1], r[1]) + nu_sat + l**2 * bet[0] * (y - x[0])
        etadot[1, :] = l**2 * bet[1] * (sat(eta[0], r[0]) - x[1])

        lambdot = np.zeros((2, 1))
        lambdot[0] = - self.c[0]*lamb[0] + lamb[1]
        lambdot[1] = - self.c[1]*lamb[1] + (nu_sat - nu)
        return xdot, etadot, lambdot

    def get_virtual(self, t, ref):
        K, c, rho = self.K, self.c, self.rho
        x = self.x.state
        lamb = self.lamb.state
        # lamb = np.zeros((2, 1))
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
        dots = self.deriv(*states, t, y, ref, f)
        self.x.dot, self.eta.dot, self.lamb.dot = dots

    def get_obs(self):
        return self.x.state[0]

    def get_obsdot(self):
        return self.x.state[1]

    def get_dist(self):
        return self.x.state[2]


if __name__ == "__main__":
    pass
