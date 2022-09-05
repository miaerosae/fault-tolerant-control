from fym.core import BaseEnv, BaseSystem

import numpy as np


def func_g(x, theta, delta):
    # return abs(x)**theta * np.sign(x)
    if abs(x) < delta:
        return x / delta**(1-theta)
    else:
        return np.sign(x) * abs(x)**theta


def sat(x, L):
    if x < - L:
        return - L
    elif x > L:
        return L
    else:
        return x


class outerLoop(BaseEnv):
    def __init__(self, l, alp, bet, R, K, rho, k, noise, init, theta):
        super().__init__()
        self.e = BaseSystem(np.zeros((3, 1)))
        self.eta = BaseSystem(np.zeros((2, 1)))

        self.alp, self.K, self.k = alp, K, k
        self.l, self.bet, self.R = l, bet, R
        self.rho_0, self.rho_inf = rho.ravel()
        self.theta = np.array([theta, 2*theta-1, 3*theta-2])
        self.noise = noise

    def get_delta(self, t):
        rho_0, rho_inf, k = self.rho_0, self.rho_inf, self.k
        rho = (rho_0-rho_inf) * np.exp(-k*t) + rho_inf
        return 2*rho

    def deriv(self, e, eta, y, ref, t):
        l, alp, bet, theta = self.l, self.alp, self.bet, self.theta
        R = self.R
        delta = self.get_delta(t)
        e_real = y - ref

        if self.noise is True:
            e_real = e_real + 0.001*np.random.randn(1)

        q = self.get_virtual(t)
        edot = np.zeros((3, 1))
        edot[0, :] = eta[0] + alp[0] * func_g(l*(e_real - e[0]), theta[0], delta)
        edot[1, :] = (eta[1] + q
                      + alp[1] * func_g(l*(sat(eta[0], R[0]) - e[1]), theta[1], delta))
        edot[2, :] = alp[2] * func_g(l*(sat(eta[1], R[1]) - e[2]), theta[2], delta)

        etadot = np.zeros((2, 1))
        etadot[0, :] = (sat(eta[1], R[1]) + q
                        + bet[0] * l*func_g(l*(e_real - e[0]), theta[0], delta))
        etadot[1, :] = bet[1] * l*func_g(l*(sat(eta[0], R[0]) - e[1]), theta[1], delta)
        # edot[0, :] = eta[0] + alp[0] * sat(l*(e_real - e[0]), l*delta)
        # edot[1, :] = (eta[1] + q
        #               + alp[1] * sat(l*(sat(eta[0], R[0]) - e[1]), l*delta))
        # edot[2, :] = alp[2] * sat(l*(sat(eta[1], R[1]) - e[2]), l*delta)

        # etadot = np.zeros((2, 1))
        # etadot[0, :] = (sat(eta[1], R[1]) + q
        #                 + bet[0] * sat(l**2*(e_real - e[0]), l**2*delta))
        # etadot[1, :] = bet[1] * sat(l**2*(sat(eta[0], R[0]) - e[1]), l**2*delta)

        return edot, etadot

    def get_virtual(self, t):
        rho_0, rho_inf, k, K = self.rho_0, self.rho_inf, self.k, self.K
        e = self.e.state
        rho = (rho_0-rho_inf) * np.exp(-k*t) + rho_inf
        drho = - k * (rho_0-rho_inf) * np.exp(-k*t)
        ddrho = k**2 * (rho_0-rho_inf) * np.exp(-k*t)

        z1 = e[0] / rho
        dz1 = e[1]/rho - e[0]*drho/rho**2
        alpha = - rho*K[0]*z1 + drho*z1
        z2 = e[1] - alpha
        dalpha = ddrho*z1 + drho*dz1 - drho*K[0]*z1 - rho*K[0]*dz1
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
    def __init__(self, l, alp, bet, dist_range, K, xi, rho, c, b, g, theta, noise):
        super().__init__()
        self.x = BaseSystem(np.zeros((3, 1)))
        self.eta = BaseSystem(np.zeros((2, 1)))
        self.lamb = BaseSystem(np.zeros((2, 1)))

        self.alp, self.K = alp, K
        self.l, self.bet, self.dist_range = l, bet, dist_range
        self.xi, self.rho = xi, rho
        self.c, self.b, self.g = c, b, g
        self.theta = np.array([theta, 2*theta-1, 3*theta-2])
        self.noise = noise

    def get_delta(self, t):
        return 2*self.rho[0]

    def get_R(self, t):
        return np.array([self.rho[1], self.dist_range])

    def deriv(self, x, eta, lamb, t, y, ref):
        l, alp, bet, theta = self.l, self.alp, self.bet, self.theta
        R = self.get_R(t)
        delta = self.get_delta(t)
        nu = self.get_virtual(t, ref)
        bound = self.b*self.xi
        nu_sat = np.clip(nu, bound[0], bound[1])

        if self.noise is True:
            y = y + np.deg2rad(0.001)*np.random.randn(1)

        xdot = np.zeros((3, 1))
        xdot[0, :] = eta[0] + alp[0] * func_g(l*(y - x[0]), theta[0], delta)
        xdot[1, :] = (eta[1] + nu_sat
                      + alp[1] * func_g(l*(sat(eta[0], R[0]) - x[1]), theta[1], delta))
        xdot[2, :] = alp[2] * func_g(l*(sat(eta[1], R[1]) - x[2]), theta[2], delta)

        etadot = np.zeros((2, 1))
        etadot[0, :] = (sat(eta[1], R[1]) + nu_sat
                        + bet[0] * l*func_g(l*(y - x[0]), theta[0], delta))
        etadot[1, :] = bet[1] * l*func_g(l*(sat(eta[0], R[0]) - x[1]), theta[1], delta)
        # xdot[0, :] = eta[0] + alp[0] * sat(l*(y - x[0]), l*delta)
        # xdot[1, :] = (eta[1] + nu_sat
        #               + alp[1] * sat(l*(sat(eta[0], R[0]) - x[1]), l*delta))
        # xdot[2, :] = alp[2] * sat(l*(sat(eta[1], R[1]) - x[2]), l*delta)

        # etadot = np.zeros((2, 1))
        # etadot[0, :] = (sat(eta[1], R[1]) + nu_sat
        #                 + bet[0] * sat(l**2*(y - x[0]), l**2*delta))
        # etadot[1, :] = bet[1] * sat(l**2*(sat(eta[0], R[0]) - x[1]), l**2*delta)

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

    def get_u(self, t, ref):
        nu = self.get_virtual(t, ref)
        bound = self.b*self.xi
        nu_sat = np.clip(nu, bound[0], bound[1])
        u = nu_sat / self.b
        return u

    def set_dot(self, t, y, ref):
        states = self.observe_list()
        dots = self.deriv(*states, t, y, ref)
        self.x.dot, self.eta.dot, self.lamb.dot = dots

    def get_obs(self):
        return self.x.state[0]

    def get_obsdot(self):
        return self.x.state[1]

    def get_dist(self):
        return self.x.state[2]


if __name__ == "__main__":
    pass
