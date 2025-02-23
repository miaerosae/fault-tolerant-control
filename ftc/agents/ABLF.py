from fym.core import BaseEnv, BaseSystem

import numpy as np


class outerLoop(BaseEnv):
    def __init__(self, alp, eps, K, rho, k, gamma, init, noise=False):
        super().__init__()
        self.e = BaseSystem(np.vstack([init, 0, 0]))
        self.theta = BaseSystem(np.zeros((1,)))

        self.alp, self.eps, self.K, self.k = alp, eps, K, k
        self.rho_0, self.rho_inf = rho.ravel()
        self.gamma = gamma
        self.noise = noise

    def deriv(self, e, theta, y, ref, t):
        alp, eps, gamma = self.alp, self.eps, self.gamma
        e_real = y - ref
        if self.noise is True:
            e_real = e_real + 0.001*np.random.randn(1)

        q, z2 = self.get_virtual(t)
        edot = np.zeros((3, 1))
        edot[0, :] = e[1] + (alp[0]/eps) * (e_real - e[0])
        edot[1, :] = e[2] + q + (alp[1]/eps**2) * (e_real - e[0])
        edot[2, :] = (alp[2]/eps**3) * (e_real - e[0])
        thetadot = gamma * z2
        return edot, thetadot

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
        return q, z2

    def set_dot(self, t, y, ref):
        states = self.observe_list()
        self.e.dot, self.theta.dot = self.deriv(*states, y, ref, t)

    def get_err(self):
        return self.e.state[0]

    def get_dist(self):
        return self.e.state[2]

    def get_theta(self):
        return self.theta.state


class innerLoop(BaseEnv):
    '''
    xi: lower and upper bound of u (moments for my case), [lower, upper]
    rho: bound of state x, dx
    virtual input nu = f + b*u
    '''
    def __init__(self, alp, eps, K, xi, rho, c, b, g, gamma, noise):
        super().__init__()
        self.x = BaseSystem(np.zeros((3, 1)))
        self.lamb = BaseSystem(np.zeros((2, 1)))
        self.theta = BaseSystem(np.zeros((1,)))

        self.alp, self.eps, self.K = alp, eps, K
        self.xi, self.rho = xi, rho
        self.c, self.b, self.g = c, b, g
        self.gamma = gamma
        self.noise = noise

    def deriv(self, x, lamb, theta, t, y, ref, f):
        alp, eps = self.alp, self.eps
        nu, z2 = self.get_virtual(t, ref)
        bound = f + self.b*self.xi
        nu_sat = np.clip(nu, bound[0], bound[1])
        if self.noise is True:
            y = y + np.deg2rad(0.001)*np.random.randn(1)
        xdot = np.zeros((3, 1))
        xdot[0, :] = x[1] + (alp[0]/eps) * (y - x[0])
        xdot[1, :] = x[2] + nu_sat + (alp[1]/eps**2) * (y - x[0])
        xdot[2, :] = (alp[2]/eps**3) * (y - x[0])
        lambdot = np.zeros((2, 1))
        lambdot[0] = - self.c[0]*lamb[0] + lamb[1]
        lambdot[1] = - self.c[1]*lamb[1] + (nu_sat - nu)
        thetadot = self.gamma * z2
        return xdot, lambdot, thetadot

    def get_virtual(self, t, ref):
        K, c, rho = self.K, self.c, self.rho
        x = self.x.state
        lamb = self.lamb.state
        # lamb = np.vstack([0, 0])
        dref, ddref = 0, 0

        z1 = x[0] - ref - lamb[0]
        alpha = - K[0]*z1 - c[0]*lamb[0]
        z2 = x[1] - dref - alpha - lamb[1]
        dalpha = - K[0]*(x[1] - dref + c[0]*lamb[0] - lamb[1]) \
            - c[0]*(-c[0]*lamb[0] + lamb[1])
        nu = - c[1]*lamb[1] + dalpha + ddref - K[1]*z2 \
            - (rho[1]**2 - z2**2)/(rho[0]**2 - z1**2)*z1 - x[2]
        return nu, z2

    def get_u(self, t, ref, f):
        nu, _ = self.get_virtual(t, ref)
        bound = f + self.b*self.xi
        nu_sat = np.clip(nu, bound[0], bound[1])
        u = (nu_sat - f) / self.b
        return u

    def set_dot(self, t, y, ref, f):
        states = self.observe_list()
        dots = self.deriv(*states, t, y, ref, f)
        self.x.dot, self.lamb.dot, self.theta.dot = dots

    def get_obs(self):
        return self.x.state[0]

    def get_obsdot(self):
        return self.x.state[1]

    def get_dist(self):
        return self.x.state[2]

    def get_theta(self):
        return self.theta.state


if __name__ == "__main__":
    pass
