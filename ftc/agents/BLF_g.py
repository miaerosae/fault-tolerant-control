from fym.core import BaseEnv, BaseSystem

import numpy as np


def func_g(x, theta):
    # return np.sign(x) * abs(x)**theta
    delta = 1
    if abs(x) < delta:
        return x / delta**(1-theta)
    else:
        return np.sign(x) * abs(x)**theta


class outerLoop(BaseEnv):
    def __init__(self, alp, eps, K, rho, k, noise, init, theta, BLF=True):
        super().__init__()
        self.e = BaseSystem(np.vstack([init, 0, 0]))
        self.integ_e = BaseSystem(np.zeros((1,)))

        self.alp, self.eps, self.K, self.k = alp, eps, K, k
        self.rho_0, self.rho_inf = rho.ravel()
        # self.theta = np.ones((3,)) * theta
        self.theta = np.array([theta, 2*theta-1, 3*theta-2])
        self.noise = noise
        self.BLF = BLF

    def deriv(self, e, integ_e, y, ref, t, *args):
        alp, eps, theta = self.alp, self.eps, self.theta
        if self.BLF is True:
            e_real = y - ref  # for error-subsystem estimation
        else:
            e_real = y  # for state-subsystem estimation

        if self.noise is True:
            e_real = e_real + 0.001*np.random.randn(1)

        q = self.get_virtual(t, ref, *args)
        edot = np.zeros((3, 1))
        edot[0, :] = e[1] + (alp[0]/eps) * func_g(eps**2 * (e_real - e[0]), theta[0])
        edot[1, :] = e[2] + q + alp[1] * func_g(eps**2 * (e_real - e[0]), theta[1])
        edot[2, :] = alp[2] * eps * func_g(eps**2 * (e_real - e[0]), theta[2])
        integ_edot = y - ref
        return edot, integ_edot

    def get_virtual(self, t, ref, *args):
        e = self.e.state
        integ_e = self.integ_e.state
        rho_0, rho_inf, k, K = self.rho_0, self.rho_inf, self.k, self.K

        if self.BLF is True:
            rho = (rho_0-rho_inf) * np.exp(-k*t) + rho_inf
            drho = - k * (rho_0-rho_inf) * np.exp(-k*t)
            ddrho = k**2 * (rho_0-rho_inf) * np.exp(-k*t)

            z1 = e[0] / rho
            dz1 = e[1]/rho - e[0]*drho/rho**2
            alpha = - rho*K[0]*z1 + drho*z1 - K[2]*(1-z1**2)*rho**2*integ_e
            z2 = e[1] - alpha
            dalpha = ddrho*z1 + drho*dz1 - drho*K[0]*z1 - rho*K[0]*dz1 \
                - K[2]*(1-z1**2)*(rho**2*e[0]+2*rho*drho*integ_e) \
                + K[2]*2*z1*dz1*rho**2*integ_e
            q = - e[2] + dalpha - K[1]*z2 - z1/(1-z1**2)/rho
        else:
            dref, ddref = args
            alpha = K[0]*(ref-e[0]) + dref
            dalpha = K[0]*(dref-e[1]) + ddref
            q = -e[2] + dalpha + K[1]*(alpha-e[1])

        return q

    def set_dot(self, t, y, ref, *args):
        states = self.observe_list()
        self.e.dot, self.integ_e.dot = self.deriv(*states, y, ref, t, *args)

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
    def __init__(self, alp, eps, K, xi, rho, c, b, g, theta, noise, BLF=True):
        super().__init__()
        self.x = BaseSystem(np.zeros((3, 1)))
        self.lamb = BaseSystem(np.zeros((2, 1)))
        self.integ_e = BaseSystem(np.zeros((1,)))

        self.alp, self.eps, self.K = alp, eps, K
        self.xi, self.rho = xi, rho
        self.c, self.b, self.g = c, b, g
        self.theta = np.array([theta, 2*theta-1, 3*theta-2])
        self.noise = noise
        self.BLF = BLF

    def deriv(self, x, lamb, integ_e, t, y, ref):
        alp, eps, theta = self.alp, self.eps, self.theta
        nu = self.get_virtual(t, ref)
        bound = self.b*self.xi
        nu_sat = np.clip(nu, bound[0], bound[1])

        if self.noise is True:
            y = y + np.deg2rad(0.001)*np.random.randn(1)

        xdot = np.zeros((3, 1))
        xdot[0, :] = x[1] + (alp[0]/eps) * func_g(eps**2 * (y - x[0]), theta[0])
        xdot[1, :] = x[2] + nu_sat + alp[1] * func_g(eps**2 * (y - x[0]), theta[1])
        xdot[2, :] = alp[2] * eps * func_g(eps**2 * (y - x[0]), theta[2])
        lambdot = np.zeros((2, 1))
        lambdot[0] = - self.c[0]*lamb[0] + lamb[1]
        lambdot[1] = - self.c[1]*lamb[1] + (nu_sat - nu)
        integ_edot = y - ref
        return xdot, lambdot, integ_edot

    def get_virtual(self, t, ref):
        K, c, rho = self.K, self.c, self.rho
        x = self.x.state
        lamb = self.lamb.state
        integ_e = self.integ_e.state
        dref, ddref = 0, 0

        if self.BLF is True:
            z1 = x[0] - ref - lamb[0]
            dz1 = x[1] - dref + c[0]*lamb[0] - lamb[1]
            alpha = - K[0]*z1 - c[0]*lamb[0] - K[2]*integ_e*(rho[0]**2-z1**2)
            z2 = x[1] - dref - alpha - lamb[1]
            dalpha = - K[0]*(x[1] - dref + c[0]*lamb[0] - lamb[1]) \
                - c[0]*(-c[0]*lamb[0] + lamb[1]) - K[2]*(x[0]-ref)*(rho[0]**2-z1**2) \
                + K[2]*2*z1*dz1*integ_e
            nu = - c[1]*lamb[1] + dalpha + ddref - K[1]*z2 \
                - (rho[1]**2 - z2**2)/(rho[0]**2 - z1**2)*z1 - x[2]
        else:
            alpha = K[0]*(ref-x[0]) + dref
            dalpha = K[0]*(dref-x[1]) + ddref
            nu = -x[2] + dalpha + K[1]*(alpha-x[1])

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
        self.x.dot, self.lamb.dot, self.integ_e.dot = dots

    def get_obs(self):
        return self.x.state[0]

    def get_obsdot(self):
        return self.x.state[1]

    def get_dist(self):
        return self.x.state[2]


if __name__ == "__main__":
    pass
