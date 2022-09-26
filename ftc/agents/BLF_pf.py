from fym.core import BaseEnv, BaseSystem

import numpy as np


def sat(x, L):
    if x < - L:
        return - L
    elif x > L:
        return L
    else:
        return x


def q(x):
    if x > 0:
        return 1
    else:
        return 0


class outerLoop(BaseEnv):
    def __init__(self, l, alp, bet, R, K, rho, k, noise, init):
        super().__init__()
        self.e = BaseSystem(np.zeros((3, 1)))
        self.eta = BaseSystem(np.zeros((2, 1)))
        self.integ_e = BaseSystem(np.zeros((1,)))

        self.alp, self.K, self.k = alp, K, k
        self.l, self.bet, self.R = l, bet, R
        self.rho_0, self.rho_inf = rho.ravel()
        self.noise = noise

    def deriv(self, e, eta, integ_e, y, ref, t):
        l, alp, bet = self.l, self.alp, self.bet
        R = self.R
        e_real = y - ref

        if self.noise is True:
            e_real = e_real + 0.001*np.random.randn(1)

        q = self.get_virtual(t)
        edot = np.zeros((3, 1))
        edot[0, :] = eta[0] + l * alp[0] * (e_real - e[0])
        edot[1, :] = eta[1] + q + l * alp[1] * (sat(eta[0], R[0]) - e[1])
        edot[2, :] = l * alp[2] * (sat(eta[1], R[1]) - e[2])

        etadot = np.zeros((2, 1))
        etadot[0, :] = sat(eta[1], R[1]) + q + l**2 * bet[0] * (e_real - e[0])
        etadot[1, :] = l**2 * bet[1] * (sat(eta[0], R[0]) - e[1])

        integ_edot = y - ref
        return edot, etadot, integ_edot

    def get_virtual(self, t):
        rho_0, rho_inf, k, K = self.rho_0, self.rho_inf, self.k, self.K
        e = self.e.state
        integ_e = self.integ_e.state
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
        return q

    def set_dot(self, t, y, ref):
        states = self.observe_list()
        dots = self.deriv(*states, y, ref, t)
        self.e.dot, self.eta.dot, self.integ_e.dot = dots

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
    def __init__(self, l, alp, bet, dist_range, K, xi, rho, c, b, g, noise):
        super().__init__()
        self.x = BaseSystem(np.zeros((3, 1)))
        self.eta = BaseSystem(np.zeros((2, 1)))
        self.lamb = BaseSystem(np.zeros((2, 1)))
        self.integ_e = BaseSystem(np.zeros((1,)))

        self.alp, self.K = alp, K
        self.l, self.bet, self.dist_range = l, bet, dist_range
        self.xi, self.rho = xi, rho
        self.c, self.b, self.g = c, b, g
        self.noise = noise

    def get_R(self, t):
        return np.array([self.rho[1], self.dist_range])

    def deriv(self, x, eta, lamb, integ_e, t, y, ref):
        l, alp, bet = self.l, self.alp, self.bet
        R = self.get_R(t)
        nu = self.get_virtual(t, ref)
        bound = self.b*self.xi
        nu_sat = np.clip(nu, bound[0], bound[1])

        if self.noise is True:
            y = y + np.deg2rad(0.001)*np.random.randn(1)

        xdot = np.zeros((3, 1))
        xdot[0, :] = eta[0] + l * alp[0] * (y - x[0])
        xdot[1, :] = eta[1] + nu_sat + l * alp[1] * (sat(eta[0], R[0]) - x[1])
        xdot[2, :] = l * alp[2] * (sat(eta[1], R[1]) - x[2])

        etadot = np.zeros((2, 1))
        etadot[0, :] = sat(eta[1], R[1]) + nu_sat + l**2 * bet[0] * (y - x[0])
        etadot[1, :] = l**2 * bet[1] * (sat(eta[0], R[0]) - x[1])

        lambdot = np.zeros((2, 1))
        lambdot[0] = - self.c[0]*lamb[0] + lamb[1]
        lambdot[1] = - self.c[1]*lamb[1] + (nu_sat - nu)
        integ_edot = y - ref
        return xdot, etadot, lambdot, integ_edot

    def get_virtual(self, t, ref):
        K, c, rho = self.K, self.c, self.rho
        x = self.x.state
        lamb = self.lamb.state
        dlamb = self.lamb.dot
        integ_e = self.integ_e.state
        dref = 0

        rho1a = ref + lamb[0] + rho[0]
        rho1b = rho[0] - ref - lamb[0]
        drho1a = dlamb[0]
        drho1b = - dlamb[0]
        ddrho1a = - c[0]*dlamb[0] + dlamb[1]
        ddrho1b = c[0]*dlamb[0] - dlamb[1]

        z1 = x[0] - ref - lamb[0]
        dz1 = x[1] - dref - dlamb[0]

        xi_1a = z1 / rho1a
        dxi_1a = (dz1*rho1a-z1*drho1a) / (rho1a**2)
        xi_1b = z1 / rho1b
        dxi_1b = (dz1*rho1b-z1*drho1b) / (rho1b**2)
        xi1 = q(z1)*xi_1b + (1-q(z1))*xi_1a
        dxi1 = q(z1)*dxi_1b + (1-q(z1))*dxi_1a

        bar_k1 = ((drho1a/rho1a)**2 + (drho1b/rho1b)**2 + 0.1) ** (1/2)
        alpha = - (K[0] + bar_k1)*z1 - c[0]*lamb[0] - K[2]*integ_e*(1-xi1**2)

        dbar_k1 = 1 / 2 / bar_k1 * (
            2*drho1a*(ddrho1a*rho1a-drho1a**2)/(rho1a**3)
            + 2*drho1b*(ddrho1b*rho1b-drho1b**2)/(rho1b**3)
        )
        dalpha = (- dbar_k1*z1 - bar_k1*dz1 - c[0]*dlamb[0]
                  - K[2]*(x[0]-ref)*(1-xi1**2)
                  - K[2]*integ_e*(-2*xi1*dxi1))

        rho2a = alpha + lamb[1] + rho[1]
        rho2b = rho[1] - alpha - lamb[1]
        drho2a = dalpha + dlamb[1]
        drho2b = - dalpha - dlamb[1]

        z2 = x[1] - alpha - lamb[1]

        mu1 = q(z1)/(rho1b**2-z1**2) + (1-q(z1))/(rho1a**2-z1**2)
        bar_k2 = ((drho2a/rho2a)**2 + (drho2b/rho2b)**2 + 0.1) ** (1/2)
        nu = - (K[1] + bar_k2)*z2 + dalpha - x[2] - c[1]*lamb[1] - mu1*z1

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
        self.x.dot, self.eta.dot, self.lamb.dot, self.integ_e.dot = dots

    def get_obs(self):
        return self.x.state[0]

    def get_obsdot(self):
        return self.x.state[1]

    def get_dist(self):
        return self.x.state[2]


if __name__ == "__main__":
    pass
