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
    def __init__(self, K, rho, k, init, BLF):
        super().__init__()
        self.integ_e = BaseSystem(np.zeros((1,)))

        self.K, self.k = K, k
        self.rho_0, self.rho_inf = rho.ravel()
        self.BLF = BLF

    def get_virtual(self, t, e, *args):
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
            q = dalpha - K[1]*z2 - z1/(1-z1**2)/rho
        else:
            x2, dref, ddref = args
            q = ddref - K[0]*(x2-dref) - K[1]*(x2-dref+K[0]*e[0]) - e[0]

        return q

    def set_dot(self, e):
        self.integ_e.dot = e


class innerLoop(BaseEnv):
    '''
    xi: lower and upper bound of u (moments for my case), [lower, upper]
    rho: bound of state x, dx
    virtual input nu = b*u
    '''
    def __init__(self, K, rho, b, BLF):
        super().__init__()
        self.integ_e = BaseSystem(np.zeros((1,)))

        self.K = K
        self.rho = rho
        self.b = b
        self.BLF = BLF

    def get_virtual(self, t, x, ref):
        K, rho = self.K, self.rho
        integ_e = self.integ_e.state
        dref, ddref = 0, 0

        if self.BLF is True:
            rho1a = ref + rho[0]
            rho1b = rho[0] - ref
            z1 = x[0] - ref
            dz1 = x[1] - dref
            xi_1a = z1 / rho1a
            dxi_1a = (dz1*rho1a) / (rho1a**2)
            xi_1b = z1 / rho1b
            dxi_1b = (dz1*rho1b) / (rho1b**2)
            xi1 = q(z1)*xi_1b + (1-q(z1))*xi_1a
            dxi1 = q(z1)*dxi_1b + (1-q(z1))*dxi_1a
            bar_k1 = 0
            alpha = - (K[0] + bar_k1)*z1 - K[2]*integ_e*(1-xi1**2)
            dbar_k1 = 0
            dalpha = (- dbar_k1*z1 - (K[0] + bar_k1)*dz1
                      - K[2]*(x[0]-ref)*(1-xi1**2)
                      - K[2]*integ_e*(-2*xi1*dxi1))
            rho2a = alpha + rho[1]
            rho2b = rho[1] - alpha
            drho2a = dalpha
            drho2b = - dalpha
            z2 = x[1] - alpha
            mu1 = q(z1)/(rho1b**2-z1**2) + (1-q(z1))/(rho1a**2-z1**2)
            bar_k2 = ((drho2a/rho2a)**2 + (drho2b/rho2b)**2 + 0.1) ** (1/2)
            nu = - (K[1] + bar_k2)*z2 + dalpha - mu1*z1
        else:
            x2 = x[1]
            z1 = x[0] - ref
            nu = ddref - K[0]*(x2-dref) - K[1]*(x2-dref+K[0]*z1) - z1

        return nu

    def get_u(self, t, x, ref, f):
        nu = self.get_virtual(t, x, ref)
        u = (nu-f) / self.b
        return u

    def set_dot(self, t, y, ref):
        self.integ_e.dot = y - ref


if __name__ == "__main__":
    pass
