from fym.core import BaseEnv, BaseSystem

import numpy as np


def sat(L, x):
    if x > L:
        return L
    elif x < -L:
        return -L
    else:
        return x


def fun_psi(Sxi, Bsig, b0, F):
    return 1 / b0 * (- Bsig + F.dot(Sxi))


class lowPowerESO(BaseEnv):
    '''
    References
    Wu, Yuanqing, Alberto Isidori, and Lorenzo Marconi. "Achieving Almost
    Feedback-Linearization via Low-Power Extended Observer." IEEE Control
    Systems Letters 4.4 (2020): 1030-1035.

    Parameters:
        n: relative degree (dimension of subsystem)
        l: high-gain parameter
        K: design paramters, n by 2
        b0: dot(x_n) = f0 + b0u
        F: (A+BF) Hurwitz
        L: saturation value

    '''
    def __init__(self, n, l, K, b0, F, L):
        super().__init__()
        self.xi = BaseSystem(np.zeros((n-1, 2, 1)))
        self.sig = BaseSystem(np.zeros((2, 1)))

        self.n, self.l = n, l
        self.A = np.array([[0, 1], [0, 0]])
        self.N = np.array([[0, 0], [0, 1]])
        self.D = np.diag([l, l**2])
        self.B = np.array([[0], [1]])
        self.C = np.array([[1, 0]])
        self.K = np.reshape(K, (n, 2, 1))  # K(i) = K[i,:,:]
        S = np.zeros((n, 2*n-2))
        S[0, 0] = 1
        S[1, 1] = 1
        for i in range(n-2):
            S[i+2, 2*(i+1)+1] = 1
        self.S = S

        self.b0 = b0
        self.F = F
        self.L = L

    def deriv(self, xi, sig, y, ref):
        n = self.n
        A, N, D, B, C = self.A, self.N, self.D, self.B, self.C
        K, S, F, b0, L = self.K, self.S, self.F, self.b0, self.L
        # observer
        xidot = np.zeros((n-1, 2, 1))
        for i in range(n-2):
            if i == 0:
                xidot[i, :, :] = A.dot(xi[i, :, :]) + N.dot(xi[i+1, :, :]) \
                    + D.dot(K[i, :, :])*(y-C.dot(xi[i, :, :]))
            else:
                xidot[i, :, :] = A.dot(xi[i, :, :]) + N.dot(xi[i+1, :, :]) \
                    + D.dot(K[i, :, :])*(B.T.dot(xi[i-1, :, :])-C.dot(xi[i, :, :]))

        xi_stack = np.reshape(xi, (2*n-2, 1))
        xidot[n-2, :, :] = A.dot(xi[n-2, :, :]) + N.dot(sig) \
            + B*(b0*sat(L, fun_psi(S.dot(xi_stack)-ref, B.T.dot(sig), b0, F))) \
            + D.dot(K[n-2, :, :])*(B.T.dot(xi[n-3, :, :])-C.dot(xi[n-2, :, :]))
        sigdot = A.dot(sig) \
            + C.T*(b0*sat(L, fun_psi(S.dot(xi_stack)-ref, B.T.dot(sig), b0, F))) \
            + D.dot(K[n-1, :, :])*(B.T.dot(xi[n-2, :, :])-C.dot(sig))

        return xidot, sigdot

    def get_virtual(self, t, ref):
        L, S, B, b0, F = self.L, self.S, self.B, self.b0, self.F
        xi_stack = np.reshape(self.xi.state, (2*self.n-2, 1))
        sig = self.sig.state
        ctrl = sat(L, fun_psi(S.dot(xi_stack)-ref, B.T.dot(sig), b0, F))
        return ctrl

    def get_obs(self):
        observation = self.xi.state[0][0][0]
        return observation

    def get_dist(self):
        return self.sig.state[1]

    def set_dot(self, t, y, ref):
        '''
        y: desired observer's real state
        '''
        states = self.observe_list()
        dots = self.deriv(*states, y, ref)
        self.xi.dot, self.sig.dot = dots


class highGainESO(BaseEnv):
    def __init__(self, eps, H, b0, F, L):
        super().__init__()
        self.x = BaseSystem(np.zeros((2, 1)))
        self.sig = BaseSystem(np.zeros((1, 1)))

        self.b0, self.eps, self.F, self.L, self.alp = b0, eps, F, L, H[0, 2]
        self.H = (H[0, 0:2] / np.array([eps, eps**2]))[:, None]
        self.A = np.eye(2, 2, 1)
        self.B = np.array([0, 1])[:, None]
        self.C = np.array([[1, 0]])

    def deriv(self, x, sig, y, ref):
        b0, eps, H, alp = self.b0, self.eps, self.H, self.alp
        A, B, C = self.A, self.B, self.C

        u = self.get_virtual(ref)
        xdot = A.dot(x) + B*(sig + b0*u) + H*(y - C.dot(x))
        sigdot = (alp / eps**3) * (y - C.dot(x))
        return xdot, sigdot

    def get_virtual(self, ref):
        psi = fun_psi(self.x.state-ref, self.sig.state, self.b0, self.F)
        u = self.L * sat(1, psi/self.L)
        return u

    def set_dot(self, t, y, ref):
        states = self.observe_list()
        dots = self.deriv(*states, y, ref)
        self.x.dot, self.sig.dot = dots

    def get_obs(self):
        return self.x.state[0]

    def get_dist(self):
        return self.sig.state


class Controller(BaseEnv):
    def __init__(self, m, g):
        super().__init__()
        self.m, self.g = m, g
        self.u1 = BaseSystem(np.array((m*g)))
        self.du1 = BaseSystem(np.zeros((1)))

        self.F1 = - np.array([[1, 1, 1]])*23.95
        self.F2 = - np.array([[1, 1, 1]])*48.11
        self.F3 = - np.array([[1, 1, 1]])*32.95
        self.F4 = - np.array([[1, 1, 1]])*9.79
        self.F_psi = - np.array([[4, 5, 0, 0]])
        F = np.hstack([self.F1.T, self.F2.T, self.F3.T, self.F4.T])
        self.F = np.vstack([F, self.F_psi])

    def get_virtual(self, t, obs_u):
        d2u1, u2, u3, u4 = obs_u.ravel()
        return d2u1, np.array([u2, u3, u4])[:, None]

    def get_FM(self, ctrl):
        return np.vstack((self.u1.state, ctrl[1]))

    def set_dot(self, ctrl):
        self.u1.dot = self.du1.state
        self.du1.dot = ctrl[0]


if __name__ == "__main__":
    system = lowPowerESO(4, 3, np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), 3, np.array([[1, 2, 3, 4]]))
    system.set_dot(t=0, y=3, L=4)
    print(repr(system))
