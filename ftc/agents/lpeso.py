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

        self.F = F
        self.b0 = b0
        self.L = L

    def deriv(self, xi, sig, y):
        n = self.n
        A, N, D, B, C = self.A, self.N, self.D, self.B, self.C
        K, S, F, b0 = self.K, self.S, self.F, self.b0
        # observer
        b0 = self.b0
        L = self.L
        xidot = np.zeros((n-1, 2, 1))
        for i in range(n-2):
            if i == 0:
                xidot[i, :, :] = A.dot(xi[i, :, :]) + N.dot(xi[i+1, :, :]) \
                    + D.dot(K[i, :, :].dot(y-C.dot(xi[i, :, :])))
            else:
                xidot[i, :, :] = A.dot(xi[i, :, :]) + N.dot(xi[i+1, :, :]) \
                    + D.dot(K[i, :, :].dot(B.T.dot(xi[i-1, :, :])-C.dot(xi[i, :, :])))

        xi_stack = np.reshape(xi, (2*n-2, 1))
        xidot[n-2, :, :] = A.dot(xi[n-2, :, :]) + N.dot(sig) \
            + B.dot(b0*sat(L, fun_psi(S.dot(xi_stack), B.T.dot(sig), b0, F))) \
            + D.dot(K[n-2, :, :].dot(B.T.dot(xi[n-3, :, :])-C.dot(xi[n-2, :, :])))
        sigdot = A.dot(sig) + C.T.dot(b0*sat(L, fun_psi(S.dot(xi_stack), B.T.dot(sig), b0, F))) \
            + D.dot(K[n-1, :, :].dot(B.T.dot(xi[n-2, :, :])-C.dot(sig)))

        ctrl = sat(L, fun_psi(S.dot(xi_stack), B.T.dot(sig), b0, F))

        return xidot, sigdot, ctrl, xi_stack[0]

    def get_virtual(self, t, y):
        states = self.observe_list()
        *_, ctrl, obs = self.deriv(*states, y)
        return ctrl

    def get_dist_obs(self, t, y):
        states = self.observe_list()
        *_, y1 = self.deriv(*states, y)
        disturbance = y - y1
        observation = y1
        return disturbance, observation

    def set_dot(self, t, y):
        '''
        y: desired observer's real state
        '''
        states = self.observe_list()
        dots = self.deriv(*states, y)
        self.xi.dot, self.sig.dot, *_ = dots


if __name__ == "__main__":
    system = lowPowerESO(4, 3, np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), 3, np.array([[1, 2, 3, 4]]))
    system.set_dot(t=0, y=3, L=4)
    print(repr(system))
