from fym.core import BaseEnv, BaseSystem

import numpy as np


def sat(L, x):
    result = np.zeros((np.shape(x)))
    for i in range(np.size(x)):
        if x[i] > L:
            result[i] = L
        elif x[i] < -L:
            result[i] = -L
        else:
            result[i] = x[i]
    return result


def fun_psi(Sxi, Bsig, b0, F):
    return 1 / b0 * (- Bsig + F.dot(Sxi))


class lowPowerESO(BaseEnv):
    '''
    References
    Lei W., Christopher M. K., "Robust Output Feedback Stabilization of MIMO
    Invertible Nonlinear Systems with Output-Dependent Multipliers", IEEE
    Transactions on Automatic Control, 2021

    Parameters:
        n: relative degree matrix (dimension of subsystems)
           for now, all of n[i]s' should have same value
        l: high-gain parameter
        K: design paramters, n by 2
        B_hat: B in trim condition
        F: (A+BF) Hurwitz
        L: saturation function value

    '''
    def __init__(self, n, l, K, B_hat, F, L, F_psi, K_psi):
        super().__init__()
        nu = int(n[0])  # number of subdimension
        self.xi = BaseSystem(np.zeros((np.size(n), nu, 2, 1)))
        self.psi = BaseSystem(np.zeros((2, 2, 1)))

        self.n, self.l = n, l
        self.A = np.array([[0, 1], [0, 0]])
        self.N = np.array([[0, 0], [0, 1]])
        self.D = np.diag([l, l**2])
        self.B = np.array([[0], [1]])
        self.C = np.array([[1, 0]])
        self.K = np.reshape(K, (nu, 2, 1))  # K(i) = K[i,:,:]
        S = np.zeros((np.size(n)*nu+2, 2*(np.size(n)*nu+2)))
        for i in range(np.size(n)*nu+2):
            S[i, 2*i] = 1
        self.S = S

        self.B_hat = B_hat
        self.B_hat_inv = np.linalg.inv(B_hat)
        self.F = F
        self.L = L

        self.F_psi = F_psi
        self.K_psi = K_psi

        self.nu = nu

    def deriv(self, xi, psi, y, ref):
        n, nu = self.n, self.nu
        A, N, D, B, C = self.A, self.N, self.D, self.B, self.C
        K, B_hat = self.K, self.B_hat
        K_psi = self.K_psi
        # observer
        xidot = np.zeros((np.size(n), nu, 2, 1))
        u = self.get_virtual(ref)
        for j in range(np.size(n)):
            for i in range(nu-2):
                if i == 0:
                    xidot[j, i, :, :] = A.dot(xi[j, i, :, :]) + N.dot(xi[j, i+1, :, :]) \
                        + D.dot(K[i, :, :])*(y[j]-C.dot(xi[j, i, :, :]))
                else:
                    xidot[j, i, :, :] = A.dot(xi[j, i, :, :]) + N.dot(xi[j, i+1, :, :]) \
                        + D.dot(K[i, :, :])*(B.T.dot(xi[j, i-1, :, :])-C.dot(xi[j, i, :, :]))

            xidot[j, nu-2, :, :] = A.dot(xi[j, nu-2, :, :]) + N.dot(xi[j, nu-1, :, :]) \
                + B*(B_hat[j, :].dot(u)) \
                + D.dot(K[nu-2, :, :])*(B.T.dot(xi[j, nu-3, :, :])-C.dot(xi[j, nu-2, :, :]))
            xidot[j, nu-1, :, :] = A.dot(xi[j, nu-1, :, :]) \
                + C.T*(B_hat[j, :].dot(u)) \
                + D.dot(K[nu-1, :, :])*(B.T.dot(xi[j, nu-2, :, :])-C.dot(xi[j, nu-1, :, :]))

        psidot = np.zeros((2, 2, 1))
        psidot[0, :, :] = A.dot(psi[0, :, :]) + N.dot(psi[1, :, :]) \
            + B*(B_hat[3, :].dot(u)) \
            + D.dot(K_psi[0, :][:, None])*(y[3]-C.dot(psi[0, :, :]))
        psidot[1, :, :] = A.dot(psi[1, :, :]) + C.T*(B_hat[3, :].dot(u)) \
            + D.dot(K_psi[1, :][:, None])*(B.T.dot(psi[0, :, :]) - C.dot(psi[1, :, :]))

        return xidot, psidot

    def get_virtual(self, ref):
        n, nu = self.n, self.nu
        F, L, S, B_hat_inv = self.F, self.L, self.S, self.B_hat_inv
        F_psi = self.F_psi
        sn = np.size(n)
        xi_stack = np.reshape(self.xi.state, [sn*nu*2, 1])
        psi_stack = np.reshape(self.psi.state, [4, 1])
        stack = np.vstack([xi_stack, psi_stack])
        xi = S.dot(stack)
        F_blk = np.block([[F, np.zeros((1, 2*nu+2))],
                          [np.zeros((1, nu)), F, np.zeros((1, nu+2))],
                          [np.zeros((1, 2*nu)), F, np.zeros((1, 2))],
                          [np.zeros((1, 3*nu)), F_psi]])
        sig_xi = np.reshape(self.xi.state[:, nu-1, 1, :], [sn, 1])
        sig_psi = self.psi.state[1, 1, :]
        sig = np.vstack([sig_xi, sig_psi])
        u = B_hat_inv.dot(sat(L, -sig+F_blk.dot(xi-ref)))
        # u = (B_hat_inv.dot(-sig+F_blk.dot(xi-ref)))
        return u

    def get_obs(self):
        sn = np.size(self.n)
        observe_xyz = np.reshape(self.xi.state[:, 0, 0, :], [sn, 1])
        observe = np.vstack([observe_xyz, self.psi.state[0, 0, :]])
        return observe

    def set_dot(self, t, y, ref):
        '''
        y: desired observer's real state
        '''
        states = self.observe_list()
        self.xi.dot, self.psi.dot = self.deriv(*states, y, ref)


class Controller(BaseEnv):
    def __init__(self, m, g):
        super().__init__()
        self.m, self.g = m, g
        self.u1 = BaseSystem(np.array((m*g)))
        self.du1 = BaseSystem(np.zeros((1)))

        self.F = - np.array([[23.95, 48.11, 32.95, 9.79]])
        self.F_psi = - np.array([[4, 5]])

    def get_virtual(self, t, obs_u):
        d2u1, u2, u3, u4 = obs_u.ravel()
        return d2u1, np.array([u2, u3, u4])[:, None]

    def get_FM(self, ctrl):
        return np.vstack((self.u1.state, ctrl[1]))

    def set_dot(self, ctrl):
        self.u1.dot = self.du1.state
        self.du1.dot = ctrl[0]


if __name__ == "__main__":
    system = lowPowerESO(4*np.ones((3)), 3, np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), np.eye(3), np.array([[1, 2, 3, 4]]), 50)
    system.set_dot(t=0, y=3, ref=np.ones((12, 1)))
    print(repr(system))
