import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize


class Grouping():
    def __init__(self, B):
        self.B = B.copy()

    def get(self, fault_index):
        if fault_index in [0, 1]:
            self.B[:, :2] = np.zeros((4, 2))
            G = self.B
        elif fault_index in [2, 3]:
            self.B[:, 2:4] = np.zeros((4, 2))
            G = self.B
        elif fault_index in [4, 5]:
            self.B[:, 4:] = np.zeros((4, 2))
            G = self.B
        return G


class CA():
    def __init__(self, B):
        self.B = B.copy()

    def get(self, What, fault_index=()):
        """Notes
        `fault_index` should be 1d array, e.g., `fault_index = [1]`.
        """
        BB = self.B @ What
        BB[:, fault_index] = np.zeros((4, 1))
        return np.linalg.pinv(BB)


class ConstrainedCA():
    """Reference:
    [1] W. Durham, K. A. Bordignon, and R. Beck, “Aircraft Control Allocation,”
    Aircraft Control Allocation, 2016, doi: 10.1002/9781118827789.

    Method:
    solve_lp: Linear Programming
    solve_opt: SLSQP
    """
    def __init__(self, B):
        self.B = B.copy()
        self.n_rotor = len(B[0])
        self.u_prev = np.zeros((self.n_rotor,))

    def get_faulted_B(self, fault_index):
        _B = np.delete(self.B, fault_index, 1)
        return _B

    def solve_lp(self, fault_index, v, rotor_min, rotor_max):
        n = self.n_rotor - len(fault_index)
        c = np.ones((n,))
        A_ub = np.vstack((-np.eye(n), np.eye(n)))
        b_ub = np.hstack((-rotor_min*np.ones((n,)), rotor_max*np.ones((n,))))
        A_eq = self.get_faulted_B(fault_index)
        b_eq = v.reshape((len(v),))

        sol = linprog(c, A_ub, b_ub, A_eq, b_eq, method="simplex")
        _u = sol.x
        for i in range(len(fault_index)):
            _u = np.insert(_u, fault_index[i], 0)
        return np.vstack(_u)

    def solve_opt(self, fault_index, v, rotor_min, rotor_max):
        n = self.n_rotor - len(fault_index)
        self.u_prev = np.delete(self.u_prev, fault_index)
        bnds = np.hstack((rotor_min*np.zeros((n, 1)), rotor_max*np.ones((n, 1))))
        A_eq = self.get_faulted_B(fault_index)
        b_eq = v.reshape((len(v),))
        cost = (lambda u: np.linalg.norm(u, np.inf)
                + 1e5*np.linalg.norm(b_eq - A_eq.dot(u), 1))

        opts = {"ftol": 1e-5, "maxiter": 1000}
        sol = minimize(cost, self.u_prev, method="SLSQP",
                       bounds=bnds, options=opts)
        _u = sol.x
        for i in range(len(fault_index)):
            _u = np.insert(_u, fault_index[i], 0)
        self.u_prev = _u
        return np.vstack(_u)

    def solve_miae(self, fault_index, v, Lambda, mu, rotor_min, rotor_max):
        nr = self.n_rotor
        bnds = np.hstack((np.zeros((nr, 1)), np.ones((nr, 1))))
        b_eq = v.reshape((len(v),))  # virtual input
        v_d_bar = b_eq - self.B.dot(Lambda.dot(rotor_min*np.ones((6, 1))))
        B_p_bar = self.B.dot((rotor_max-rotor_min)*np.eye(6))
        if len(fault_index) == 0:
            weight = np.array([100, 100, 20, 1])[:, None]**(1/2)
        else:
            weight = np.array([200, 200, 20, 0])[:, None]**(1/2)

        cost = (lambda u:
                # np.linalg.norm(u-self.u_prev, np.inf)  # worse
                + np.linalg.norm((v_d_bar - B_p_bar.dot(Lambda.dot(u))) * weight, 1)
                )

        opts = {"ftol": 1e-5, "maxiter": 1000}
        sol = minimize(cost, self.u_prev, method="SLSQP",
                       bounds=bnds, options=opts)

        _u = (rotor_max-rotor_min)*sol.x + rotor_min
        self.u_prev = _u
        return np.vstack(_u)


if __name__ == "__main__":
    pass
