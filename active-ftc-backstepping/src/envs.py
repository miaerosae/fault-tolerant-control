import numpy as np
import scipy
from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import angle2dcm, dcm2quat, quat2angle
import fym.logging as logging


def dcm2angle(rot):
    return quat2angle(dcm2quat(rot))

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

"""
# Refs.
- Hexacopter information
[1] V. S. Akkinapalli, G. P. Falconí, and F. Holzapfel,
“Attitude control of a multicopter using L1 augmented quaternion based backstepping,”
Proceeding - ICARES 2014 2014 IEEE Int. Conf. Aerosp. Electron. Remote Sens. Technol.,
no. November, pp. 170–178, 2014.
[2] M. C. Achtelik, K. M. Doth, D. Gurdan, and J. Stumpf,
“Design of a multi rotor MAV with regard to efficiency, dynamics and redundancy,”
AIAA Guid. Navig. Control Conf. 2012, no. August, pp. 1–17, 2012.

# Variables
pos in R^{3}: position in I-frame (inertial)
vel in R^{3}: velocity w.r.t. I-frame
rot in R^{3 x 3}: R_{BI}, i.e., rotation matrix "from I- to B-frame" (see the ref.)
omega in R^{3}: angular rate of the B-frame relative to I-frame
nu in R^{4}: nu = [T M^{T}]^{T}, virtual input

# Parameters
m in R: mass
g in R^{3}: (constant) gravitational acceleration
J in R^{3 x 3}: moment of inertia
kT, kM: effectiveness constants
ll: length
"""
class HexacopterEnv(BaseEnv):
    # m = 0.64  # kg
    m = 4.34  # kg
    g = np.array([0, 0, 9.81])  # m/s^2
    # J = np.array([[0.010007, 0, 0],
    #               [0, 0.0102335, 0],
    #               [0, 0, 0.0081]])
    J = np.array([[0.0820, 0, 0],
                  [0, 0.0845, 0],
                  [0, 0, 0.1377]])
    J_inv = np.linalg.inv(J)
    # kT = 6.546e-6  # N s^2 rad^-1
    kT = 1.0  # N s^2 rad^-1
    # kM = 1.2864e-7  # N m s^2 rad^-1
    kM = 8.004e-4  # N m s^2 rad^-1
    # ll = 0.215  # m
    ll = 0.0315  # m
    B = np.array([[kT, kT, kT, kT, kT, kT],
                  [0.5*ll*kT, ll*kT, 0.5*ll*kT, -0.5*ll*kT, -ll*kT, -0.5*ll*kT],
                  [0.5*np.sqrt(3)*ll*kT, 0, -0.5*np.sqrt(3)*ll*kT, -0.5*np.sqrt(3)*ll*kT, 0, 0.5*np.sqrt(3)*ll*kT],
                  [kM, -kM, kM, -kM, kM, -kM]])
    B_pinv = np.linalg.pinv(B)
    def __init__(self, pos0, vel0, rot0, omega0, Lambda=lambda t: np.eye(6), **kwargs):
        super().__init__(**kwargs)
        self.pos = BaseSystem(pos0)
        self.vel = BaseSystem(vel0)
        self.rot = BaseSystem(rot0)
        self.omega = BaseSystem(omega0)
        self.Lambda = Lambda

    def reset(self):
        super().reset()

    def observe(self):
        return self.pos.state, self.vel.state, self.rot.state, self.omega.state

    def saturate(self, u):
        return np.clip(u, 0.0, 2*(self.m*np.linalg.norm(self.g)/self.kT)/6)

    def dynamics(self, t, vel, rot, omega, u):
        # force and moment
        nu = self.B @ self.Lambda(t) @ u
        M = nu[1:]
        zB = rot.T @ [0, 0, 1]
        thrust = -nu[0] * zB
        # derivatives
        pos_dot = vel
        vel_dot = (1/self.m)*thrust + self.g
        rot_dot = -skew(omega) @ rot
        omega_dot = (-self.J_inv @ np.cross(omega, self.J@omega) + (self.J_inv @ M))
        return pos_dot, vel_dot, rot_dot, omega_dot

    def set_dot(self, t):
        u = 10000*np.array([1, 2, 3, 4, 5, 6])  # TODO
        vel = self.vel.state
        rot = self.rot.state
        omega = self.omega.state
        self.pos.dot, self.vel.dot, self.rot.dot, self.omega.dot = self.dynamics(t, vel, rot, omega, u)

    def step(self):
        t = self.clock.get()
        pos, vel, rot, omega = self.observe()
        u = np.ones(6)  # TODO
        nu = self.B @ self.Lambda(t) @ u
        M = nu[1:3]
        zB = rot.T @ [0, 0, 1]
        thrust = -nu[0] * zB
        angle = dcm2angle(rot)
        info = dict(t=t, pos=pos, vel=vel, rot=rot, rot_flatten=rot.flatten(), omega=omega, angle=angle, thrust=thrust, M=M, nu=nu)
        self.update()  # update
        next_obs = dict(pos=self.pos.state, vel=self.vel.state, rot=self.rot.state, omega=self.omega.state)
        reward = np.zeros(1)
        done = self.clock.time_over()
        return next_obs, reward, info, done


def T_omega(T):
    return np.array([[0, -T, 0],
                     [T, 0, 0],
                     [0, 0, 0]])
def T_u_inv(T):
    return np.array([[0, 1/T, 0], [-1/T, 0, 0], [0, 0, -1]])

def T_u_inv_dot(T, T_dot):
    return np.array([[0, -T_dot/(T**2), 0],
                     [T_dot/(T**2), 0, 0],
                     [0, 0, 0]])

"""
# Refs.
- Controller
[1] G. P. Falconi and F. Holzapfel,
“Adaptive Fault Tolerant Control Allocation for a Hexacopter System,”
Proc. Am. Control Conf., vol. 2016-July, pp. 6760–6766, 2016.
- Reference model (e.g., xd, vd, ad, ad_dot, ad_ddot)
[2] S. J. Su, Y. Y. Zhu, H. R. Wang, and C. Yun, “A Method to Construct a Reference Model for Model Reference Adaptive Control,” Adv. Mech. Eng., vol. 11, no. 11, pp. 1–9, 2019.
"""
class BacksteppingController(BaseEnv):
    def __init__(self, pos0, m, grav, **kwargs):
        super().__init__(**kwargs)
        self.xd = BaseSystem(pos0)
        self.vd = BaseSystem(np.zeros(3))
        self.ad = BaseSystem(np.zeros(3))
        self.ad_dot = BaseSystem(np.zeros(3))
        self.ad_ddot = BaseSystem(np.zeros(3))
        self.Td = BaseSystem(m*grav)
        # position
        self.Kx = m*1*np.eye(3)
        self.Kv = m*1*1.82*np.eye(3)
        self.Kp = np.hstack([self.Kx, self.Kv])
        self.Q = np.diag(1*np.ones(6))
        # thrust
        self.Kt = np.diag(4*np.ones(3))
        # angular
        self.Komega = np.diag(20*np.ones(3))
        # reference model
        self.Kxd = np.diag(1*np.ones(3))
        self.Kvd = np.diag(3.4*np.ones(3))
        self.Kad = np.diag(5.4*np.ones(3))
        self.Kad_dot = np.diag(4.9*np.ones(3))
        self.Kad_ddot = np.diag(2.7*np.ones(3))

    def reset(self):
        super().reset()

    def observe(self):
        return self.xd.state, self.vd.state, self.ad.state, self.ad_dot.state, self.ad_ddot.state, self.Td.state

    def dynamics(self, xd, vd, ad, ad_dot, ad_ddot, Td_dot, xc):
        d_xd = vd
        d_vd = ad
        d_ad = ad_dot
        d_ad_dot = ad_ddot
        d_ad_ddot = (-self.Kxd @ xd -self.Kvd @ vd - self.Kad @ ad - self.Kad_dot @ ad_dot - self.Kad_ddot @ ad_ddot + self.Kxd @ xc)
        d_Td = Td_dot
        return d_xd, d_vd, d_ad, d_ad_dot, d_ad_ddot, d_Td

    def set_dot(self, t):
        xd, vd, ad, ad_dot, ad_ddot, Td = self.observe()
        xc = np.ones(3)  # TODO
        Td_dot = np.ones(1)  # TODO
        self.xd.dot, self.vd.dot, self.ad.dot, self.ad_dot.dot, self.ad_ddot.dot, self.Td.dot = self.dynamics(xd, vd, ad, ad_dot, ad_ddot, Td_dot, xc)

    def command(self, pos, vel, rot, omega,
                      xd, vd, ad, ad_dot, ad_ddot, Td,
                      m, J, g):
        ex = xd - pos
        ev = vd - vel
        ep = np.hstack([ex, ev])
        # u1
        u1 = m * (ad - g) + self.Kp @ ep
        zB = rot.T @ [0, 0, 1]
        td = -Td * zB
        et = u1 - td
        Ap = np.vstack([np.hstack([np.zeros((3, 3)), np.eye(3)]),
                        np.hstack([-(1/m)*self.Kx, -(1/m)*self.Kv])])
        Bp = np.vstack([np.zeros((3, 3)), (1/m)*np.eye(3)])
        ep_dot = Ap @ ep + Bp @ et
        u1_dot = m * ad_dot + self.Kp @ ep_dot
        P = scipy.linalg.solve_lyapunov(Ap.T, -self.Q)
        T = Td  # TODO: no lag
        # u2
        u2 = T_u_inv(T[0]) @ rot @ (2*Bp.T @ P @ ep + u1_dot + self.Kt @ et)
        Td_dot = u2[-1]  # third element
        T_dot = Td_dot  # TODO: no lag
        zB_dot = -rot.T @ T_omega(1.0) @ omega
        et_dot = u1_dot + u2[-1] * zB + Td * zB_dot
        ep_ddot = Ap @ ep_dot + Bp @ et_dot
        u1_ddot = m * ad_ddot + self.Kp @ ep_ddot
        rot_dot = -skew(omega) @ rot
        u2_dot = (T_u_inv_dot(T[0], T_dot) @ rot + T_u_inv(T[0]) @ rot_dot) @ (2*Bp.T @ P @ ep + u1_dot + self.Kt@et) + (T_u_inv(T[0]) @ rot @ (2*Bp.T @ P @ ep_dot + u1_ddot + self.Kt @ et_dot))
        omegad_dot = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]]) @ u2_dot
        omegad = np.array([u2[0], u2[1], 0])
        eomega = omegad - omega
        Md = np.cross(omega, J@omega) + J @ (T_omega(T[0]).T @ rot @ et + omegad_dot + self.Komega @ eomega)
        nud = np.hstack([Td, Md])
        import ipdb; ipdb.set_trace()
        return nud, Td_dot

    def step(self):
        t = self.clock.get()
        xd, vd, ad, ad_dot, ad_ddot, Td = self.observe()
        info = dict(t=t, xd=xd, vd=vd, ad=ad, ad_dot=ad_dot, ad_ddot=ad_ddot, Td=Td)
        self.update()  # update
        done = self.clock.time_over()
        next_obs = self.observe()
        return next_obs, np.zeros(1), info, done


"""
# Notes
This is implemented as the simplest form of actuator FDI.
It may not make sense in practice.
"""
class ActuatorFDIFirstOrderLagEnv(BaseEnv):
    tau = 0.10
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Lambda_hat = BaseSystem(np.eye(6))

    def reset(self):
        super().reset()

    def observe(self):
        return self.Lambda_hat.state

    def dynamics(self, Lambda, Lambda_hat):
        return (Lambda - Lambda_hat) / self.tau

    def set_dot(self, t):
        Lambda_hat = self.Lambda_hat.state
        Lambda = 0.5*np.eye(6)  # TODO
        self.Lambda_hat.dot = self.dynamics(Lambda, Lambda_hat)

    def step(self):
        t = self.clock.get()
        Lambda_hat = self.Lambda_hat.state
        Lambda_hat_flatten = Lambda_hat.flatten()
        Lambda = 0.5*np.eye(6)  # TODO
        Lambda_flatten = Lambda.flatten()
        info = dict(t=t, Lambda=Lambda, Lambda_hat=Lambda_hat, Lambda_flatten=Lambda_flatten, Lambda_hat_flatten=Lambda_hat_flatten)
        self.update()
        next_obs = self.observe()
        done = self.clock.time_over()
        return next_obs, np.zeros(1), info, done


"""
# Notes
Integrated environment; see each component environment for more details.
"""
class HexacopterBacksteppingControllerEnv(BaseEnv):
    def __init__(self, hexa, ctrl, fdi, position_commander=lambda t: np.array([1, 2, 3]), **kwargs):
        super().__init__(**kwargs)
        self.hexa = hexa
        self.ctrl = ctrl
        self.fdi = fdi
        self.position_commander = position_commander

    def reset(self):
        super().reset()

    def set_dot(self, t):
        xc = self.position_commander(t)
        hexa, ctrl, fdi = self.hexa, self.ctrl, self.fdi
        Lambda = hexa.Lambda(t)
        # hexa
        pos, vel, rot, omega = hexa.observe()
        # ctrl
        xd, vd, ad, ad_dot, ad_ddot, Td = ctrl.observe()
        # fdi
        Lambda_hat = fdi.observe()
        # command
        nud, Td_dot = self.ctrl.command(pos, vel, rot, omega,
                                        xd, vd, ad, ad_dot, ad_ddot, Td,
                                        hexa.m, hexa.J, hexa.g)
        # u_raw = hexa.B_pinv @ nud
        u_raw = np.linalg.pinv(hexa.B @ Lambda_hat) @ nud
        u = hexa.saturate(u_raw)
        # dot (hexa)
        hexa.pos.dot, hexa.vel.dot, hexa.rot.dot, hexa.omega.dot = hexa.dynamics(t, vel, rot, omega, u)
        # dot (ctrl)
        ctrl.xd.dot, ctrl.vd.dot, ctrl.ad.dot, ctrl.ad_dot.dot, ctrl.ad_ddot.dot, ctrl.Td.dot = ctrl.dynamics(xd, vd, ad, ad_dot, ad_ddot, Td_dot, xc)
        fdi.Lambda_hat.dot = fdi.dynamics(Lambda, Lambda_hat)

    def step(self):
        t = self.clock.get()
        hexa, ctrl, fdi = self.hexa, self.ctrl, self.fdi
        Lambda = hexa.Lambda(t)
        Lambda_flatten = Lambda.flatten()
        Lambda_diagonal = Lambda.diagonal()
        pos, vel, rot, omega = hexa.observe()
        xd, vd, ad, ad_dot, ad_ddot, Td = ctrl.observe()
        Lambda_hat = fdi.observe()
        Lambda_hat_flatten = Lambda_hat.flatten()
        Lambda_hat_diagonal = Lambda_hat.diagonal()
        nud, Td_dot = ctrl.command(pos, vel, rot, omega,
                                   xd, vd, ad, ad_dot, ad_ddot, Td,
                                   hexa.m, hexa.J, hexa.g)
        # u_raw = hexa.B_pinv @ nud
        u_raw = np.linalg.pinv(hexa.B @ Lambda_hat) @ nud
        u = hexa.saturate(u_raw)
        nu = nud  # no lag
        T = nu[0]
        M = nu[1:]
        zB = rot.T @ [0, 0, 1]
        thrust = -nu[0] * zB
        ex = xd - pos
        xc = self.position_commander(t)
        x, y, z = pos
        angle = dcm2angle(rot)
        info = dict(t=t,
                    Lambda_flatten=Lambda_flatten,
                    Lambda_diagonal=Lambda_diagonal,
                    Lambda_hat_flatten=Lambda_hat_flatten,
                    Lambda_hat_diagonal=Lambda_hat_diagonal,
                    ex=ex,
                    pos=pos, vel=vel, rot=rot, rot_flatten=rot.flatten(),
                    omega=omega, angle=angle,
                    nu=nu, u=u, T=T, M=M, thrust=thrust,
                    xc=xc, xd=xd, vd=vd, ad=ad, ad_dot=ad_dot,
                    ad_ddot=ad_ddot, Td=Td,)
        self.update()  # update
        next_obs = hexa.observe(), ctrl.observe()
        done = self.clock.time_over()
        return next_obs, np.zeros(1), info, done


if __name__ == "__main__":
    pos0 = np.zeros(3)
    vel0 = np.zeros(3)
    rot0 = np.eye(3)
    omega0 = np.zeros(3)
    env = Env(pos0, vel0, rot0, omega0)
    env.reset()
