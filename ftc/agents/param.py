'''
define b0 of hexacopter used for low-power ESO
'''
import numpy as np
import math
from numpy import sin, cos
import fym
import ftc.config

cfg = ftc.config.load()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_b0(m, g, J):
    b0 = np.array([-g/J[1, 1], g/J[0, 0], -1/m, 1/J[2, 2]])
    return b0


def get_B_hat(m, g, J):
    B_hat = np.array([[0, 0, -g/J[1, 1], 0],
                      [0, g/J[0, 0], 0, 0],
                      [-1/m, 0, 0, 0],
                      [0, 0, 0, 1/J[2, 2]]])
    return B_hat


def get_B(m, J, euler, forces):
    phi, theta, psi = euler
    u1, *_ = forces
    B = np.zeros((4, 4))
    B[0, 0] = - (sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta))/m
    B[0, 1] = (-cos(phi)*sin(psi) + cos(psi)*sin(phi)*sin(theta))*u1/J[0, 0]/m
    B[0, 2] = -cos(phi)*cos(psi)*cos(theta)*u1/J[1, 1]/m
    B[0, 3] = (-cos(psi)*sin(phi) + cos(phi)*sin(psi)*sin(theta))*u1/J[2, 2]/m
    B[1, 0] = (cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta))/m
    B[1, 1] = (cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta))*u1/J[0, 0]/m
    B[1, 2] = - cos(phi)*cos(theta)*sin(psi)*u1/J[1, 1]/m
    B[1, 3] = - (sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta))*u1/J[2, 2]/m
    B[2, 0] = - cos(phi)*cos(theta)/m
    B[2, 1] = cos(theta)*sin(phi)*u1/J[0, 0]/m
    B[2, 2] = cos(phi)*sin(theta)*u1/J[1, 1]/m
    B[3, 3] = 1/J[2, 2]
    return B


def get_K():
    K = np.array([[23, 484],
                  [23, 183.1653],
                  [23, 80.3686],
                  [23, 30.0826],
                  [23, 0]])
    return K


def get_K_ls():
    # satisfy low-power strong stability condition
    K = np.array([[2.9, 10.1],
                  [2.9, 4.0387],
                  [2.9, 2.0187],
                  [2.9, 1.009],
                  [2.9, 0.4035]])
    return K


def get_uncertainties(t, uncertainty):
    upos = np.zeros((3, 1))
    uvel = np.zeros((3, 1))
    ueuler = np.zeros((3, 1))
    uomega = np.zeros((3, 1))
    if uncertainty is True:
        # upos와 ueuler가 어떤 의미가 있지..?
        upos = np.vstack([
            0.1*np.cos(0.2*t),
            0.2*np.sin(0.5*np.pi*t),
            0.3*np.cos(t),
        ])
        uvel = np.vstack([
            0.1*np.sin(t) - 0.2,
            0.2*np.sin(np.pi*t),
            0.2*np.sin(3*t) - 0.1*np.sin(0.5*np.pi*t)
        ])
        ueuler = np.vstack([
            0.2*np.sin(t),
            0.1*np.cos(np.pi*t+np.pi/4),
            0.2*np.sin(0.5*np.pi*t),
        ])
        uomega = np.vstack([
            - 0.2*np.sin(0.5*np.pi*t) + 0.1,
            0.1*np.cos(np.sqrt(2)*t),
            0.1*np.cos(2*t+1) + 0.05
        ])
    return upos, uvel, ueuler, uomega


def get_sumOfDist(t, condi):
    pi = np.pi
    ref_dist = np.zeros((6, 1))
    ref_dist[0] = - (- pi/5*np.cos(t/2)*np.sin(pi*t/5)
                     - (1/4 + pi**2/25)*np.sin(t/2)*np.cos(pi*t/5)) * np.cos(np.pi/4)
    ref_dist[1] = - (pi/5*np.cos(t/2)*np.cos(pi*t/5)
                     - (1/4 + pi**2/25)*np.sin(t/2)*np.sin(pi*t/5)) * np.cos(np.pi/4)

    if condi is True:
        ext_dist = np.zeros((6, 1))
        m1, m2, m3, m4 = get_uncertainties(t, True)
        ext_dist[0:3] = m2
        ext_dist[3:6] = m4
        int_dist = np.vstack([- 0.1*0.2*np.sin(0.2*t),
                              0.2*0.5*pi*np.cos(0.5*pi*t),
                              - 0.3*np.sin(t),
                              0.2*np.cos(t),
                              - 0.1*pi*np.sin(pi*t+pi/4),
                              0.2*0.5*pi*np.cos(0.5*pi*t)])
        ref_dist = ref_dist + ext_dist + int_dist
    return ref_dist


def get_W(t, fault):
    if fault is True:
        if t > 5:
            W1 = 0.5
        # elif t > 3:
        #     W1 = (- 40/17**2 * (t+14) * (t-20) + 40) * 0.01
        else:
            W1 = 1

        if t > 7:
            W2 = 0.8
        else:
            W2 = 1

        if t > 11:
            W3 = 0.3
        elif t > 6:
            W3 = 0.7/25*(t-11)**2 + 0.3
        else:
            W3 = 1

        # if t > 25:
        #     W4 = 0.5
        # else:
        #     W4 = 1
        W = np.diag([W1, W2, W3, 1])

    # else:
    #     if t > 3:
    #         W = np.diag([0.6, 1, 1, 1])
    #     else:
    #         W = np.diag([1, 1, 1, 1])
    else:
        W = np.diag([1, 1, 1, 1])
    return W


def get_PID_gain(param):
    '''
    convert BLF parameter(k1, k2, k3) to PID gain
    '''
    kpos = param.Kxy
    kang = param.Kang
    rhoinf = param.oL.rho[1]
    kP1 = kpos[0]*kpos[1] + kpos[2]*rhoinf**2 + 1/rhoinf**2
    kD1 = kpos[0] + kpos[1]
    kI1 = kpos[1]*kpos[2]*rhoinf**2
    kP2 = kang[0]*kang[1] + kang[2]
    kD2 = kang[0] + kang[1]
    kI2 = kang[1]*kang[2]
    return np.array([kP1, kD1, kI1]), np.array([kP2, kD2, kI2])


def get_PID_gain_reverse(config, param):
    '''
    change PID gain to BLF parameter
    '''
    kP1, kD1, kI1 = config["k11"], config["k12"], config["k13"]
    kP2, kD2, kI2 = config["k21"], config["k22"], config["k23"]
    # position
    rhoinf = param.oL.rho[1]
    k2pos = np.roots([1, -kD1, kP1+1/rhoinf**2, -kI1])
    k1pos = k3pos = []
    for i in k2pos:
        k1pos.append(kD1 - k2pos[i])
        k3pos.append(kI1 / k2pos[i] / rhoinf**2)
    k2ang = np.roots([1, -kD2, kP2, -kI2])
    k1ang = k3ang = []
    for i in k2ang:
        k1ang.append(kD2 - k2ang[i])
        k3ang.append(kI2 / k2ang[i])
    return k1pos, k2pos, k3pos, k1ang, k2ang, k3ang
