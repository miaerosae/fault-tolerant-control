'''
define b0 of hexacopter used for low-power ESO
'''
import numpy as np
import math
from numpy import sin, cos


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


def stable_sigmoid(x):

    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig


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
            0.1*np.sin(t),
            0.2*np.sin(np.pi*t),
            0.2*np.sin(3*t) - 0.1*np.sin(0.5*np.pi*t)
        ])
        ueuler = np.vstack([
            0.2*np.sin(t),
            0.1*np.cos(np.pi*t+np.pi/4),
            0.2*np.sin(0.5*np.pi*t),
        ])
        uomega = np.vstack([
            - 0.2*np.sin(0.5*np.pi*t),
            0.1*np.cos(np.sqrt(2)*t),
            0.1*np.cos(2*t+1)
        ])
    return upos, uvel, ueuler, uomega


def get_sumOfDist(t, condi):
    pi = np.pi
    ref_dist = np.zeros((6, 1))
    ref_dist[0] = - (- pi/5*np.cos(t/2)*np.sin(pi*t/5)
                     - (1/4 + pi**2/25)*np.sin(t/2)*np.cos(pi*t/5))
    ref_dist[1] = - (pi/5*np.cos(t/2)*np.cos(pi*t/5)
                     - (1/4 + pi**2/25)*np.sin(t/2)*np.sin(pi*t/5))

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

        # if t > 11:
        #     W3 = 0.3
        # elif t > 6:
        #     W3 = 0.7/25*(t-11)**2 + 0.3
        # else:
        #     W3 = 1

        # if t > 25:
        #     W4 = 0.5
        # else:
        #     W4 = 1
        W = np.diag([W1, W2, 1, 1])

    # else:
    #     if t > 3:
    #         W = np.diag([0.6, 1, 1, 1])
    #     else:
    #         W = np.diag([1, 1, 1, 1])
    else:
        W = np.diag([1, 1, 1, 1])
    return W


def get_faulty_input(W, rotors):
    return W.dot(rotors)
