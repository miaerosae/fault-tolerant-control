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
            0.1*np.cos(2*np.pi*t),
            0.2*np.sin(0.5*np.pi*t),
            0.3*stable_sigmoid(t),
        ])
        uvel = np.vstack([
            0.1*stable_sigmoid(t),
            0.2*np.sin(np.pi*t),
            0.2*np.sin(3*t) - 0.1*np.sin(0.5*np.pi*t)
        ])
        ueuler = np.vstack([
            0.3*stable_sigmoid(t),
            0.1*np.cos(np.pi*t+np.pi/4),
            0.2*np.sin(0.5*np.pi*t),
        ])
        uomega = np.vstack([
            0.2*stable_sigmoid(t) - 0.4*np.sin(0.5*np.pi*t),
            0.1*np.tanh(np.sqrt(2)*t),
            0.1*np.cos(2*t+1)
        ])
    return upos, uvel, ueuler, uomega


def get_W(t):
    if t > 20:
        W1 = 0.4
    elif t > 3:
        W1 = (- 40/17**2 * (t+14) * (t-20) + 40) * 0.01
    else:
        W1 = 1

    if t > 11:
        W2 = 0.7
    elif t > 6:
        W2 = (6/5 * (t-11)**2 + 70) * 0.01
    else:
        W2 = 1

    if t > 10:
        W3 = 0.9
    else:
        W3 = 1

    if t > 25:
        W4 = 0.5
    else:
        W4 = 1

    W = np.diag([W1, W2, W3, W4])
    return W


def get_What(t, delay):
    if t > 20 + delay:
        W1 = 0.4
    elif t > 3 + delay:
        W1 = (- 40/17**2 * (t+14) * (t-20) + 40) * 0.01
    else:
        W1 = 1

    if t > 11 + delay:
        W2 = 0.7
    elif t > 6 + delay:
        W2 = (6/5 * (t-11)**2 + 70) * 0.01
    else:
        W2 = 1

    if t > 10 + delay:
        W3 = 0.9
    else:
        W3 = 1

    if t > 25 + delay:
        W4 = 0.5
    else:
        W4 = 1

    What = np.diag([W1, W2, W3, W4])
    return What


def get_faulty_input(W, rotors):
    return W.dot(rotors)
