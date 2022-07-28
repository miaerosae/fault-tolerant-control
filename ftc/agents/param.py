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
    if self.uncertainty is True:
        upos = np.vstack([
            0.5*np.cos(2*np.pi*t),
            0.2*np.sin(0.5*np.pi*t),
            0.3*stable_sigmoid(t),
        ])
        uvel = np.vstack([
            0.8*stable_sigmoid(t),
            0.5*np.sin(np.pi*t),
            0.2*np.sin(3*t) + 0.1*np.sin(0.5*np.pi*t)
        ])
        ueuler = np.vstack([
            0.1*np.tanh(np.sqrt(2)*t),
            0.3*np.cos(2*t+1) + 0.2*stable_sigmoid(t),
            - 0.5*np.sin(5*np.pi*t),
        ])
        uomega = np.vstack([
            0.2*stable_sigmoid(t) - 0.5*np.sin(0.5*np.pi*t),
            0.5*np.tanh(np.sqrt(2)*t),
            0.3*np.cos(2*t+1)
        ])
    return upos, uvel, ueuler, uomega
