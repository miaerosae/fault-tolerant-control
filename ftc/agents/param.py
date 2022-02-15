'''
define b0 of hexacopter used for low-power ESO
'''
import numpy as np


def get_b0(m, g, J):
    b0 = np.array([-g/J[1, 1], g/J[0, 0], -1/m, 1/J[2, 2]])
    return b0
