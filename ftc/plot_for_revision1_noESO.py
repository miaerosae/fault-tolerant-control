import numpy as np
import matplotlib.pyplot as plt
import fym
from fym.utils.rot import angle2quat, quat2angle
from ftc.agents.param import get_uncertainties
import ftc.config
from ftc.agents.param import get_sumOfDist

cfg = ftc.config.load()


def exp_plot(loggerpath1, lp2, lp3):
    data1, info = fym.load(loggerpath1, with_info=True)  # base
    data2 = fym.load(lp2)  # attitude
    data3 = fym.load(lp3)  # position error

    # observation: position error
    plt.figure()
    rho1 = np.array([1.5, 0.2])
    rho3 = np.array([1.5, 1])
    rho_k = 1
    pos_bounds1 = np.zeros((np.shape(data1["x"]["pos"][:, 0, 0])[0]))  # BLF-1, 2
    pos_bounds3 = np.zeros((np.shape(data1["x"]["pos"][:, 0, 0])[0]))  # BLF-3
    for i in range(np.shape(data1["x"]["pos"][:, 0, 0])[0]):
        pos_bounds1[i] = (rho1[0]-rho1[1]) * np.exp(-rho_k*data1["t"][i]) + rho1[1]
        pos_bounds3[i] = (rho3[0]-rho3[1]) * np.exp(-rho_k*data3["t"][i]) + rho3[1]
    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip([r"$e_{1x} [m]$", r"$e_{1y} [m]$", r"$e_{1z} [m]$"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], data1["x"]["pos"][:, i, 0]-data1["ref"][:, i, 0], "k--", label="BLF-1")
        plt.plot(data2["t"], data2["x"]["pos"][:, i, 0]-data2["ref"][:, i, 0], "b-", label="BLF-2")
        plt.plot(data3["t"], data3["x"]["pos"][:, i, 0]-data3["ref"][:, i, 0], "g.-", label="BLF-3")
        plt.plot(data1["t"], pos_bounds1, "b:")
        plt.plot(data1["t"], -pos_bounds1, "b:")
        plt.plot(data3["t"], pos_bounds3, "g:")
        plt.plot(data3["t"], -pos_bounds3, "g:")
        plt.ylabel(_label)
    plt.gcf().supxlabel("Time [s]")
    plt.tight_layout()

    # Euler angles
    plt.figure()
    ang_bound1 = [45, 150, 45, 180]
    ang_bound2 = [90, 300, 90, 300]
    angles1 = np.vstack([quat2angle(data1["x"]["quat"][j, :, 0]) for j in range(len(data1["x"]["quat"][:, 0, 0]))])
    angles2 = np.vstack([quat2angle(data2["x"]["quat"][j, :, 0]) for j in range(len(data2["x"]["quat"][:, 0, 0]))])
    angles3 = np.vstack([quat2angle(data3["x"]["quat"][j, :, 0]) for j in range(len(data3["x"]["quat"][:, 0, 0]))])

    ax = plt.subplot(311)
    for i, _label in enumerate([r"$\phi [deg]$", r"$\theta [deg]$", r"$\psi [deg]$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], np.rad2deg(angles1[:, 2-i]), "k--", label="BLF-1")
        plt.plot(data2["t"], np.rad2deg(angles2[:, 2-i]), "b-", label="BLF-2")
        plt.plot(data3["t"], np.rad2deg(angles3[:, 2-i]), "g.-", label="BLF-3")
        if i == 2:
            plt.plot(data1["t"], np.ones((np.size(data1["t"])))*ang_bound1[2], "g:")
            plt.plot(data1["t"], -np.ones((np.size(data1["t"])))*ang_bound1[2], "g:")
            plt.plot(data2["t"], np.ones((np.size(data1["t"])))*ang_bound2[2], "b:")
            plt.plot(data2["t"], -np.ones((np.size(data1["t"])))*ang_bound2[2], "b:")
        else:
            plt.plot(data1["t"], np.ones((np.size(data1["t"])))*ang_bound1[0], "g:")
            plt.plot(data1["t"], -np.ones((np.size(data1["t"])))*ang_bound1[0], "g:")
            plt.plot(data2["t"], np.ones((np.size(data1["t"])))*ang_bound2[0], "b:")
            plt.plot(data2["t"], -np.ones((np.size(data1["t"])))*ang_bound2[0], "b:")
        plt.ylabel(_label)
    plt.gcf().supxlabel("Time [s]")
    plt.tight_layout()

    # Angular rate
    plt.figure()
    ax = plt.subplot(311)
    for i, _label in enumerate([r"$p [deg/s]$", r"$q [deg/s]$", r"$r [deg/s]$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], np.rad2deg(data1["x"]["omega"][:, i, 0]), "k--", label="BLF-1")
        plt.plot(data2["t"], np.rad2deg(data2["x"]["omega"][:, i, 0]), "b-", label="BLF-2")
        plt.plot(data3["t"], np.rad2deg(data3["x"]["omega"][:, i, 0]), "g.-", label="BLF-3")
        if i == 2:
            plt.plot(data1["t"], np.ones((np.size(data1["t"])))*ang_bound1[3], "g:")
            plt.plot(data1["t"], -np.ones((np.size(data1["t"])))*ang_bound1[3], "g:")
            plt.plot(data2["t"], np.ones((np.size(data1["t"])))*ang_bound2[3], "b:")
            plt.plot(data2["t"], -np.ones((np.size(data1["t"])))*ang_bound2[3], "b:")
        else:
            plt.plot(data1["t"], np.ones((np.size(data1["t"])))*ang_bound1[1], "g:")
            plt.plot(data1["t"], -np.ones((np.size(data1["t"])))*ang_bound1[1], "g:")
            plt.plot(data2["t"], np.ones((np.size(data1["t"])))*ang_bound2[1], "b:")
            plt.plot(data2["t"], -np.ones((np.size(data1["t"])))*ang_bound2[1], "b:")
        plt.ylabel(_label)
    plt.gcf().supxlabel("Time [s]")
    plt.tight_layout()

    plt.show()
