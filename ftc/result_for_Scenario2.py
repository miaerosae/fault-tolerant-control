import numpy as np
import matplotlib.pyplot as plt
import fym
from fym.utils.rot import angle2quat, quat2angle
from ftc.agents.param import get_uncertainties
import ftc.config
from ftc.agents.param import get_sumOfDist
import statistics
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


plt.rc("text", usetex=False)
plt.rc("lines", linewidth=1.5)
plt.rc("axes", grid=True, labelsize=15, titlesize=12)
plt.rc("grid", linestyle="--", alpha=0.8)
plt.rc("legend", fontsize=11)

cfg = ftc.config.load()


def exp_plot(path1):
    data, info = fym.load(path1, with_info=True)
    rotor_min = info["rotor_min"]
    rotor_max = info["rotor_max"]

    # FDI
    plt.figure(figsize=(6, 4.5))

    name = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", r"$\lambda_4$"]
    for i in range(data["W"].shape[1]):
        plt.ylim([0-0.1, 1+0.1])
        plt.plot(data["t"], data["W"][:, i, i], "--", label=name[i])
    plt.legend(loc=[0, 1.03], ncol=4, mode="expand")
    plt.xlabel("Time [sec]")
    plt.tight_layout()
    # plt.savefig("lambda.png", dpi=300)

    # 4d) tracking error (subplots)
    plt.figure(figsize=(9, 7))

    rho = np.array([0.5, 0.25])
    rho_k = 0.5
    pos_bounds = np.zeros((np.shape(data["x"]["pos"][:, 0, 0])[0]))
    for i in range(np.shape(data["x"]["pos"][:, 0, 0])[0]):
        pos_bounds[i] = (rho[0]-rho[1]) * np.exp(-rho_k*data["t"][i]) + rho[1]
    ax = plt.subplot(311)
    for i, _label in enumerate([r"$e_x$", r"$e_y$", r"$e_z$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], data["x"]["pos"][:, i, 0]-data["ref"][:, i, 0], "k-", label="Real Value")
        plt.plot(data["t"], data["obs_pos"][:, i, 0], "b--", label="Estimated Value")
        plt.plot(data["t"], pos_bounds, "r:", label="Prescribed Bound")
        plt.plot(data["t"], -pos_bounds, "r:")
        plt.ylabel(_label)
        plt.axvspan(5, 5.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(7, 7.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(10, 10.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(14, 14.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(15, 15.1, alpha=0.2, color="mediumslateblue")
        plt.axvline(5, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(7, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(10, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(14, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(15, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        if i == 0:
            plt.legend(loc=[0, 1.03], ncol=3, mode="expand")
    plt.gcf().supxlabel("Time [sec]")
    plt.tight_layout()

    # 5a) Euler angle trajectories
    plt.figure(figsize=(9, 7))
    bound = 45
    plt.ylim([-bound-5, bound+5])

    angles = np.vstack([quat2angle(data["x"]["quat"][j, :, 0]) for j in range(len(data["x"]["quat"][:, 0, 0]))])
    ax = plt.subplot(311)
    for i, _label in enumerate([r"$\phi$", r"$\theta$", r"$\psi$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], np.rad2deg(angles[:, 2-i]), "k-", label="Real Value")
        plt.plot(data["t"], np.rad2deg(data["obs_ang"][:, i, 0]), "b--", label="Estimated Value")
        plt.plot(data["t"],
                 np.ones((np.size(data["t"])))*bound, "r:", label="Prescribed Bound")
        plt.plot(data["t"], -np.ones((np.size(data["t"])))*bound, "r:")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc=[0, 1.03], ncol=3, mode="expand")
        plt.axvspan(5, 5.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(7, 7.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(10, 10.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(14, 14.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(15, 15.1, alpha=0.2, color="mediumslateblue")
        plt.axvline(5, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(7, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(10, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(14, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(15, alpha=0.8, color="mediumslateblue", linewidth=0.5)
    plt.gcf().supxlabel("Time [sec]")
    plt.tight_layout()
    # plt.savefig("angle.png", dpi=300)

    # 5b) Angular rate trajectories
    plt.figure(figsize=(9, 7))
    bound = 150
    bound_psi = 180

    ax = plt.subplot(311)
    for i, _label in enumerate(["p", "q", "r"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], np.rad2deg(data["x"]["omega"][:, i, 0]), "k-")
        if i == 2:
            plt.plot(data["t"], np.ones((np.size(data["t"])))*bound_psi, "r:",
                     label="Prescribed Bound")
            plt.plot(data["t"], -np.ones((np.size(data["t"])))*bound_psi, "r:")
            plt.ylim([-bound_psi-15, bound_psi+15])
        else:
            plt.plot(data["t"], np.ones((np.size(data["t"])))*bound, "r:",
                     label="Prescribed Bound")
            plt.plot(data["t"], -np.ones((np.size(data["t"])))*bound, "r:")
            plt.ylim([-bound-15, bound+15])
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc="lower right", bbox_to_anchor=[1, 1.03], ncol=1)
        plt.axvspan(5, 5.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(7, 7.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(10, 10.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(14, 14.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(15, 15.1, alpha=0.2, color="mediumslateblue")
        plt.axvline(5, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(7, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(10, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(14, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(15, alpha=0.8, color="mediumslateblue", linewidth=0.5)
    plt.gcf().supxlabel("Time [sec]")
    plt.tight_layout()
    # plt.savefig("angular.png", dpi=300)

    # 6a) rotor input comparison
    plt.figure(figsize=(7, 5))

    name = [r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$", r"$\Omega_4$"]
    ax = plt.subplot(411)
    for i in range(data["rotors"].shape[1]):
        if i != 0:
            plt.subplot(411+i, sharex=ax)
        plt.ylim([rotor_min-5, np.sqrt(rotor_max)+5])
        plt.plot(data["t"], np.sqrt(data["rotors"][:, i]), "k-", label="Response")
        plt.plot(data["t"], np.sqrt(data["rotors_cmd"][:, i]), "r--", label="Command")
        plt.ylabel(name[i])
        if i == 0:
            plt.legend(loc="lower right", bbox_to_anchor=[1, 1.03], ncol=2)
        plt.axvspan(5, 5.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(7, 7.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(10, 10.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(14, 14.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(15, 15.1, alpha=0.2, color="mediumslateblue")
        plt.axvline(5, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(7, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(10, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(14, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(15, alpha=0.8, color="mediumslateblue", linewidth=0.5)
    plt.gcf().supxlabel("Time [sec]")
    plt.tight_layout()
    # plt.savefig("rotor_input.png")

    # 6b) generalized forces comparison
    plt.figure(figsize=(7, 5))

    ax = plt.subplot(411)
    for i, _label in enumerate([r"$u_{1}$", r"$u_{2}$", r"$u_{3}$", r"$u_{4}$"]):
        if i != 0:
            plt.subplot(411+i, sharex=ax)
        plt.plot(data["t"], data["virtual_u"][:, i], "k-", label=_label)
        if i == 0:
            plt.ylabel(_label, labelpad=23)
        elif i == 1:
            plt.ylabel(_label, labelpad=12)
        elif i == 2:
            plt.ylabel(_label, labelpad=8)
        elif i == 3:
            plt.ylabel(_label, labelpad=0)
        plt.axvspan(5, 5.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(7, 7.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(10, 10.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(14, 14.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(15, 15.1, alpha=0.2, color="mediumslateblue")
        plt.axvline(5, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(7, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(10, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(14, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(15, alpha=0.8, color="mediumslateblue", linewidth=0.5)
    plt.gcf().supxlabel("Time [sec]")
    plt.tight_layout()
    # plt.savefig("forces.png", dpi=300)

    # disturbance
    plt.figure(figsize=(12, 9))

    real_dist = np.zeros((6, np.size(data["t"])))
    ext_dist = cfg.simul_condi.ext_unc
    for i in range(np.size(data["t"])):
        t = data["t"][i]
        real_dist[:, i] = get_sumOfDist(t, ext_dist).ravel()
    for i in range(3):
        real_dist[i, :] = (real_dist[i, :]
                           + data["model_uncert_vel"][:, i, 0]
                           + data["int_uncert_vel"][:, i, 0])
    for i in range(3):
        real_dist[i+3, :] = (real_dist[i+3, :]
                             + data["f"][:, i, 0]
                             + data["model_uncert_omega"][:, i, 0])

    ax = plt.subplot(611)
    for i, _label in enumerate([r"$e_{3x}$", r"$e_{3y}$", r"$e_{3z}$",
                                r"$e_{3\phi}$", r"$e_{3\theta}$", r"$e_{3\psi}$"]):
        if i != 0:
            plt.subplot(611+i, sharex=ax)
        plt.plot(data["t"], real_dist[i, :], "k-", label="Real Value")
        plt.plot(data["t"], data["dist"][:, i, 0], "b--", label="Estimated Value")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc="lower right", bbox_to_anchor=[1, 1.03], ncol=2)
        plt.axvspan(5, 5.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(7, 7.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(10, 10.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(14, 14.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(15, 15.1, alpha=0.2, color="mediumslateblue")
        plt.axvline(5, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(7, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(10, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(14, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(15, alpha=0.8, color="mediumslateblue", linewidth=0.5)
    plt.gcf().supxlabel("Time [sec]")
    plt.tight_layout()

    # BLF gain
    plt.figure(figsize=(9, 7))

    # calculate gain of Scenario 2
    kpos = np.array([1, 0.5, 0.5/30/(0.2**2)])
    kang = np.array([400/30, 30, 1/30/np.deg2rad(45)**2])
    rhoinf = 0.25
    kP1 = kpos[0]*kpos[1] + kpos[2]*rhoinf**2 + 1/rhoinf**2
    kD1 = kpos[0] + kpos[1]
    kI1 = kpos[1]*kpos[2]*rhoinf**2
    kP2 = kang[0]*kang[1] + kang[2] + (kang[0] + kang[1])*np.sqrt(0.1) + 0.1
    kD2 = kang[0] + kang[1] + 2*np.sqrt(0.1)
    kI2 = (kang[1] + np.sqrt(0.1))*kang[2]

    ax = plt.subplot(331)
    for i in range(9):
        if i != 0:
            plt.subplot(331+i, sharex=ax)
        if i == 1:
            plt.plot(data["t"], data["gain"][:, i, 0], "r", label="Real Gain")
        else:
            plt.plot(data["t"], data["gain"][:, i, 0], "r")
        if i % 3 == 0:
            plt.plot(data["t"], np.ones(np.shape(data["t"]))*kP1, "b--", label="PID-like Gain")
        elif i % 3 == 1:
            plt.plot(data["t"], np.ones(np.shape(data["t"]))*kD1, "b--")
        elif i % 3 == 2:
            plt.plot(data["t"], np.ones(np.shape(data["t"]))*kI1, "b--", label="PID-like Gain")
        if i == 0:
            plt.ylabel("x subsystem", labelpad=10)
            plt.title(r"$k_{P}$")
        elif i == 1:
            plt.legend(loc="lower right", bbox_to_anchor=[1, 1.13], ncol=1, edgecolor="white")
            plt.title(r"$k_{D}$")
        elif i == 2:
            plt.legend(loc="lower right", bbox_to_anchor=[1, 1.13], ncol=1, edgecolor="white")
            plt.title(r"$k_{I}$")
        elif i == 3:
            plt.ylabel("y subsystem")
        elif i == 6:
            plt.ylabel("z subsystem", labelpad=8)
        plt.axvspan(5, 5.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(7, 7.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(10, 10.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(14, 14.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(15, 15.1, alpha=0.2, color="mediumslateblue")
        plt.axvline(5, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(7, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(10, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(14, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(15, alpha=0.8, color="mediumslateblue", linewidth=0.5)

    plt.gcf().supxlabel("Time [sec]")
    plt.tight_layout()

    plt.figure(figsize=(9, 7))

    ax = plt.subplot(331)
    for i in range(9):
        if i != 0:
            plt.subplot(331+i, sharex=ax)
        if i == 1:
            plt.plot(data["t"], data["gain"][:, i+9, 0], "r", label="Real Gain")
        else:
            plt.plot(data["t"], data["gain"][:, i+9, 0], "r")
        if i % 3 == 0:
            plt.plot(data["t"], np.ones(np.shape(data["t"]))*kP2, "b--", label="PID-like Gain")
        elif i % 3 == 1:
            plt.plot(data["t"], np.ones(np.shape(data["t"]))*kD2, "b--")
        elif i % 3 == 2:
            plt.plot(data["t"], np.ones(np.shape(data["t"]))*kI2, "b--", label="PID-like Gain")
        if i == 0:
            plt.ylabel(r"$\phi$" + " subsystem", labelpad=10)
            plt.title(r"$k_{P}$")
        elif i == 1:
            plt.legend(loc="lower right", bbox_to_anchor=[1, 1.13], ncol=1, edgecolor="white")
            plt.title(r"$k_{D}$")
        elif i == 2:
            plt.legend(loc="lower right", bbox_to_anchor=[1, 1.13], ncol=1, edgecolor="white")
            plt.title(r"$k_{I}$")
        elif i == 3:
            plt.ylabel(r"$\theta$" + " subsystem")
        elif i == 6:
            plt.ylabel(r"$\psi$" + " subsystem", labelpad=9)
        plt.axvspan(5, 5.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(7, 7.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(10, 10.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(14, 14.1, alpha=0.2, color="mediumslateblue")
        plt.axvspan(15, 15.1, alpha=0.2, color="mediumslateblue")
        plt.axvline(5, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(7, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(10, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(14, alpha=0.8, color="mediumslateblue", linewidth=0.5)
        plt.axvline(15, alpha=0.8, color="mediumslateblue", linewidth=0.5)
    plt.gcf().supxlabel("Time [sec]")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    exp_plot("Scenario2.h5")
