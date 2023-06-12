import numpy as np
import matplotlib.pyplot as plt
import fym
from fym.utils.rot import angle2quat, quat2angle
from ftc.agents.param import get_uncertainties
import ftc.config
from ftc.agents.param import get_sumOfDist
import matplotlib.patches as patches

plt.rcParams.update({'font.size': 13})
cfg = ftc.config.load()

plt.rc("text", usetex=False)
plt.rc("lines", linewidth=1.5)
plt.rc("axes", grid=True, labelsize=15, titlesize=15)
plt.rc("grid", linestyle="--", alpha=0.8)
plt.rc("legend", fontsize=15)

style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=style, color="k")


def exp_plot(loggerpath1, lp2, lp3):
    data1, info = fym.load(loggerpath1, with_info=True)  # base
    data2 = fym.load(lp2)  # attitude
    data3 = fym.load(lp3)  # position error

    # observation: position error
    fig, axes = plt.subplots(nrows=3, figsize=(9, 10), sharex=True)
    rho1 = np.array([1.5, 0.2])
    rho3 = np.array([1.5, 1])
    rho_k = 1
    pos_bounds1 = np.zeros((np.shape(data1["x"]["pos"][:, 0, 0])[0]))  # BLF-1, 2
    pos_bounds3 = np.zeros((np.shape(data1["x"]["pos"][:, 0, 0])[0]))  # BLF-3
    for i in range(np.shape(data1["x"]["pos"][:, 0, 0])[0]):
        pos_bounds1[i] = (rho1[0]-rho1[1]) * np.exp(-rho_k*data1["t"][i]) + rho1[1]
        pos_bounds3[i] = (rho3[0]-rho3[1]) * np.exp(-rho_k*data3["t"][i]) + rho3[1]
    for i, (_label, ax) in enumerate(zip([r"$e_{1x}$", r"$e_{1y}$", r"$e_{1z}$"], axes)):
        ax.plot(data1["t"], pos_bounds1, linestyle=":", color="red", linewidth=1)
        ax.plot(data1["t"], -pos_bounds1, linestyle=":", color="red", linewidth=1)
        ax.plot(data3["t"], pos_bounds3, linestyle=":", color="green", linewidth=1)
        ax.plot(data3["t"], -pos_bounds3, linestyle=":", color="green", linewidth=1)
        ax.plot(data1["t"], data1["x"]["pos"][:, i, 0]-data1["ref"][:, i, 0], "r-", label="BLF-1")
        ax.plot(data2["t"], data2["x"]["pos"][:, i, 0]-data2["ref"][:, i, 0], "b--", label="BLF-2")
        ax.plot(data3["t"], data3["x"]["pos"][:, i, 0]-data3["ref"][:, i, 0], "g-.", label="BLF-3")
        ax.set_ylabel(_label + " [m]")
        ax.set_ylim([-1.6, 1.6])
        ax.set_xlim([-0.5, 20.5])
        if i == 2:
            ax.set_xlabel("Time [s]", labelpad=5)
        if i == 0:
            a1 = patches.FancyArrowPatch((12.5, 1), (13.5, 1.3),
                                         connectionstyle="arc3,rad=-.5", **kw)
            ax.add_patch(a1)
            ax.text(13.7, 1.2, "BLF-3 constraint")
            a2 = patches.FancyArrowPatch((12, 0.2), (13, 0.5),
                                         connectionstyle="arc3,rad=-.5", **kw)
            ax.add_patch(a2)
            ax.text(13.2, 0.4, "BLF-1, BLF-2 constraint")
            ax.legend(loc=[0, 1.03], ncol=3, mode="expand")
    # ax.tight_layout()
    plt.savefig("Case1_poserr.png", dpi=600, bbox_inches='tight')

    # Euler angles
    fig, axes = plt.subplots(nrows=3, figsize=(9, 10), sharex=True)
    ang_bound1 = [45, 150, 45, 180]
    ang_bound2 = [90, 300, 90, 300]
    angles1 = np.vstack([quat2angle(data1["x"]["quat"][j, :, 0]) for j in range(len(data1["x"]["quat"][:, 0, 0]))])
    angles2 = np.vstack([quat2angle(data2["x"]["quat"][j, :, 0]) for j in range(len(data2["x"]["quat"][:, 0, 0]))])
    angles3 = np.vstack([quat2angle(data3["x"]["quat"][j, :, 0]) for j in range(len(data3["x"]["quat"][:, 0, 0]))])

    for i, (_label, ax) in enumerate(zip([r"$\phi$", r"$\theta$", r"$\psi$"], axes)):
        if i == 2:
            ax.plot(data1["t"], np.ones((np.size(data1["t"])))*ang_bound1[2], linestyle=":", color="red", linewidth=1)
            ax.plot(data1["t"], -np.ones((np.size(data1["t"])))*ang_bound1[2], linestyle=":", color="red", linewidth=1)
            ax.plot(data2["t"], np.ones((np.size(data1["t"])))*ang_bound2[2], linestyle=":", color="blue", linewidth=1)
            ax.plot(data2["t"], -np.ones((np.size(data1["t"])))*ang_bound2[2], linestyle=":", color="blue", linewidth=1)
        else:
            ax.plot(data1["t"], np.ones((np.size(data1["t"])))*ang_bound1[0], linestyle=":", color="red", linewidth=1)
            ax.plot(data1["t"], -np.ones((np.size(data1["t"])))*ang_bound1[0], linestyle=":", color="red", linewidth=1)
            ax.plot(data2["t"], np.ones((np.size(data1["t"])))*ang_bound2[0], linestyle=":", color="blue", linewidth=1)
            ax.plot(data2["t"], -np.ones((np.size(data1["t"])))*ang_bound2[0], linestyle=":", color="blue", linewidth=1)
        ax.plot(data1["t"], np.rad2deg(angles1[:, 2-i]), "r-", label="BLF-1")
        ax.plot(data2["t"], np.rad2deg(angles2[:, 2-i]), "b--", label="BLF-2")
        ax.plot(data3["t"], np.rad2deg(angles3[:, 2-i]), "g-.", label="BLF-3")
        ax.set_ylabel(_label + " [deg]")
        ax.set_xlim([-0.5, 20.5])
        ax.set_ylim([-110, 110])
        if i == 2:
            ax.set_xlabel("Time [s]", labelpad=5)
        if i == 0:
            a1 = patches.FancyArrowPatch((12, 90), (13, 65),
                                         connectionstyle="arc3,rad=.5", **kw)
            ax.add_patch(a1)
            ax.text(13.2, 61, "BLF-2 constraint")
            a2 = patches.FancyArrowPatch((2, 45), (3, 70),
                                         connectionstyle="arc3,rad=-.5", **kw)
            ax.add_patch(a2)
            ax.text(3.2, 68, "BLF-1, BLF-3 constraint")
            ax.legend(loc=[0, 1.03], ncol=3, mode="expand")

    # Angular rate
    fig, axes = plt.subplots(nrows=3, figsize=(9, 10), sharex=True)
    for i, (_label, ax) in enumerate(zip([r"$p$ [deg/s]", r"$q$ [deg/s]", r"$r$ [deg/s]"], axes)):
        if i == 2:
            ax.plot(data1["t"], np.ones((np.size(data1["t"])))*ang_bound1[3], linestyle=":", color="red", linewidth=1)
            ax.plot(data1["t"], -np.ones((np.size(data1["t"])))*ang_bound1[3], linestyle=":", color="red", linewidth=1)
            ax.plot(data2["t"], np.ones((np.size(data1["t"])))*ang_bound2[3], linestyle=":", color="blue", linewidth=1)
            ax.plot(data2["t"], -np.ones((np.size(data1["t"])))*ang_bound2[3], linestyle=":", color="blue", linewidth=1)
        else:
            ax.plot(data1["t"], np.ones((np.size(data1["t"])))*ang_bound1[1], linestyle=":", color="red", linewidth=1)
            ax.plot(data1["t"], -np.ones((np.size(data1["t"])))*ang_bound1[1], linestyle=":", color="red", linewidth=1)
            ax.plot(data2["t"], np.ones((np.size(data1["t"])))*ang_bound2[1], linestyle=":", color="blue", linewidth=1)
            ax.plot(data2["t"], -np.ones((np.size(data1["t"])))*ang_bound2[1], linestyle=":", color="blue", linewidth=1)
        ax.plot(data1["t"], np.rad2deg(data1["x"]["omega"][:, i, 0]), "r-", label="BLF-1")
        ax.plot(data2["t"], np.rad2deg(data2["x"]["omega"][:, i, 0]), "b--", label="BLF-2")
        ax.plot(data3["t"], np.rad2deg(data3["x"]["omega"][:, i, 0]), "g-.", label="BLF-3")
        ax.set_ylabel(_label + "[deg/s]")
        ax.set_ylim([-340, 340])
        ax.set_xlim([-0.5, 20.5])
        if i == 2:
            ax.set_xlabel("Time [s]", labelpad=5)
        if i == 0:
            a1 = patches.FancyArrowPatch((12, 300), (13, 230),
                                         connectionstyle="arc3,rad=.5", **kw)
            ax.add_patch(a1)
            ax.text(13.2, 220, "BLF-2 constraint")
            a2 = patches.FancyArrowPatch((2, 150), (3, 200),
                                         connectionstyle="arc3,rad=-.5", **kw)
            ax.add_patch(a2)
            ax.text(3.2, 190, "BLF-1, BLF-3 constraint")
            ax.legend(loc=[0, 1.03], ncol=3, mode="expand")

    plt.show()
