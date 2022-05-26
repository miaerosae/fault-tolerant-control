import numpy as np
import matplotlib.pyplot as plt
import fym
from fym.utils.rot import angle2quat, quat2angle


def exp_plot(loggerpath1, loggerpath2):
    data1, info1 = fym.load(loggerpath1, with_info=True)
    data2, info2 = fym.load(loggerpath2, with_info=True)
    # detection_time = info1["detection_time"]
    rotor_min = info1["rotor_min"]
    rotor_max = info1["rotor_max"]

    # FDI
    plt.figure()

    ax = plt.subplot(321)
    for i in range(data1["W"].shape[1]):
        if i != 0:
            plt.subplot(321+i, sharex=ax)
        plt.ylim([0-0.1, 1+0.1])
        plt.plot(data1["t"], data1["W"][:, i, i], "r--", label="Actual")
        plt.plot(data1["t"], data1["What"][:, i, i], "k-", label="Estimated")
        if i == 0:
            plt.legend()
    plt.gcf().supylabel("FDI")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

    # Rotor
    plt.figure()

    ax = plt.subplot(321)
    for i in range(data1["rotors"].shape[1]):
        if i != 0:
            plt.subplot(321+i, sharex=ax)
        plt.ylim([rotor_min-5, rotor_max+5])
        plt.plot(data1["t"], data1["rotors"][:, i], "k-", label="Response")
        plt.plot(data2["t"], data2["rotors"][:, i], "b-", label="Response")
        # plt.plot(data1["t"], data1["rotors_cmd"][:, i], "r--", label="Command")
        if i == 0:
            plt.legend()
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Rotor thrust")
    plt.tight_layout()
    # plt.savefig("lpeso_rotor_input.png")

    # Position
    plt.figure()
    # plt.ylim([-5, 5])

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["x", "y", "z"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        # plt.plot(data1["t"], data1["obs"][:, i, 0], "b", label=_label+" (observation)")
        plt.plot(data1["t"], data1["x"]["pos"][:, i, 0], "k-.", label=_label)
        plt.plot(data2["t"], data2["x"]["pos"][:, i, 0], "b-.", label=_label)
        plt.plot(data1["t"], data1["ref"][:, i, 0], "r--", label=_label+" (cmd)")
        plt.legend(loc="upper right")
    # plt.axvspan(3, detection_time[0], alpha=0.2, color="b")
    # plt.axvline(detection_time[0], alpha=0.8, color="b", linewidth=0.5)
    # plt.annotate("Rotor 0 fails", xy=(3, 0), xytext=(3.5, 0.5),
    #              arrowprops=dict(arrowstyle='->', lw=1.5))
    # plt.axvspan(6, detection_time[1], alpha=0.2, color="b")
    # plt.axvline(detection_time[1], alpha=0.8, color="b", linewidth=0.5)
    # plt.annotate("Rotor 2 fails", xy=(6, 0), xytext=(7.5, 0.2),
    #              arrowprops=dict(arrowstyle='->', lw=1.5))
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Position, m")
    plt.tight_layout()
    plt.legend()
    # plt.savefig("lpeso_pos.png", dpi=300)

    # velocity
    plt.figure()
    plt.ylim([-5, 5])

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["Vx", "Vy", "Vz"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], data1["x"]["vel"][:, i, 0], "k"+_ls, label=_label)
        plt.plot(data2["t"], data2["x"]["vel"][:, i, 0], "b"+_ls, label=_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Velocity, m/s")
    plt.tight_layout()
    plt.legend()

    # euler angles
    plt.figure()
    plt.ylim([-40, 40])

    ax = plt.subplot(311)
    angles1 = np.vstack([quat2angle(data1["x"]["quat"][j, :, 0]) for j in range(len(data1["x"]["quat"][:, 0, 0]))])
    angles2 = np.vstack([quat2angle(data2["x"]["quat"][j, :, 0]) for j in range(len(data2["x"]["quat"][:, 0, 0]))])
    ax = plt.subplot(311)
    for i, _label in enumerate([r"$\phi$", r"$\theta$", r"$\psi$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        # if i == 2:
            # plt.plot(data1["t"], np.rad2deg(data1["obs"][:, 3, 0]), "b", label=r"$\psi$"+" (observation)")
        plt.plot(data1["t"], np.rad2deg(angles1[:, 2-i]), "k"+_ls, label=_label)
        plt.plot(data2["t"], np.rad2deg(angles2[:, 2-i]), "b"+_ls, label=_label)
        plt.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Euler angles, deg")
    plt.tight_layout()
    plt.legend()
    # plt.savefig("lpeso_angle.png", dpi=300)

    # angular rates
    plt.figure()
    plt.ylim([-90, 90])

    for i, (_label, _ls) in enumerate(zip(["p", "q", "r"], ["-.", "--", "-"])):
        plt.plot(data1["t"], np.rad2deg(data1["x"]["omega"][:, i, 0]), "k"+_ls, label=_label)
        plt.plot(data2["t"], np.rad2deg(data2["x"]["omega"][:, i, 0]), "b"+_ls, label=_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Angular rates, deg/s")
    plt.tight_layout()
    plt.legend()
    # plt.savefig("Figure_3.png")

    # observation
    # plt.figure()

    # ax = plt.subplot(411)
    # for i in range(data1["observation"].shape[1]):
    #     if i != 0:
    #         plt.subplot(411+i, sharex=ax)
    #     # plt.xlim([0, 5])
    #     # plt.ylim([0, 10])
    #     plt.plot(data1["t"], data1["observation"][:, i, 0], "r--", label="observation")
    #     if i == 0:
    #         plt.legend()
    # plt.gcf().supylabel("observation")
    # plt.gcf().supxlabel("Time, sec")
    # plt.tight_layout()

    # virtual control
    plt.figure()

    ax = plt.subplot(411)
    for i, _label in enumerate([r"$F$", r"$M_{\phi}$", r"$M_{\theta}$", r"$M_{\psi}$"]):
        if i != 0:
            plt.subplot(411+i, sharex=ax)
        plt.plot(data1["t"], data1["virtual_u"][:, i], "k-", label=_label)
        plt.plot(data2["t"], data2["virtual_u"][:, i], "b-", label=_label)
        plt.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Generalized forces")
    plt.tight_layout()
    # plt.savefig("lpeso_forces.png", dpi=300)

    # disturbance
    # plt.figure()

    # ax = plt.subplot(411)
    # for i in range(data1["dist"].shape[1]):
    #     if i != 0:
    #         plt.subplot(411+i, sharex=ax)
    #     plt.plot(data1["t"], data1["dist"][:, i, 0], "r--", label="distarbance")
    #     plt.plot(data1["t"], data1["d"][:, i, 0], "k", label=" realdistarbance")
    #     if i == 0:
    #         plt.legend()
    # plt.gcf().supylabel("dist")
    # plt.gcf().supxlabel("Time, sec")
    # plt.tight_layout()

    # obs control
    plt.figure()

    ax = plt.subplot(411)
    for i in range(data1["obs_u"].shape[1]):
        if i != 0:
            plt.subplot(411+i, sharex=ax)
        plt.plot(data1["t"], data1["obs_u"][:, i, 0], "k--", label="observer control input")
        plt.plot(data2["t"], data2["obs_u"][:, i, 0], "b--", label="observer control input")
        if i == 0:
            plt.legend()
    plt.gcf().supylabel("observer control input")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()
    # plt.savefig("Figure_6.png")

    # fdi
    # plt.figure()

    # ax = plt.subplot(411)
    # for i, _label in enumerate([r"$fa_{1}$", r"$fa_{2}$", r"$fa_{3}$", r"$fa_{4}$"]):
    #     if i != 0:
    #         plt.subplot(411+i, sharex=ax)
    #     plt.plot(data1["t"], data1["tfa"][:, i], "r", label=_label)
    #     plt.plot(data1["t"], data1["fa"][:, i], "k--", label=_label)
    #     plt.legend(loc="upper right")
    # plt.gcf().supxlabel("Time, sec")
    # plt.gcf().supylabel("FDI info1rmation")
    # plt.tight_layout()

    # plt.figure()
    # plt.plot(data1["t"], data1["comp"][:, 0], "r-")
    # plt.plot(data1["t"], data1["comp"][:, 1], "k--")

    plt.show()


if __name__ == "__main__":
    exp_plot("leso_reduced.h5", "lpeso_reduced.h5")
