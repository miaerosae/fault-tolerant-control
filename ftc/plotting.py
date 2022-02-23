import numpy as np
import matplotlib.pyplot as plt
import fym
from fym.utils.rot import angle2quat, quat2angle


def exp_plot(loggerpath):
    data, info = fym.load(loggerpath, with_info=True)
    # detection_time = info["detection_time"]
    rotor_min = info["rotor_min"]
    rotor_max = info["rotor_max"]

    # FDI
    plt.figure()

    ax = plt.subplot(321)
    for i in range(data["W"].shape[1]):
        if i != 0:
            plt.subplot(321+i, sharex=ax)
        plt.ylim([0-0.1, 1+0.1])
        plt.plot(data["t"], data["W"][:, i, i], "r--", label="Actual")
        plt.plot(data["t"], data["What"][:, i, i], "k-", label="Estimated")
        if i == 0:
            plt.legend()
    plt.gcf().supylabel("FDI")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

    # Rotor
    plt.figure()

    ax = plt.subplot(321)
    for i in range(data["rotors"].shape[1]):
        if i != 0:
            plt.subplot(321+i, sharex=ax)
        plt.ylim([rotor_min-5, rotor_max+5])
        plt.plot(data["t"], data["rotors"][:, i], "k-", label="Response")
        plt.plot(data["t"], data["rotors_cmd"][:, i], "r--", label="Command")
        if i == 0:
            plt.legend()
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Rotor thrust")
    plt.tight_layout()
    # plt.savefig("Figure_4.png")

    # Position
    plt.figure()
    # plt.ylim([-5, 5])

    for i, (_label, _ls) in enumerate(zip(["x", "y", "z"], ["-", "--", "-."])):
        plt.plot(data["t"], data["x"]["pos"][:, i, 0], "k"+_ls, label=_label)
        plt.plot(data["t"], data["obs"][:, i, 0], "b"+_ls, label="observation")
        plt.plot(data["t"], data["ref"][:, i, 0], "r"+_ls, label=_label+" (cmd)")
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
    # plt.savefig("Figure_1.png")

    # velocity
    plt.figure()
    plt.ylim([-5, 5])

    for i, (_label, _ls) in enumerate(zip(["Vx", "Vy", "Vz"], ["-", "--", "-."])):
        plt.plot(data["t"], data["x"]["vel"][:, i, 0], "k"+_ls, label=_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Velocity, m/s")
    plt.tight_layout()
    plt.legend()

    # euler angles
    plt.figure()
    plt.ylim([-40, 40])

    angles = np.vstack([quat2angle(data["x"]["quat"][j, :, 0]) for j in range(len(data["x"]["quat"][:, 0, 0]))])
    for i, (_label, _ls) in enumerate(zip(["yaw", "pitch", "roll"], ["-.", "--", "-"])):
        plt.plot(data["t"], np.rad2deg(angles[:, i]), "k"+_ls, label=_label)
    plt.plot(data["t"], np.rad2deg(data["obs"][:, 3, 0]), "b")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Euler angles, deg")
    plt.tight_layout()
    plt.legend()
    # plt.savefig("Figure_2.png")

    # angular rates
    plt.figure()
    plt.ylim([-90, 90])

    for i, (_label, _ls) in enumerate(zip(["p", "q", "r"], ["-.", "--", "-"])):
        plt.plot(data["t"], np.rad2deg(data["x"]["omega"][:, i, 0]), "k"+_ls, label=_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Angular rates, deg/s")
    plt.tight_layout()
    plt.legend()
    # plt.savefig("Figure_3.png")

    # observation
    # plt.figure()

    # ax = plt.subplot(411)
    # for i in range(data["observation"].shape[1]):
    #     if i != 0:
    #         plt.subplot(411+i, sharex=ax)
    #     # plt.xlim([0, 5])
    #     # plt.ylim([0, 10])
    #     plt.plot(data["t"], data["observation"][:, i, 0], "r--", label="observation")
    #     if i == 0:
    #         plt.legend()
    # plt.gcf().supylabel("observation")
    # plt.gcf().supxlabel("Time, sec")
    # plt.tight_layout()

    # virtual control
    plt.figure()

    ax = plt.subplot(411)
    for i in range(data["virtual_u"].shape[1]):
        if i != 0:
            plt.subplot(411+i, sharex=ax)
        plt.plot(data["t"], data["virtual_u"][:, i, 0], "r--", label="virtual control input")
        if i == 0:
            plt.legend()
    plt.gcf().supylabel("virtual control input")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()
    # plt.savefig("Figure_5.png")

    # disturbance
    # plt.figure()

    # ax = plt.subplot(411)
    # for i in range(data["dist"].shape[1]):
    #     if i != 0:
    #         plt.subplot(411+i, sharex=ax)
    #     plt.plot(data["t"], data["dist"][:, i, 0], "r--", label="distarbance")
    #     plt.plot(data["t"], data["d"][:, i, 0], "k", label=" realdistarbance")
    #     if i == 0:
    #         plt.legend()
    # plt.gcf().supylabel("dist")
    # plt.gcf().supxlabel("Time, sec")
    # plt.tight_layout()

    # obs control
    plt.figure()

    ax = plt.subplot(411)
    for i in range(data["obs_u"].shape[1]):
        if i != 0:
            plt.subplot(411+i, sharex=ax)
        plt.plot(data["t"], data["obs_u"][:, i, 0], "r--", label="observer control input")
        if i == 0:
            plt.legend()
    plt.gcf().supylabel("observer control input")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()
    # plt.savefig("Figure_6.png")

    plt.show()
