import numpy as np
import matplotlib.pyplot as plt
import fym
from fym.utils.rot import angle2quat, quat2angle
from ftc.agents.param import get_uncertainties
import ftc.config
from ftc.agents.param import get_sumOfDist

cfg = ftc.config.load()


def exp_plot(loggerpath1, loggerpath2):
    data1, info = fym.load(loggerpath1, with_info=True)
    data2 = fym.load(loggerpath2)
    # data3 = fym.load(loggerpath3)
    # data4 = fym.load(loggerpath4)
    # detection_time = info["detection_time"]
    rotor_min = info["rotor_min"]
    rotor_max = info["rotor_max"]

    # Rotor
    plt.figure()

    ax = plt.subplot(221)
    for i in range(data1["rotors"].shape[1]):
        if i != 0:
            plt.subplot(221+i, sharex=ax)
        plt.ylim([rotor_min-5, np.sqrt(rotor_max)+5])
        plt.plot(data1["t"], np.sqrt(data1["rotors"][:, i]), "k-", label="Response")
        plt.plot(data1["t"], np.sqrt(data2["rotors"][:, i]), "b--", label="Response")
        # plt.plot(data1["t"], np.sqrt(data3["rotors"][:, i]), "g--", label="Response")
        # plt.plot(data1["t"], np.sqrt(data4["rotors"][:, i]), "m--", label="Response")
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Angular rate of each rotor")
    plt.tight_layout()
    # plt.savefig("lpeso_rotor_input.png")

    # Position
    plt.figure()
    # plt.ylim([-5, 5])

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["x", "y", "z"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], data1["x"]["pos"][:, i, 0], "k-", label="Real")
        plt.plot(data1["t"], data2["x"]["pos"][:, i, 0], "b--", label="Real")
        plt.plot(data1["t"], data1["obs_pos"][:, i, 0]+data1["ref"][:, i, 0], "g--", label="Real")
        plt.plot(data1["t"], data2["obs_pos"][:, i, 0]+data1["ref"][:, i, 0], "m--", label="Real")
        plt.plot(data1["t"], data1["ref"][:, i, 0], "r-.", label="Desired")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Position, m")
    plt.tight_layout()

    # velocity
    plt.figure()
    plt.ylim([-5, 5])

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["Vx", "Vy", "Vz"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], data1["x"]["vel"][:, i, 0], "k-", label=_label)
        plt.plot(data1["t"], data2["x"]["vel"][:, i, 0], "b--", label=_label)
        # plt.plot(data1["t"], data3["x"]["vel"][:, i, 0], "g--", label=_label)
        # plt.plot(data1["t"], data4["x"]["vel"][:, i, 0], "m--", label=_label)
        plt.ylabel(_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Velocity, m/s")
    plt.tight_layout()

    # observation: position error
    plt.figure()

    rho = cfg.agents.BLF.oL.rho
    rho_k = cfg.agents.BLF.oL.rho_k
    pos_bounds = np.zeros((np.shape(data1["x"]["pos"][:, 0, 0])[0]))
    for i in range(np.shape(data1["x"]["pos"][:, 0, 0])[0]):
        pos_bounds[i] = (rho[0]-rho[1]) * np.exp(-rho_k*data1["t"][i]) + rho[1]
    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["ex", "ey", "ez"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], data1["obs_pos"][:, i, 0], "k-", label="Estimated")
        plt.plot(data1["t"], data2["obs_pos"][:, i, 0], "b--", label="Estimated")
        # plt.plot(data1["t"], data3["obs_pos"][:, i, 0], "g--", label="Estimated")
        # plt.plot(data1["t"], data4["obs_pos"][:, i, 0], "m--", label="Estimated")
        plt.plot(data1["t"], pos_bounds, "c")
        plt.plot(data1["t"], -pos_bounds, "c")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Error observation, m/s")
    plt.tight_layout()

    # euler angles
    plt.figure()
    bound = cfg.agents.BLF.iL.rho[0]
    plt.ylim(np.rad2deg([-bound, bound])+[-5, 5])

    ax = plt.subplot(311)
    angles1 = np.vstack([quat2angle(data1["x"]["quat"][j, :, 0]) for j in range(len(data1["x"]["quat"][:, 0, 0]))])
    angles2 = np.vstack([quat2angle(data2["x"]["quat"][j, :, 0]) for j in range(len(data2["x"]["quat"][:, 0, 0]))])
    # angles3 = np.vstack([quat2angle(data3["x"]["quat"][j, :, 0]) for j in range(len(data2["x"]["quat"][:, 0, 0]))])
    for i, _label in enumerate([r"$\phi$", r"$\theta$", r"$\psi$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], np.rad2deg(data1["obs_ang"][:, i, 0]), "k-", label="Estimated")
        # plt.plot(data1["t"], np.rad2deg(data1["eulerd"][:, i, 0]), "r-", label="Desired")
        # plt.plot(data1["t"], np.rad2deg(angles1[:, 2-i]), "k-.", label="Real")
        plt.plot(data1["t"], np.rad2deg(data2["obs_ang"][:, i, 0]), "b--", label="Estimated")
        # plt.plot(data1["t"], np.rad2deg(data2["eulerd"][:, i, 0]), "r--", label="Desired")
        # plt.plot(data2["t"], np.rad2deg(angles2[:, 2-i]), "b-.", label="Real")
        # plt.plot(data1["t"], np.rad2deg(data3["obs_ang"][:, i, 0]), "g-", label="Estimated")
        # plt.plot(data1["t"], np.rad2deg(data3["eulerd"][:, i, 0]), "g-", label="Desired")
        # plt.plot(data1["t"], np.rad2deg(angles3[:, 2-i]), "g-.", label="Real")
        # plt.plot(data1["t"], np.rad2deg(data4["obs_ang"][:, i, 0]), "m-", label="Estimated")
        plt.plot(data1["t"],
                 np.ones((np.size(data1["t"])))*np.rad2deg(cfg.agents.BLF.iL.rho[0]), "c")
        plt.plot(data1["t"],
                 -np.ones((np.size(data1["t"])))*np.rad2deg(cfg.agents.BLF.iL.rho[0]), "c")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Euler angles, deg")
    plt.tight_layout()
    # plt.savefig("lpeso_angle.png", dpi=300)

    # angular rates
    plt.figure()
    bound = cfg.agents.BLF.iL.rho[1]
    plt.ylim(np.rad2deg([-bound, bound])+[-5, 5])

    for i, (_label, _ls) in enumerate(zip(["p", "q", "r"], ["-.", "--", "-"])):
        plt.plot(data1["t"], np.rad2deg(data1["x"]["omega"][:, i, 0]), "k"+_ls, label=_label)
        plt.plot(data1["t"], np.rad2deg(data2["x"]["omega"][:, i, 0]), "b"+_ls, label=_label)
        # plt.plot(data1["t"], np.rad2deg(data3["x"]["omega"][:, i, 0]), "g"+_ls, label=_label)
        # plt.plot(data1["t"], np.rad2deg(data4["x"]["omega"][:, i, 0]), "m"+_ls, label=_label)
    plt.plot(data1["t"],
             np.ones((np.size(data1["t"])))*np.rad2deg(cfg.agents.BLF.iL.rho[1]), "c")
    plt.plot(data1["t"],
             -np.ones((np.size(data1["t"])))*np.rad2deg(cfg.agents.BLF.iL.rho[1]), "c")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Angular rates, deg/s")
    plt.tight_layout()
    plt.legend(loc='upper right')

    # virtual control
    plt.figure()

    ax = plt.subplot(411)
    for i, _label in enumerate([r"$F$", r"$M_{\phi}$", r"$M_{\theta}$", r"$M_{\psi}$"]):
        if i != 0:
            plt.subplot(411+i, sharex=ax)
        plt.plot(data1["t"], data1["virtual_u"][:, i], "k-", label=_label)
        plt.plot(data1["t"], data2["virtual_u"][:, i], "b--", label=_label)
        # plt.plot(data1["t"], data3["virtual_u"][:, i], "g--", label=_label)
        # plt.plot(data1["t"], data4["virtual_u"][:, i], "m--", label=_label)
        plt.ylabel(_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Generalized forces")
    plt.tight_layout()

    # disturbance
    plt.figure()

    real_dist = np.zeros((6, np.size(data1["t"])))
    ext_dist = cfg.simul_condi.ext_unc
    for i in range(np.size(data1["t"])):
        t = data1["t"][i]
        real_dist[:, i] = get_sumOfDist(t, ext_dist).ravel()

    ax = plt.subplot(611)
    for i, _label in enumerate([r"$d_x$", r"$d_y$", r"$d_z$",
                                r"$d_\phi$", r"$d_\theta$", r"$d_\psi$"]):
        if i != 0:
            plt.subplot(611+i, sharex=ax)
        plt.plot(data1["t"], real_dist[i, :], "r-", label="true")
        plt.plot(data1["t"], data1["dist"][:, i, 0], "k--", label=" distarbance")
        plt.plot(data1["t"], data2["dist"][:, i, 0], "b--", label=" distarbance")
        # plt.plot(data1["t"], data3["dist"][:, i, 0], "g--", label=" distarbance")
        # plt.plot(data1["t"], data4["dist"][:, i, 0], "m--", label=" distarbance")
        plt.ylabel(_label)
    plt.gcf().supylabel("dist")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

    # q
    plt.figure()

    ax = plt.subplot(311)
    for i, _label in enumerate([r"$q_x$", r"$q_y$", r"$q_z$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], data1["q"][:, i, 0], "k-")
        plt.plot(data1["t"], data2["q"][:, i, 0], "b--")
        # plt.plot(data1["t"], data3["q"][:, i, 0], "g--")
        # plt.plot(data1["t"], data4["q"][:, i, 0], "m--")
        plt.ylabel(_label)
    plt.gcf().supylabel("observer control input")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

    plt.show()


def exp_plot4(loggerpath1, loggerpath2, loggerpath3, loggerpath4):
    data1, info = fym.load(loggerpath1, with_info=True)
    data2 = fym.load(loggerpath2)
    data3 = fym.load(loggerpath3)
    data4 = fym.load(loggerpath4)
    # detection_time = info["detection_time"]
    rotor_min = info["rotor_min"]
    rotor_max = info["rotor_max"]

    # Rotor
    plt.figure()

    ax = plt.subplot(221)
    for i in range(data1["rotors"].shape[1]):
        if i != 0:
            plt.subplot(221+i, sharex=ax)
        plt.ylim([rotor_min-5, np.sqrt(rotor_max)+5])
        plt.plot(data1["t"], np.sqrt(data1["rotors"][:, i]), "k-", label="Response")
        plt.plot(data1["t"], np.sqrt(data2["rotors"][:, i]), "b--", label="Response")
        plt.plot(data1["t"], np.sqrt(data3["rotors"][:, i]), "g--", label="Response")
        plt.plot(data1["t"], np.sqrt(data4["rotors"][:, i]), "m--", label="Response")
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Angular rate of each rotor")
    plt.tight_layout()
    # plt.savefig("lpeso_rotor_input.png")

    # Position
    plt.figure()
    # plt.ylim([-5, 5])

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["x", "y", "z"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], data1["x"]["pos"][:, i, 0], "k-", label="Real")
        plt.plot(data1["t"], data2["x"]["pos"][:, i, 0], "b--", label="Real")
        plt.plot(data1["t"], data3["x"]["pos"][:, i, 0], "g--", label="Real")
        plt.plot(data1["t"], data4["x"]["pos"][:, i, 0], "m--", label="Real")
        plt.plot(data1["t"], data1["ref"][:, i, 0], "r-.", label="Desired")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Position, m")
    plt.tight_layout()

    # velocity
    plt.figure()
    plt.ylim([-5, 5])

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["Vx", "Vy", "Vz"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], data1["x"]["vel"][:, i, 0], "k-", label=_label)
        plt.plot(data1["t"], data2["x"]["vel"][:, i, 0], "b--", label=_label)
        plt.plot(data1["t"], data3["x"]["vel"][:, i, 0], "g--", label=_label)
        plt.plot(data1["t"], data4["x"]["vel"][:, i, 0], "m--", label=_label)
        plt.ylabel(_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Velocity, m/s")
    plt.tight_layout()

    # observation: position error
    plt.figure()

    rho = cfg.agents.BLF.oL.rho
    rho_k = cfg.agents.BLF.oL.rho_k
    pos_bounds = np.zeros((np.shape(data1["x"]["pos"][:, 0, 0])[0]))
    for i in range(np.shape(data1["x"]["pos"][:, 0, 0])[0]):
        pos_bounds[i] = (rho[0]-rho[1]) * np.exp(-rho_k*data1["t"][i]) + rho[1]
    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["ex", "ey", "ez"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], data1["obs_pos"][:, i, 0], "k-", label="Estimated")
        plt.plot(data1["t"], data2["obs_pos"][:, i, 0], "b--", label="Estimated")
        plt.plot(data1["t"], data3["obs_pos"][:, i, 0], "g--", label="Estimated")
        plt.plot(data1["t"], data4["obs_pos"][:, i, 0], "m--", label="Estimated")
        plt.plot(data1["t"], pos_bounds, "c")
        plt.plot(data1["t"], -pos_bounds, "c")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Error observation, m/s")
    plt.tight_layout()

    # euler angles
    plt.figure()
    bound = cfg.agents.BLF.iL.rho[0]
    plt.ylim(np.rad2deg([-bound, bound])+[-5, 5])

    ax = plt.subplot(311)
    # angles1 = np.vstack([quat2angle(data1["x"]["quat"][j, :, 0]) for j in range(len(data1["x"]["quat"][:, 0, 0]))])
    # angles2 = np.vstack([quat2angle(data2["x"]["quat"][j, :, 0]) for j in range(len(data2["x"]["quat"][:, 0, 0]))])
    # angles3 = np.vstack([quat2angle(data3["x"]["quat"][j, :, 0]) for j in range(len(data2["x"]["quat"][:, 0, 0]))])
    for i, _label in enumerate([r"$\phi$", r"$\theta$", r"$\psi$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], np.rad2deg(data1["obs_ang"][:, i, 0]), "k-", label="Estimated")
        # plt.plot(data1["t"], np.rad2deg(data1["eulerd"][:, i, 0]), "r-", label="Desired")
        # plt.plot(data1["t"], np.rad2deg(angles1[:, 2-i]), "k-.", label="Real")
        plt.plot(data1["t"], np.rad2deg(data2["obs_ang"][:, i, 0]), "b--", label="Estimated")
        # plt.plot(data1["t"], np.rad2deg(data2["eulerd"][:, i, 0]), "r--", label="Desired")
        # plt.plot(data2["t"], np.rad2deg(angles2[:, 2-i]), "b-.", label="Real")
        # plt.plot(data1["t"], np.rad2deg(data3["obs_ang"][:, i, 0]), "g-", label="Estimated")
        plt.plot(data1["t"], np.rad2deg(data3["eulerd"][:, i, 0]), "g-", label="Desired")
        # plt.plot(data1["t"], np.rad2deg(angles3[:, 2-i]), "g-.", label="Real")
        plt.plot(data1["t"], np.rad2deg(data4["obs_ang"][:, i, 0]), "m-", label="Estimated")
        plt.plot(data1["t"],
                 np.ones((np.size(data1["t"])))*np.rad2deg(cfg.agents.BLF.iL.rho[0]), "c")
        plt.plot(data1["t"],
                 -np.ones((np.size(data1["t"])))*np.rad2deg(cfg.agents.BLF.iL.rho[0]), "c")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Euler angles, deg")
    plt.tight_layout()
    # plt.savefig("lpeso_angle.png", dpi=300)

    # angular rates
    plt.figure()
    bound = cfg.agents.BLF.iL.rho[1]
    plt.ylim(np.rad2deg([-bound, bound])+[-5, 5])

    for i, (_label, _ls) in enumerate(zip(["p", "q", "r"], ["-.", "--", "-"])):
        plt.plot(data1["t"], np.rad2deg(data1["x"]["omega"][:, i, 0]), "k"+_ls, label=_label)
        plt.plot(data1["t"], np.rad2deg(data2["x"]["omega"][:, i, 0]), "b"+_ls, label=_label)
        plt.plot(data1["t"], np.rad2deg(data3["x"]["omega"][:, i, 0]), "g"+_ls, label=_label)
        plt.plot(data1["t"], np.rad2deg(data4["x"]["omega"][:, i, 0]), "m"+_ls, label=_label)
    plt.plot(data1["t"],
             np.ones((np.size(data1["t"])))*np.rad2deg(cfg.agents.BLF.iL.rho[1]), "c")
    plt.plot(data1["t"],
             -np.ones((np.size(data1["t"])))*np.rad2deg(cfg.agents.BLF.iL.rho[1]), "c")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Angular rates, deg/s")
    plt.tight_layout()
    plt.legend(loc='upper right')

    # virtual control
    plt.figure()

    ax = plt.subplot(411)
    for i, _label in enumerate([r"$F$", r"$M_{\phi}$", r"$M_{\theta}$", r"$M_{\psi}$"]):
        if i != 0:
            plt.subplot(411+i, sharex=ax)
        plt.plot(data1["t"], data1["virtual_u"][:, i], "k-", label=_label)
        plt.plot(data1["t"], data2["virtual_u"][:, i], "b--", label=_label)
        plt.plot(data1["t"], data3["virtual_u"][:, i], "g--", label=_label)
        plt.plot(data1["t"], data4["virtual_u"][:, i], "m--", label=_label)
        plt.ylabel(_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Generalized forces")
    plt.tight_layout()

    # disturbance
    real_dist1 = np.zeros((6, np.size(data1["t"])))
    ext_dist1 = cfg.simul_condi.ext_unc
    for i in range(np.size(data1["t"])):
        t = data1["t"][i]
        real_dist1[:, i] = get_sumOfDist(t, ext_dist1).ravel()
    for i in range(3):
        real_dist1[i, :] = real_dist1[i, :] + data1["model_uncert_vel"][:, i, 0]
    for i in range(3):
        real_dist1[i+3, :] = real_dist1[i+3, :] + data1["f"][:, i, 0] \
            + data1["model_uncert_omega"][:, i, 0]
    real_dist2 = np.zeros((6, np.size(data2["t"])))
    ext_dist2 = cfg.simul_condi.ext_unc
    for i in range(np.size(data2["t"])):
        t = data2["t"][i]
        real_dist2[:, i] = get_sumOfDist(t, ext_dist2).ravel()
    for i in range(3):
        real_dist2[i, :] = real_dist2[i, :] + data2["model_uncert_vel"][:, i, 0]
    for i in range(3):
        real_dist2[i+3, :] = real_dist2[i+3, :] + data2["f"][:, i, 0] \
            + data2["model_uncert_omega"][:, i, 0]
    real_dist3 = np.zeros((6, np.size(data3["t"])))
    ext_dist3 = cfg.simul_condi.ext_unc
    for i in range(np.size(data3["t"])):
        t = data3["t"][i]
        real_dist3[:, i] = get_sumOfDist(t, ext_dist3).ravel()
    for i in range(3):
        real_dist3[i, :] = real_dist3[i, :] + data3["model_uncert_vel"][:, i, 0]
    for i in range(3):
        real_dist3[i+3, :] = real_dist3[i+3, :] + data3["f"][:, i, 0] \
            + data3["model_uncert_omega"][:, i, 0]
    real_dist4 = np.zeros((6, np.size(data1["t"])))
    ext_dist4 = cfg.simul_condi.ext_unc
    for i in range(np.size(data1["t"])):
        t = data4["t"][i]
        real_dist4[:, i] = get_sumOfDist(t, ext_dist4).ravel()
    for i in range(3):
        real_dist4[i, :] = real_dist4[i, :] + data4["model_uncert_vel"][:, i, 0]
    for i in range(3):
        real_dist4[i+3, :] = real_dist4[i+3, :] + data4["f"][:, i, 0] \
            + data4["model_uncert_omega"][:, i, 0]

    ax = plt.subplot(611)
    for i, _label in enumerate([r"$d_x$", r"$d_y$", r"$d_z$",
                                r"$d_\phi$", r"$d_\theta$", r"$d_\psi$"]):
        plt.figure()
        plt.plot(data1["t"], real_dist1[i, :]-data1["dist"][:, i, 0], "r-", label="true")
        plt.plot(data1["t"], real_dist2[i, :]-data2["dist"][:, i, 0], "r-", label="true")
        plt.plot(data1["t"], real_dist3[i, :]-data3["dist"][:, i, 0], "r-", label="true")
        plt.plot(data1["t"], real_dist4[i, :]-data4["dist"][:, i, 0], "r-", label="true")
        plt.ylabel("dist" + _label)
        plt.xlabel("time")
        plt.tight_layout()
    # plt.gcf().supylabel("dist")
    # plt.gcf().supxlabel("Time, sec")
    # plt.tight_layout()

    # q
    plt.figure()

    ax = plt.subplot(311)
    for i, _label in enumerate([r"$q_x$", r"$q_y$", r"$q_z$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data1["t"], data1["q"][:, i, 0], "k-")
        plt.plot(data1["t"], data2["q"][:, i, 0], "b--")
        plt.plot(data1["t"], data3["q"][:, i, 0], "g--")
        plt.plot(data1["t"], data4["q"][:, i, 0], "m--")
        plt.ylabel(_label)
    plt.gcf().supylabel("observer control input")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

    plt.show()
