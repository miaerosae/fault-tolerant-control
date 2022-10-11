import numpy as np
import matplotlib.pyplot as plt
import fym
from fym.utils.rot import angle2quat, quat2angle
from ftc.agents.param import get_uncertainties
import ftc.config
from ftc.agents.param import get_sumOfDist

cfg = ftc.config.load()


def exp_plot(loggerpath, pf):
    data, info = fym.load(loggerpath, with_info=True)
    # detection_time = info["detection_time"]
    rotor_min = info["rotor_min"]
    rotor_max = info["rotor_max"]

    # FDI
    plt.figure()

    ax = plt.subplot(221)
    for i in range(data["W"].shape[1]):
        if i != 0:
            plt.subplot(221+i, sharex=ax)
        plt.ylim([0-0.1, 1+0.1])
        plt.plot(data["t"], data["W"][:, i, i], "r--", label="Actual")
        plt.plot(data["t"], data["What"][:, i, i], "k-", label="Estimated")
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supylabel("FDI")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

    # Rotor
    plt.figure()

    ax = plt.subplot(221)
    for i in range(data["rotors"].shape[1]):
        if i != 0:
            plt.subplot(221+i, sharex=ax)
        plt.ylim([rotor_min-5, np.sqrt(rotor_max)+5])
        plt.plot(data["t"], np.sqrt(data["rotors"][:, i]), "k-", label="Response")
        plt.plot(data["t"], np.sqrt(data["rotors_cmd"][:, i]), "r--", label="Command")
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
        plt.plot(data["t"], data["obs_pos"][:, i, 0]+data["ref"][:, i, 0], "b-", label="Estimated")
        plt.plot(data["t"], data["x"]["pos"][:, i, 0], "k-.", label="Real")
        plt.plot(data["t"], data["ref"][:, i, 0], "r--", label="Desired")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
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
    # plt.savefig("lpeso_pos.png", dpi=300)

    # velocity
    plt.figure()
    plt.ylim([-5, 5])

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["Vx", "Vy", "Vz"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], data["x"]["vel"][:, i, 0], "k"+_ls, label=_label)
        plt.ylabel(_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Velocity, m/s")
    plt.tight_layout()

    # observation: position error
    plt.figure()

    if pf is True:
        rho = cfg.agents.BLF.pf.oL.rho
        rho_k = cfg.agents.BLF.pf.oL.rho_k
    else:
        rho = cfg.agents.BLF.oL.rho
        rho_k = cfg.agents.BLF.oL.rho_k
    pos_bounds = np.zeros((np.shape(data["x"]["pos"][:, 0, 0])[0]))
    for i in range(np.shape(data["x"]["pos"][:, 0, 0])[0]):
        pos_bounds[i] = (rho[0]-rho[1]) * np.exp(-rho_k*data["t"][i]) + rho[1]
    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["ex", "ey", "ez"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], data["obs_pos"][:, i, 0], "b-", label="Estimated")
        plt.plot(data["t"], data["x"]["pos"][:, i, 0]-data["ref"][:, i, 0], "k-.", label="Real")
        plt.plot(data["t"], pos_bounds, "c")
        plt.plot(data["t"], -pos_bounds, "c")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Error observation, m/s")
    plt.tight_layout()

    # euler angles
    plt.figure()
    if pf is True:
        bound = cfg.agents.BLF.pf.iL.rho[0]
        bound_psi = cfg.agents.BLF.pf.iL.rho_psi[0]
    else:
        bound = cfg.agents.BLF.iL.rho[0]
        bound_psi = cfg.agents.BLF.iL.rho_psi[0]
    plt.ylim(np.rad2deg([-bound, bound])+[-5, 5])

    ax = plt.subplot(311)
    angles = np.vstack([quat2angle(data["x"]["quat"][j, :, 0]) for j in range(len(data["x"]["quat"][:, 0, 0]))])
    ax = plt.subplot(311)
    for i, _label in enumerate([r"$\phi$", r"$\theta$", r"$\psi$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], np.rad2deg(data["obs_ang"][:, i, 0]), "b-", label="Estimated")
        plt.plot(data["t"], np.rad2deg(angles[:, 2-i]), "k-.", label="Real")
        plt.plot(data["t"], np.rad2deg(data["eulerd"][:, i, 0]), "r--", label="Desired")
        if i != 2:
            plt.plot(data["t"],
                     np.ones((np.size(data["t"])))*np.rad2deg(bound), "c")
            plt.plot(data["t"],
                     -np.ones((np.size(data["t"])))*np.rad2deg(bound), "c")
        else:
            plt.plot(data["t"],
                     np.ones((np.size(data["t"])))*np.rad2deg(bound_psi), "c")
            plt.plot(data["t"],
                     -np.ones((np.size(data["t"])))*np.rad2deg(bound_psi), "c")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Euler angles, deg")
    plt.tight_layout()
    # plt.savefig("lpeso_angle.png", dpi=300)

    # angular rates
    plt.figure()
    if pf is True:
        bound = cfg.agents.BLF.pf.iL.rho[1]
        bound_psi = cfg.agents.BLF.pf.iL.rho_psi[1]
    else:
        bound = cfg.agents.BLF.iL.rho[1]
        bound_psi = cfg.agents.BLF.iL.rho_psi[1]
    plt.ylim(np.rad2deg([-bound, bound])+[-5, 5])

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["p", "q", "r"], ["-.", "--", "-"])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], np.rad2deg(data["x"]["omega"][:, i, 0]), "k"+_ls, label=_label)
        if i != 2:
            plt.plot(data["t"],
                     np.ones((np.size(data["t"])))*np.rad2deg(bound), "c")
            plt.plot(data["t"],
                     -np.ones((np.size(data["t"])))*np.rad2deg(bound), "c")
        else:
            plt.plot(data["t"],
                     np.ones((np.size(data["t"])))*np.rad2deg(bound_psi), "c")
            plt.plot(data["t"],
                     -np.ones((np.size(data["t"])))*np.rad2deg(bound_psi), "c")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Angular rates, deg/s")
    plt.tight_layout()
    plt.legend(loc='upper right')
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
    for i, _label in enumerate([r"$F$", r"$M_{\phi}$", r"$M_{\theta}$", r"$M_{\psi}$"]):
        if i != 0:
            plt.subplot(411+i, sharex=ax)
        plt.plot(data["t"], data["virtual_u"][:, i], "k-", label=_label)
        plt.ylabel(_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Generalized forces")
    plt.tight_layout()
    # plt.savefig("lpeso_forces.png", dpi=300)

    # disturbance
    plt.figure()

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
    for i, _label in enumerate([r"$d_x$", r"$d_y$", r"$d_z$",
                                r"$d_\phi$", r"$d_\theta$", r"$d_\psi$"]):
        if i != 0:
            plt.subplot(611+i, sharex=ax)
        plt.plot(data["t"], data["dist"][:, i, 0] - real_dist[i, :], "r-", label="true")
        # plt.plot(data["t"], data["dist"][:, i, 0], "k", label=" distarbance")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supylabel("dist")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

    # Update parameter
    # plt.figure()

    # ax = plt.subplot(611)
    # for i in range(data["dist"].shape[1]):
    #     if i != 0:
    #         plt.subplot(611+i, sharex=ax)
    #     plt.plot(data["t"], data["theta"][:, i, 0], "k", label="update parameter")
    #     if i == 0:
    #         plt.legend()
    # plt.gcf().supylabel("update parameter")
    # plt.gcf().supxlabel("Time, sec")
    # plt.tight_layout()

    # q
    plt.figure()

    ax = plt.subplot(311)
    for i, _label in enumerate([r"$q_x$", r"$q_y$", r"$q_z$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], data["q"][:, i, 0], "k-")
        plt.ylabel(_label)
    plt.gcf().supylabel("observer control input")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()
    # plt.savefig("Figure_6.png")

    # fdi
    plt.figure()

    ax = plt.subplot(331)
    for i in range(9):
        if i != 0:
            plt.subplot(331+i, sharex=ax)
        plt.plot(data["t"], data["gain"][:, i, 0], "r", label=_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("position Real gain value")
    plt.tight_layout()

    plt.figure()

    ax = plt.subplot(331)
    for i in range(9):
        if i != 0:
            plt.subplot(331+i, sharex=ax)
        plt.plot(data["t"], data["gain"][:, i+9, 0], "r", label=_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("angle Real gain value")
    plt.tight_layout()

    plt.show()
