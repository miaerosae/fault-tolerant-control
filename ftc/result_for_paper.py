import numpy as np
import matplotlib.pyplot as plt
import fym
from fym.utils.rot import angle2quat, quat2angle
from ftc.agents.param import get_uncertainties
import ftc.config
from ftc.agents.param import get_sumOfDist
import statistics

cfg = ftc.config.load()


def exp_plot(loggerpath, pf):
    data, info = fym.load(loggerpath, with_info=True)
    # detection_time = info["detection_time"]
    rotor_min = info["rotor_min"]
    rotor_max = info["rotor_max"]

    ''' comparing ESOs '''
    ''' 1. disturbance '''
    # 1a) estimation graph
    real_dist = np.zeros((6, np.size(data["t"])))
    ext_dist = cfg.simul_condi.ext_unc
    for i in range(np.size(data["t"])):
        t = data["t"][i]
        real_dist[:, i] = get_sumOfDist(t, ext_dist).ravel()

    dist_error = np.zeros((6, np.size(data["t"])))
    for i in range(6):
        dist_error[i, :] = data["dist"][:, i, 0] - real_dist[i, :]

    plt.figure(figsize=(9, 7))
    ax = plt.subplot(321)
    for i, _label in enumerate([r"$e_{3x}$", r"$e_{3y}$", r"$e_{3z}$",
                                r"$e_{3\phi}$", r"$e_{3\theta}$", r"$e_{3\psi}$"]):
        if i != 0:
            plt.subplot(321+i, sharex=ax)
        plt.plot(data["t"], data["dist"][:, i, 0], "k", label="Estimated value")
        plt.plot(data["t"], real_dist[i, :], "r--", label="Real value")

        # TODO) ESO 비교군 추가

        if i == 0:
            plt.ylabel(_label, labelpad=15)
        elif i == 1:
            plt.ylabel(_label, labelpad=12)
        elif i == 2:
            plt.ylabel(_label, labelpad=20)
        elif i == 3:
            plt.ylabel(_label, labelpad=19)
        elif i == 4:
            plt.ylabel(_label, labelpad=12)
        elif i == 5:
            plt.ylabel(_label, labelpad=0)
        if i == 0:
            plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 2.0))
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()
    # plt.savefig("dist.png", dpi=300)

    # 1b) STD for observation error(ESO를 여러 개 비교한 경우)
    for i, _label in enumerate([r"$e_{3x}$", r"$e_{3y}$", r"$e_{3z}$",
                                r"$e_{3\phi}$", r"$e_{3\theta}$", r"$e_{3\psi}$"]):
        print("proposed controller: " + _label + str(statistics.stdev(dist_error[i, :])))

        # TODO) ESO 비교군 추가

    ''' 2. position '''
    # 2a) estimated/real value of proposed controller (tracking error)
    # ESO 가 여러 개일 경우 estimated term 추가
    plt.figure()

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate([r"$\tilde{e}_{1x}$", r"$\tilde{e}_{1y}$", r"$\tilde{e}_{1z}$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], (data["obs_pos"]-(data["x"]["pos"]-data["ref"]))[:, i, 0], "k-", label="Proposed")

        # TODO: ESO 추가

        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3))
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()
    # plt.savefig("lpeso_pos.png", dpi=300)

    # 2b) STD for observation error (ESO를 여러 개 비교한 경우)
    for i, _label in enumerate([r"$\tilde{e}_{1x}$", r"$\tilde{e}_{1y}$", r"$\tilde{e}_{1z}$"]):
        print("proposed controller" + _label + str(statistics.stdev((data["obs_pos"]-(data["x"]["pos"]-data["ref"]))[:, i, 0])))

        # TODO: ESO 추가

    ''' 3. euler '''
    # 3a) desired/estimated/real value of proposed controller
    # ESO 가 여러 개일 경우 estimated term 추가
    plt.figure(figsize=(9, 7))

    ax = plt.subplot(311)
    angles = np.vstack([quat2angle(data["x"]["quat"][j, :, 0]) for j in range(len(data["x"]["quat"][:, 0, 0]))])
    ax = plt.subplot(311)
    for i, _label in enumerate([r"$\tilde{\bar{r}}_{1\phi}$", r"$\tilde{\bar{r}}_{1\theta}$", r"$\tilde{\bar{r}}_{1\psi}$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], np.rad2deg(data["obs_ang"][:, i, 0]-angles[:, 2-i]), "k-", label="Real")

        # TODO: ESO 추가

        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.3))
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()
    # plt.savefig("angle.png", dpi=300)

    # 3b) STD for observation error (ESO를 여러 개 비교한 경우)
    for i, _label in enumerate([r"$\tilde{\bar{r}}_{1\phi}$", r"$\tilde{\bar{r}}_{1\theta}$", r"$\tilde{\bar{r}}_{1\psi}$"]):
        print("proposed controller: " + _label + str(statistics.stdev(np.rad2deg(data["obs_ang"][:, i, 0]-angles[:, 2-i]))))

        # TODO: ESO 추가

    ''' comparing controllers '''
    ''' 4. position '''
    # 4a) 3D plot(x, y, z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data["x"]["pos"][:, 0, 0], data["x"]["pos"][:, 1, 0], -data["x"]["pos"][:, 2, 0], "k-", label="Proposed")

    # TODO: add compare controller data here

    plt.tight_layout()
    plt.legend(loc=0)
    ax.xaxis._axinfo["grid"]['linestyle'] = "--"
    ax.yaxis._axinfo["grid"]['linestyle'] = "--"
    ax.zaxis._axinfo["grid"]['linestyle'] = "--"
    ax.set_xlabel("x, m/s")
    ax.set_ylabel("y, m/s")
    ax.set_title("z, m/s")
    # plt.savefig('total_asmc_3D.pdf')

    # 4b) xy-plane
    plt.figure()
    ax = plt.subplot(311)
    plt.plot(data["x"]["pos"][:, 0, 0], data["x"]["pos"][:, 1, 0], label="proposed")
    plt.subplot(312)

    # TODO: add compare controller data here

    plt.subplot(313)

    plt.gcf().supxlabel("x, m/s")
    plt.gcf().supylabel("y, m/s")
    plt.tight_layout()

    # 4c) only for z comparison
    plt.figure()
    plt.plot(data["t"], data["x"]["pos"][:, 2, 0], label="proposed")

    # TODO: add compare controller data here

    plt.xlabel("Time, sec")
    plt.tight_layout()

    # 4d) tracking error (subplots)
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
    for i, _label in enumerate([r"$e_x$", r"$e_y$", r"$e_z$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], data["x"]["pos"][:, i, 0]-data["ref"][:, i, 0], "k-.", label="Proposed")

        # TODO: add compare controller data here

        plt.plot(data["t"], pos_bounds, "c")
        plt.plot(data["t"], -pos_bounds, "c")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Position tracking error, m/s")
    plt.tight_layout()

    # 4e) STD of tracking error
    for i, _label in enumerate([r"$e_x$", r"$e_y$", r"$e_z$"]):
        print("proposed controller: " + _label + str(statistics.stdev(data["x"]["pos"][:, i, 0]-data["ref"][:, i, 0])))

        # TODO: add compare controller data here

    # 4f) comparison on x, y, z axis respectively (max, mean)
    for i, _label in enumerate([r"$e_x$", r"$e_y$", r"$e_z$"]):
        print("proposed controller: " + _label + str(max(abs(data["x"]["pos"][:, i, 0]-data["ref"][:, i, 0]))))
        print("proposed controller: " + _label + str(np.mean(abs(data["x"]["pos"][:, i, 0]-data["ref"][:, i, 0]))))

        # TODO: add compare controller data here

    ''' 5. euler '''
    # 5a) Euler angle trajectories
    plt.figure(figsize=(9, 7))
    if pf is True:
        bound = cfg.agents.BLF.pf.iL.rho[0]
    else:
        bound = cfg.agents.BLF.iL.rho[0]
    plt.ylim(np.rad2deg([-bound, bound])+[-5, 5])

    ax = plt.subplot(311)
    angles = np.vstack([quat2angle(data["x"]["quat"][j, :, 0]) for j in range(len(data["x"]["quat"][:, 0, 0]))])
    ax = plt.subplot(311)
    for i, _label in enumerate([r"$\phi$", r"$\theta$", r"$\psi$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], np.rad2deg(angles[:, 2-i]), "k-.", label="Real")

        # TODO: add compare controller data here

        plt.plot(data["t"],
                 np.ones((np.size(data["t"])))*np.rad2deg(cfg.agents.BLF.iL.rho[0]), "c",
                 label="Bound")
        plt.plot(data["t"],
                 -np.ones((np.size(data["t"])))*np.rad2deg(cfg.agents.BLF.iL.rho[0]), "c")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.3))
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()
    # plt.savefig("angle.png", dpi=300)

    # 5b) Angular rate trajectories
    plt.figure(figsize=(9, 7))
    if pf is True:
        bound = cfg.agents.BLF.pf.iL.rho[1]
    else:
        bound = cfg.agents.BLF.iL.rho[1]
    plt.ylim(np.rad2deg([-bound, bound])+[-5, 5])

    for i, (_label, _ls) in enumerate(zip(["p", "q", "r"], ["-.", "--", "-"])):
        plt.plot(data["t"], np.rad2deg(data["x"]["omega"][:, i, 0]), "k"+_ls, label=_label)

        # TODO: add compare controller data here

    plt.plot(data["t"],
             np.ones((np.size(data["t"])))*np.rad2deg(cfg.agents.BLF.iL.rho[1]), "c",
             label="Bound")
    plt.plot(data["t"],
             -np.ones((np.size(data["t"])))*np.rad2deg(cfg.agents.BLF.iL.rho[1]), "c")
    plt.gcf().supxlabel("Time, sec")
    plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.1))
    plt.tight_layout()
    # plt.savefig("angular.png", dpi=300)

    # 5c) tracking error
    plt.figure()
    ax = plt.subplot(311)
    for i, _label in enumerate([r"$e_{\phi}$", r"$e_{\theta}$", r"$e_{\psi}$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], np.rad2deg(angles[:, 2-i])-np.rad2deg(data["eulerd"][:, i, 0]), "k-.")

        # TODO: add compare controller data here

        plt.ylabel(_label)
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()
    # plt.savefig("angle_error.png", dpi=300)

    # 5d) STD of tracking error
    for i, _label in enumerate([r"$e_{\phi}$", r"$e_{\theta}$", r"$e_{\psi}$"]):
        print("proposed controller: " + _label + str(statistics.stdev(np.rad2deg(angles[:, 2-i])-np.rad2deg(data["eulerd"][:, i, 0]))))

        # TODO: add compare controller data here

    # 5e) comparison on phi, theta, psi respectively (max, mean)
    for i, _label in enumerate([r"$e_x$", r"$e_y$", r"$e_z$"]):
        print("proposed controller: " + _label + str(max(abs(np.rad2deg(angles[:, 2-i])-np.rad2deg(data["eulerd"][:, i, 0])))))
        print("proposed controller: " + _label + str(np.mean(abs(np.rad2deg(angles[:, 2-i])-np.rad2deg(data["eulerd"][:, i, 0])))))

        # TODO: add compare controller data here

    ''' 6. etc '''
    # 6a) rotor input comparison
    plt.figure(figsize=(7, 5))

    name = [r"$\omega_1$", r"$\omega_2$", r"$\omega_3$", r"$\omega_4$"]
    ax = plt.subplot(221)
    for i in range(data["rotors"].shape[1]):
        if i != 0:
            plt.subplot(221+i, sharex=ax)
        plt.ylim([rotor_min-5, np.sqrt(rotor_max)+5])
        plt.plot(data["t"], np.sqrt(data["rotors"][:, i]), "k-", label="Proposed")

        # TODO: add compare controller data here

        if i == 1:
            plt.ylabel(name[i])
            plt.legend(loc='upper center', ncol=2, mode="expand", bbox_to_anchor=(0.5, 1.3))
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()
    # plt.savefig("rotor_input.png")

    # 6b) generalized forces comparison
    plt.figure(figsize=(9, 7))

    ax = plt.subplot(411)
    for i, _label in enumerate([r"$u_{1}$", r"$u_{2}$", r"$u_{3}$", r"$u_{4}$"]):
        if i != 0:
            plt.subplot(411+i, sharex=ax)
        plt.plot(data["t"], data["virtual_u"][:, i], "k-", label=_label)

        # TODO: add compare controller data here

        if i == 0:
            plt.ylabel(_label, labelpad=23)
        elif i == 1:
            plt.ylabel(_label, labelpad=5)
        elif i == 2:
            plt.ylabel(_label, labelpad=10)
        elif i == 3:
            plt.ylabel(_label, labelpad=0)
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()
    # plt.savefig("forces.png", dpi=300)

    plt.show()
