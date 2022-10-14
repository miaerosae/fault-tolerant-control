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


def exp_plot(path1, path2, path3):
    data1, info = fym.load(path1, with_info=True)  # BLF
    data2 = fym.load(path2)  # BS - same gain
    data3 = fym.load(path3)  # BS
    # detection_time = info["detection_time"]
    rotor_min = info["rotor_min"]
    rotor_max = info["rotor_max"]

    fault_time = np.array([5, 7])
    dt = 0.01

    # FDI
    plt.figure(figsize=(6, 4.5))

    name = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", r"$\lambda_4$"]
    for i in range(data1["W"].shape[1]):
        plt.ylim([0-0.1, 1+0.1])
        plt.plot(data1["t"], data1["W"][:, i, i], "--", label=name[i])
    plt.legend(loc=[0, 1.03], ncol=4, mode="expand")
    plt.xlabel("Time [sec]")
    plt.tight_layout()
    # plt.savefig("lambda.png", dpi=300)

    ''' comparing controllers '''
    # 4d) tracking error (subplots)
    rho = np.array([1.5, 0.2])
    rho_k = 1
    pos_bounds = np.zeros((np.shape(data1["x"]["pos"][:, 0, 0])[0]))
    pos_err1 = data1["x"]["pos"]-data1["ref"]
    pos_err2 = data2["x"]["pos"]-data2["ref"]
    pos_err3 = data3["x"]["pos"]-data3["ref"]
    for i in range(np.shape(data1["x"]["pos"][:, 0, 0])[0]):
        pos_bounds[i] = (rho[0]-rho[1]) * np.exp(-rho_k*data1["t"][i]) + rho[1]

    fig, axes = plt.subplots(nrows=3, figsize=(9, 7), sharex=True)
    for i, (_label, ax) in enumerate(zip([r"$e_x$", r"$e_y$", r"$e_z$"], axes)):
        ax.plot(data1["t"], pos_err2[:, i, 0], "k-", label="BS (same)")
        ax.plot(data1["t"], pos_err3[:, i, 0], "g--", label="BS (different)")
        ax.plot(data1["t"], pos_err1[:, i, 0], "b--", label="Proposed")
        ax.plot(data1["t"], pos_bounds, "r:", label="Prescribed Bound")
        ax.plot(data1["t"], -pos_bounds, "r:")
        ax.set_ylabel(_label)
        if i == 0:
            ax.legend(loc=[0, 1.03], ncol=4, mode="expand")
        if i == 0:
            axins = zoomed_inset_axes(ax, 2, loc="upper right",
                                      axes_kwargs={"facecolor": "lavender"})
            axins.plot(data1["t"], pos_err2[:, i, 0], "k-")
            axins.plot(data1["t"], pos_err3[:, i, 0], "g--")
            axins.plot(data1["t"], pos_err1[:, i, 0], "b--")
            axins.plot(data1["t"], pos_bounds, "r:")
            axins.plot(data1["t"], -pos_bounds, "r:")
            x1, x2, y1, y2 = 4, 8, -0.3, 0.3
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            mark_inset(ax, axins, loc1=2, loc2=4, fc="lavender", edgecolor="lightgray", ec="0.5")
    plt.gcf().supxlabel("Time [sec]")
    plt.tight_layout()

    # 4e) STD of tracking error
    for i, _label in enumerate([r"$e_x$", r"$e_y$", r"$e_z$"]):
        print("STD of proposed controller: " + _label + str(statistics.stdev(data1["x"]["pos"][:, i, 0]-data1["ref"][:, i, 0])))
        print("STD of BS (same): " + _label + str(statistics.stdev(data2["x"]["pos"][:, i, 0]-data2["ref"][:, i, 0])))
        print("STD of BS (diff): " + _label + str(statistics.stdev(data3["x"]["pos"][:, i, 0]-data3["ref"][:, i, 0])))
        print("\n")

    # 4f) comparison on x, y, z axis respectively (max, mean)
    for i, _label in enumerate([r"$e_x$", r"$e_y$", r"$e_z$"]):
        for j in range(np.size(fault_time)):
            overshoot1 = max(abs(pos_err1[int(fault_time[j]/dt):int(fault_time[j]/dt+1.5/dt), i, 0]))
            index = np.where(abs(pos_err1[:, i, 0]) == overshoot1)
            overshoot2 = np.sign(pos_err1[index, i, 0]) * overshoot1
            print("overshoot of propsed after " + str(j) + "-th fault: " + _label + str(overshoot2))
            overshoot3 = max(abs(pos_err2[int(fault_time[j]/dt):int(fault_time[j]/dt+1.5/dt), i, 0]))
            index = np.where(abs(pos_err2[:, i, 0]) == overshoot3)
            overshoot4 = np.sign(pos_err2[index, i, 0]) * overshoot3
            print("overshoot of BS (same) after " + str(j) + "-th fault: " + _label + str(overshoot4))
            overshoot5 = max(abs(pos_err3[int(fault_time[j]/dt):int(fault_time[j]/dt+1.5/dt), i, 0]))
            index = np.where(abs(pos_err3[:, i, 0]) == overshoot5)
            overshoot6 = np.sign(pos_err3[index, i, 0]) * overshoot5
            print("overshoot of BS (diff) after " + str(j) + "-th fault: " + _label + str(overshoot6))
            print("\n")
        print("absolute mean error(overall) of proposed: " + _label + str(np.mean(abs(data1["x"]["pos"][:, i, 0]-data1["ref"][:, i, 0]))))
        print("absolute mean error(before fault) of proposed: " + _label + str(np.mean(abs(data1["x"]["pos"][:int(fault_time[j]/dt), i, 0]-data1["ref"][:int(fault_time[j]/dt), i, 0]))))
        print("absolute mean error(after fault) of proposed: " + _label + str(np.mean(abs(data1["x"]["pos"][int(fault_time[j]/dt):, i, 0]-data1["ref"][int(fault_time[j]/dt):, i, 0]))))
        print("\n")
        print("absolute mean error of BS (same): " + _label + str(np.mean(abs(data2["x"]["pos"][:, i, 0]-data2["ref"][:, i, 0]))))
        print("absolute mean error(before fault) of BS (same): " + _label + str(np.mean(abs(data2["x"]["pos"][:int(fault_time[j]/dt), i, 0]-data2["ref"][:int(fault_time[j]/dt), i, 0]))))
        print("absolute mean error(after fault) of BS (same): " + _label + str(np.mean(abs(data2["x"]["pos"][int(fault_time[j]/dt):, i, 0]-data2["ref"][int(fault_time[j]/dt):, i, 0]))))
        print("\n")
        print("absolute mean error of BS (diff): " + _label + str(np.mean(abs(data3["x"]["pos"][:, i, 0]-data3["ref"][:, i, 0]))))
        print("absolute mean error(before fault) of BS (diff): " + _label + str(np.mean(abs(data3["x"]["pos"][:int(fault_time[j]/dt), i, 0]-data3["ref"][:int(fault_time[j]/dt), i, 0]))))
        print("absolute mean error(after fault) of BS (diff): " + _label + str(np.mean(abs(data3["x"]["pos"][int(fault_time[j]/dt):, i, 0]-data3["ref"][int(fault_time[j]/dt):, i, 0]))))
        print("\n")

    ''' 5. euler '''
    # 5a) Euler angle trajectories
    plt.figure(figsize=(9, 7))
    bound = 45
    plt.ylim([-bound-5, bound+5])

    angles1 = np.vstack([quat2angle(data1["x"]["quat"][j, :, 0]) for j in range(len(data1["x"]["quat"][:, 0, 0]))])
    angles2 = np.vstack([quat2angle(data2["x"]["quat"][j, :, 0]) for j in range(len(data2["x"]["quat"][:, 0, 0]))])
    angles3 = np.vstack([quat2angle(data2["x"]["quat"][j, :, 0]) for j in range(len(data2["x"]["quat"][:, 0, 0]))])
    ax = plt.subplot(311)
    for i, _label in enumerate([r"$\phi$", r"$\theta$", r"$\psi$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data2["t"], np.rad2deg(angles2[:, 2-i]), "k-", label="BS (same)")
        plt.plot(data2["t"], np.rad2deg(angles3[:, 2-i]), "g--", label="BS (different)")
        plt.plot(data1["t"], np.rad2deg(angles1[:, 2-i]), "b--", label="Proposed")
        plt.plot(data1["t"], np.ones((np.size(data1["t"])))*bound, "r:", label="Bound")
        plt.plot(data1["t"], -np.ones((np.size(data1["t"])))*bound, "r:")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc=[0, 1.03], ncol=4, mode="expand")
    plt.gcf().supxlabel("Time [sec]")
    plt.tight_layout()
    # plt.savefig("angle.png", dpi=300)

    # 5b) Angular rate trajectories
    bound = 150
    bound_psi = 180

    fig, axes = plt.subplots(nrows=3, figsize=(9, 7), sharex=True)
    for i, (_label, ax) in enumerate(zip(["p", "q", "r"], axes)):
        ax.plot(data2["t"], np.rad2deg(data2["x"]["omega"][:, i, 0]), "k-", label="BS (same)")
        ax.plot(data2["t"], np.rad2deg(data3["x"]["omega"][:, i, 0]), "g--", label="BS (different)")
        ax.plot(data1["t"], np.rad2deg(data1["x"]["omega"][:, i, 0]), "b--", label="Proposed")
        if i == 2:
            ax.plot(data1["t"], np.ones((np.size(data1["t"])))*bound_psi, "r:",
                    label="Prescribed Bound")
            ax.plot(data1["t"], -np.ones((np.size(data1["t"])))*bound_psi, "r:")
            ax.set_ylim([-bound_psi-15, bound_psi+15])
        else:
            ax.plot(data1["t"], np.ones((np.size(data1["t"])))*bound, "r:",
                    label="Prescribed Bound")
            ax.plot(data1["t"], -np.ones((np.size(data1["t"])))*bound, "r:")
        if i == 1:
            plt.ylim([-200, 200])
        elif i == 0:
            ax.legend(loc=[0, 1.03], ncol=4, mode="expand")
            ax.set_ylim([-bound-15, bound+15])
        ax.set_ylabel(_label)
        if i == 1:
            axins = zoomed_inset_axes(ax, 5, loc="upper left",
                                      axes_kwargs={"facecolor": "lavender"})
            axins.plot(data1["t"], np.rad2deg(data1["x"]["omega"][:, i, 0]), "b--")
            axins.plot(data1["t"], -np.ones((np.size(data1["t"])))*bound, "r:")
            x1, x2, y1, y2 = 0, 0.2, -155, -130
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            mark_inset(ax, axins, loc1=3, loc2=2, fc="lavender", edgecolor="lightgray", ec="0.5")
        if i == 1:
            axins1 = zoomed_inset_axes(ax, 2.5, loc="lower center",
                                       axes_kwargs={"facecolor": "lavender"})
            axins1.plot(data2["t"], np.rad2deg(data2["x"]["omega"][:, i, 0]), "k-", label="BS (same)")
            axins1.plot(data2["t"], np.rad2deg(data3["x"]["omega"][:, i, 0]), "g--", label="BS (different)")
            axins1.plot(data1["t"], np.rad2deg(data1["x"]["omega"][:, i, 0]), "b--")
            axins1.plot(data1["t"], -np.ones((np.size(data1["t"])))*bound, "r:")
            x1, x2, y1, y2 = 4.5, 5.5, -170, -90
            axins1.set_xlim(x1, x2)
            axins1.set_ylim(y1, y2)
            axins1.set_xticks([])
            axins1.set_yticks([])
            mark_inset(ax, axins1, loc1=2, loc2=4, fc="lavender", edgecolor="lightgray", ec="0.5")
    plt.gcf().supxlabel("Time [sec]")
    plt.tight_layout()
    # plt.savefig("angular.png", dpi=300)

    # 5c) tracking error
    plt.figure()
    ax = plt.subplot(311)
    for i, _label in enumerate([r"$e_{\phi}$", r"$e_{\theta}$", r"$e_{\psi}$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data2["t"], np.rad2deg(angles2[:, 2-i])-np.rad2deg(data2["eulerd"][:, i, 0]), "k-",
                 label="BS (same)")
        plt.plot(data2["t"], np.rad2deg(angles3[:, 2-i])-np.rad2deg(data3["eulerd"][:, i, 0]), "g--",
                 label="BS (different)")
        plt.plot(data1["t"], np.rad2deg(angles1[:, 2-i])-np.rad2deg(data1["eulerd"][:, i, 0]), "b--",
                 label="Proposed")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc=[0, 1.03], ncol=3, mode="expand")
    plt.gcf().supxlabel("Time [sec]")
    plt.tight_layout()
    # plt.savefig("angle_error.png", dpi=300)

    # 5d) STD of tracking error
    for i, _label in enumerate([r"$e_{\phi}$", r"$e_{\theta}$", r"$e_{\psi}$"]):
        print("STD of proposed controller: " + _label + str(statistics.stdev(np.rad2deg(angles1[:, 2-i])-np.rad2deg(data1["eulerd"][:, i, 0]))))
        print("STD of BS (same): " + _label + str(statistics.stdev(np.rad2deg(angles2[:, 2-i])-np.rad2deg(data2["eulerd"][:, i, 0]))))
        print("STD of BS (different): " + _label + str(statistics.stdev(np.rad2deg(angles3[:, 2-i])-np.rad2deg(data3["eulerd"][:, i, 0]))))
        print("\n")

    # 5e) comparison on phi, theta, psi respectively (max, mean)
    for i, _label in enumerate([r"$\phi$", r"$\theta$", r"$\psi$"]):
        for j in range(np.size(fault_time)):
            overshoot1 = max(abs(
                angles1[int(fault_time[j]/dt):int(fault_time[j]/dt+1.5/dt), 2-i]
                - data1["eulerd"][int(fault_time[j]/dt):int(fault_time[j]/dt+1.5/dt), i, 0]))
            index = np.where(abs(angles1[:, 2-i]-data1["eulerd"][:, i, 0]) == overshoot1)
            overshoot2 = np.sign(angles1[index, 2-i]-data1["eulerd"][index, i, 0]) * overshoot1
            print("overshoot of proposed after " + str(j) + "-th fault: " + _label + str(np.rad2deg(overshoot2)))
            overshoot3 = max(abs(
                angles2[int(fault_time[j]/dt):int(fault_time[j]/dt+1.5/dt), 2-i]
                - data2["eulerd"][int(fault_time[j]/dt):int(fault_time[j]/dt+1.5/dt), i, 0]))
            index = np.where(abs(angles2[:, 2-i]-data2["eulerd"][:, i, 0]) == overshoot3)
            overshoot4 = np.sign(angles2[index, 2-i]-data2["eulerd"][index, i, 0]) * overshoot3
            print("overshoot of BS (same) after " + str(j) + "-th fault: " + _label + str(np.rad2deg(overshoot4)))
            overshoot5 = max(abs(
                angles3[int(fault_time[j]/dt):int(fault_time[j]/dt+1.5/dt), 2-i]
                - data3["eulerd"][int(fault_time[j]/dt):int(fault_time[j]/dt+1.5/dt), i, 0]))
            index = np.where(abs(angles3[:, 2-i]-data3["eulerd"][:, i, 0]) == overshoot5)
            overshoot6 = np.sign(angles3[index, 2-i]-data3["eulerd"][index, i, 0]) * overshoot5
            print("overshoot of BS (diff) after " + str(j) + "-th fault: " + _label + str(np.rad2deg(overshoot6)))
            print("\n")
        print("absoulte mean error(overall) of proposed: " + _label + str(np.mean(abs(np.rad2deg(angles1[:, 2-i])-np.rad2deg(data1["eulerd"][:, i, 0])))))
        print("absoulte mean error(before fault) of proposed: " + _label + str(np.mean(abs(np.rad2deg(angles1[:int(fault_time[j]/dt), 2-i])-np.rad2deg(data1["eulerd"][:int(fault_time[j]/dt), i, 0])))))
        print("absoulte mean error(after fault) of proposed: " + _label + str(np.mean(abs(np.rad2deg(angles1[int(fault_time[j]/dt):, 2-i])-np.rad2deg(data1["eulerd"][int(fault_time[j]/dt):, i, 0])))))
        print("\n")
        print("absoulte mean error(overall) of BS (same): " + _label + str(np.mean(abs(np.rad2deg(angles2[:, 2-i])-np.rad2deg(data2["eulerd"][:, i, 0])))))
        print("absoulte mean error(before fault) of BS (same): " + _label + str(np.mean(abs(np.rad2deg(angles2[:int(fault_time[j]/dt), 2-i])-np.rad2deg(data2["eulerd"][:int(fault_time[j]/dt), i, 0])))))
        print("absoulte mean error(after fault) of BS (same): " + _label + str(np.mean(abs(np.rad2deg(angles2[int(fault_time[j]/dt):, 2-i])-np.rad2deg(data2["eulerd"][int(fault_time[j]/dt):, i, 0])))))
        print("\n")
        print("absoulte mean error(overall) of BS (diff): " + _label + str(np.mean(abs(np.rad2deg(angles3[:, 2-i])-np.rad2deg(data3["eulerd"][:, i, 0])))))
        print("absoulte mean error(before fault) of BS (diff): " + _label + str(np.mean(abs(np.rad2deg(angles3[:int(fault_time[j]/dt), 2-i])-np.rad2deg(data3["eulerd"][:int(fault_time[j]/dt), i, 0])))))
        print("absoulte mean error(after fault) of BS (diff): " + _label + str(np.mean(abs(np.rad2deg(angles3[int(fault_time[j]/dt):, 2-i])-np.rad2deg(data3["eulerd"][int(fault_time[j]/dt):, i, 0])))))
        print("\n")

    # 5f) comparison on p, q, r respectively (max, mean)
    for i, _label in enumerate([r"$p$", r"$q$", r"$r$"]):
        for j in range(np.size(fault_time)):
            overshoot1 = max(abs(data1["x"]["omega"][int(fault_time[j]/dt):int(fault_time[j]/dt+1.5/dt), i, 0]))
            index = np.where(abs(data1["x"]["omega"][:, i, 0]) == overshoot1)
            overshoot2 = np.sign(data1["x"]["omega"][index, i, 0]) * overshoot1
            print("overshoot of proposed after " + str(j) + "-th fault: " + _label + str(np.rad2deg(overshoot2)))
            overshoot3 = max(abs(data2["x"]["omega"][int(fault_time[j]/dt):int(fault_time[j]/dt+1/dt), i, 0]))
            index = np.where(abs(data2["x"]["omega"][:, i, 0]) == overshoot3)
            overshoot4 = np.sign(data2["x"]["omega"][index, i, 0]) * overshoot3
            print("overshoot of BS (same) after " + str(j) + "-th fault: " + _label + str(np.rad2deg(overshoot4)))
            overshoot5 = max(abs(data3["x"]["omega"][int(fault_time[j]/dt):int(fault_time[j]/dt+1/dt), i, 0]))
            index = np.where(abs(data3["x"]["omega"][:, i, 0]) == overshoot5)
            overshoot6 = np.sign(data3["x"]["omega"][index, i, 0]) * overshoot5
            print("overshoot of BS (diff) after " + str(j) + "-th fault: " + _label + str(np.rad2deg(overshoot6)))
            print("\n")
        print("absolute mean error(overall) of proposed: " + _label + str(np.mean(abs(np.rad2deg(data1["x"]["omega"][:, i, 0])-np.zeros((np.shape(data1["x"]["omega"][:, i, 0])))))))
        print("absolute mean error(before fault) of proposed: " + _label + str(np.mean(abs(np.rad2deg(data1["x"]["omega"][:int(fault_time[0]/dt), i, 0])-np.zeros((np.shape(data1["x"]["omega"][:int(fault_time[0]/dt), i, 0])))))))
        print("absolute mean error(after fault) of proposed: " + _label + str(np.mean(abs(np.rad2deg(data1["x"]["omega"][int(fault_time[0]/dt):, i, 0])-np.zeros((np.shape(data1["x"]["omega"][int(fault_time[0]/dt):, i, 0])))))))
        print("\n")
        print("absolute mean error(overall) of BS (same): " + _label + str(np.mean(abs(np.rad2deg(data2["x"]["omega"][:, i, 0])-np.zeros((np.shape(data2["x"]["omega"][:, i, 0])))))))
        print("absolute mean error(before fault) of BS (same): " + _label + str(np.mean(abs(np.rad2deg(data2["x"]["omega"][:int(fault_time[0]/dt), i, 0])-np.zeros((np.shape(data2["x"]["omega"][:int(fault_time[0]/dt), i, 0])))))))
        print("absolute mean error(after fault) of BS (same): " + _label + str(np.mean(abs(np.rad2deg(data2["x"]["omega"][int(fault_time[0]/dt):, i, 0])-np.zeros((np.shape(data2["x"]["omega"][int(fault_time[0]/dt):, i, 0])))))))
        print("\n")
        print("absolute mean error(overall) of BS (diff): " + _label + str(np.mean(abs(np.rad2deg(data3["x"]["omega"][:, i, 0])-np.zeros((np.shape(data3["x"]["omega"][:, i, 0])))))))
        print("absolute mean error(before fault) of BS (diff): " + _label + str(np.mean(abs(np.rad2deg(data3["x"]["omega"][:int(fault_time[0]/dt), i, 0])-np.zeros((np.shape(data3["x"]["omega"][:int(fault_time[0]/dt), i, 0])))))))
        print("absolute mean error(after fault) of BS (diff): " + _label + str(np.mean(abs(np.rad2deg(data3["x"]["omega"][int(fault_time[0]/dt):, i, 0])-np.zeros((np.shape(data3["x"]["omega"][int(fault_time[0]/dt):, i, 0])))))))
        print("\n")

    ''' 6. etc '''
    # 6a) rotor input comparison
    plt.figure(figsize=(7, 5))

    name = [r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$", r"$\Omega_4$"]
    ax = plt.subplot(411)
    for i in range(data1["rotors"].shape[1]):
        if i != 0:
            plt.subplot(411+i, sharex=ax)
        plt.ylim([rotor_min-5, np.sqrt(rotor_max)+5])
        plt.plot(data2["t"], np.sqrt(data2["rotors"][:, i]), "k-", label="BS (same)")
        plt.plot(data2["t"], np.sqrt(data3["rotors"][:, i]), "g--", label="BS (different)")
        plt.plot(data1["t"], np.sqrt(data1["rotors"][:, i]), "b--", label="Proposed")
        plt.ylabel(name[i])
        if i == 0:
            plt.legend(loc=[0, 1.03], ncol=3, mode="expand")
    plt.gcf().supxlabel("Time [sec]")
    plt.tight_layout()
    # plt.savefig("rotor_input.png")

    # 6b) generalized forces comparison
    plt.figure(figsize=(7, 5))

    ax = plt.subplot(411)
    for i, _label in enumerate([r"$u_{1}$", r"$u_{2}$", r"$u_{3}$", r"$u_{4}$"]):
        if i != 0:
            plt.subplot(411+i, sharex=ax)
        plt.plot(data2["t"], data2["virtual_u"][:, i], "k-", label="BS (same)")
        plt.plot(data2["t"], data3["virtual_u"][:, i], "g--", label="BS (different)")
        plt.plot(data1["t"], data1["virtual_u"][:, i], "b--", label="Proposed")
        if i == 0:
            plt.legend(loc=[0, 1.03], ncol=3, mode="expand")
            plt.ylabel(_label, labelpad=18)
        elif i == 1:
            plt.ylabel(_label, labelpad=0)
        elif i == 2:
            plt.ylabel(_label, labelpad=21)
        elif i == 3:
            plt.ylabel(_label, labelpad=2)
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()
    # plt.savefig("forces.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    exp_plot("Scenario1_BLF_additional.h5", "Scenario1_Bs.h5", "Scenario1_Bs_additional.h5")
    # exp_plot("Scenario1_BLF_additional.h5", "Scenario1_Bs.h5")
