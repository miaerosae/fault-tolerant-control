import numpy as np
import matplotlib.pyplot as plt

import ftc.config


cfg = ftc.config.load()

plt.rc("text", usetex=False)
plt.rc("lines", linewidth=1.5)
plt.rc("axes", grid=True, labelsize=15, titlesize=15)
plt.rc("grid", linestyle="--", alpha=0.8)
plt.rc("legend", fontsize=15)


def gain_bound():
    rho_inf, rho_k = cfg.agents.BLF.oL.rho[1], cfg.agents.BLF.oL.rho_k
    rho1, rho2 = cfg.agents.BLF.iL.rho.ravel()

    kI = 10
    k2 = np.linspace(1, 30, int(29/0.1))

    # outer loop, kD const
    kD_1 = 32
    k1_1 = np.zeros((np.size(k2),))
    k3_1 = np.zeros((np.size(k2),))
    kP_1 = np.zeros((np.size(k2),))
    for i in range(np.size(k2)):
        k1_1[i] = kD_1 - k2[i]
        k3_1[i] = kI / k2[i] / rho_inf**2
        kP_1[i] = 1 / k2[i] * (kI + kD_1*k2[i]**2-k2[i]**3) - 1 / rho_inf**2
    fig, ax = plt.subplots()
    plt.xlim([0, max(kP_1)])
    ax.plot(kP_1, k1_1, "r-", label=r"$k_1$")
    ax.plot(kP_1, k2, "g--", label=r"$k_2$")
    ax.plot(kP_1, k3_1, "b-", label=r"$k_3$")
    k1_1[k1_1 == np.inf] = 0
    k3_1[k3_1 == np.inf] = 0
    kP_1[kP_1 == np.inf] = 0
    plt.legend(loc=[0, 1.03], ncol=3, mode="expand")
    plt.xlabel(r"$k_P$")
    plt.ylabel(r"$k_1$, $k_2$, $k_3$")
    plt.savefig("PID_gain_range_KD1.png", dpi=600)

    # outer loop, kP const
    kP_2 = 30
    k1_2 = np.zeros((np.size(k2),))
    k3_2 = np.zeros((np.size(k2),))
    kD_2 = np.zeros((np.size(k2),))
    for i in range(np.size(k2)):
        kD_2[i] = 1 / k2[i]**2 * (k2[i]**3 + (kP_2+1/rho_inf**2)*k2[i] - kI)
        k1_2[i] = kD_2[i] - k2[i]
        k3_2[i] = kI / k2[i] / rho_inf**2
    fig, ax = plt.subplots()
    plt.xlim([0, max(kD_2)])
    ax.plot(kD_2, k1_2, "r-", label=r"$k_1$")
    ax.plot(kD_2, k2, "g--", label=r"$k_2$")
    ax.plot(kD_2, k3_2, "b-", label=r"$k_3$")
    k1_2[k1_2 == np.inf] = 0
    k3_2[k3_2 == np.inf] = 0
    kD_2[kD_2 == np.inf] = 0
    plt.legend(loc=[0, 1.03], ncol=3, mode="expand")
    plt.xlabel(r"$k_D$")
    plt.ylabel(r"$k_1$, $k_2$, $k_3$")
    plt.savefig("PID_gain_range_KP1.png", dpi=600)

    # inner loop, kD const
    kD_3 = 45
    k1_3 = np.zeros((np.size(k2),))
    k3_3 = np.zeros((np.size(k2),))
    kP_3 = np.zeros((np.size(k2),))
    for i in range(np.size(k2)):
        nk = np.sqrt(0.1)
        k1_3[i] = kD_3 - k2[i] - 2*nk
        k3_3[i] = kI / (k2[i] + nk)
        kP_3[i] = ((- k2[i]**3 + (kD_3-2*nk)*k2[i]**2
                    + (1 - nk/(k2[i]+nk))*kI) / k2[i]
                   + nk * (kD_3 - 2*nk) + nk)
    fig, ax = plt.subplots()
    plt.xlim([0, max(kP_3)])
    ax.plot(kP_3, k1_3, "r-", label=r"$k_1$")
    ax.plot(kP_3, k2, "g--", label=r"$k_2$")
    ax.plot(kP_3, k3_3, "b-", label=r"$k_3$")
    # ax.fill_between(kP_3, 0, 1, where=k2 > 2*rho2/rho1,
    #                 interpolate=True, color="pink", alpha=0.2,
    #                 transform=ax.get_xaxis_transform())
    # ax.fill_between(kP_3, 0, 1, where=k1_3*k2 > k1_3-rho2**2/rho1**2,
    #                 interpolate=True, color="pink", alpha=0.2,
    #                 transform=ax.get_xaxis_transform(), label="possible range")
    plt.legend(loc=[0, 1.03], ncol=3, mode="expand")
    plt.xlabel(r"$k_P$")
    plt.ylabel(r"$k_1$, $k_2$, $k_3$")
    plt.savefig("PID_gain_range_KD2.png", dpi=600)

    # # inner loop, kP const
    kP_4 = 500
    k1_4 = np.zeros((np.size(k2),))
    k3_4 = np.zeros((np.size(k2),))
    kD_4 = np.zeros((np.size(k2),))
    for i in range(np.size(k2)):
        nk = np.sqrt(0.1)
        kD_4[i] = ((k2[i]**3 + 2*nk*k2[i]**2 + (kP_4+0.1)*k2[i]
                    - (1 - nk/(k2[i]+nk))*kI)
                   / (k2[i]**2 + nk*k2[i]))
        k1_4[i] = kD_4[i] - k2[i] - 2*nk
        k3_4[i] = kI / (k2[i] + nk)
    fig, ax = plt.subplots()
    plt.xlim([0, max(kD_4)])
    ax.plot(kD_4, k1_4, "r-", label=r"$k_1$")
    ax.plot(kD_4, k2, "g--", label=r"$k_2$")
    ax.plot(kD_4, k3_4, "b-", label=r"$k_3$")
    # ax.fill_between(kD_4, 0, 1, where=k2 > 2*rho2/rho1,
    #                 interpolate=True, color="pink", alpha=0.2,
    #                 transform=ax.get_xaxis_transform())
    # ax.fill_between(kD_4, 0, 1, where=k1_4*k2 > k1_4-rho2**2/rho1**2,
    #                 interpolate=True, color="pink", alpha=0.2,
    #                 transform=ax.get_xaxis_transform(), label="possible range")
    plt.legend(loc=[0, 1.03], ncol=3, mode="expand")
    plt.xlabel(r"$k_D$")
    plt.ylabel(r"$k_1$, $k_2$, $k_3$")
    plt.savefig("PID_gain_range_KP2.png", dpi=600)

    plt.show()


if __name__ == "__main__":
    gain_bound()
