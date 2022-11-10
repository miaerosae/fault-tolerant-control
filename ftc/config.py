"""References
[1] https://www.math.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf
[2] V. S. Akkinapalli, G. P. Falconí, and F. Holzapfel, “Attitude control of a multicopter using L1 augmented quaternion based backstepping,” Proceeding - ICARES 2014 2014 IEEE Int. Conf. Aerosp. Electron. Remote Sens. Technol., no. November, pp. 170–178, 2014.
[3] M. C. Achtelik, K. M. Doth, D. Gurdan, and J. Stumpf, “Design of a multi rotor MAV with regard to efficiency, dynamics and redundancy,” AIAA Guid. Navig. Control Conf. 2012, no. August, pp. 1–17, 2012.
[4] https://kr.mathworks.com/help/aeroblks/6dofquaternion.html#mw_f692de78-a895-4edc-a4a7-118228165a58
[5] M. C. Achtelik, K. M. Doth, D. Gurdan, and J. Stumpf, “Design of a multi rotor MAV with regard to efficiency, dynamics and redundancy,” AIAA Guid. Navig. Control Conf. 2014, no. August, pp. 1–17, 2012, doi: 10.2514/6.2012-4779.
"""
import numpy as np
from functools import reduce

import fym


default_settings = fym.parser.parse({
    # :::::: FTC Modules :::::: #

    # ====== ftc.faults ====== #

    # ------ ftc.faults.manager ------ #

    "faults.manager": {
        "delay": 0.2,
        "threshold": 0.,
        "fault_time": np.array([5, 7, 10, 11]),
        "fault_index": np.array([0, 1, 2, 2]),
        "LoE": np.array([0.5, 0.6, 0.8, 0.4]),
        # "fault_time": np.array([5]),
        # "fault_index": np.array([0]),
        # "LoE": np.array([0.5]),
    },

    # ====== ftc.plants ====== #

    # ------ ftc.plants.multicopter ------ #

    "models.multicopter": {
        # Initial states
        "init": {
            "pos": np.zeros((3, 1)),
            "vel": np.zeros((3, 1)),
            "quat": np.vstack((1, 0, 0, 0)),
            "omega": np.zeros((3, 1)),
        },

        # Mixer
        "mixer.rtype": "quad",
        # "mixer.rtype": "hexa-x",

        # Physical properties
        "physProp": {
            # General physical constants
            "g": 9.81,
            "rho": 1.225,

            # Parameters from Baldini et al., 2020
            "kr": 1e-3 * np.eye(3),  # Rotational friction coefficient [N*s*m/rad]
            # "Jr": 6e-5,  # Rotor inertia [N*m]
            "CdA": 0.08,  # Flat plate area [m^2]
            # "R": 0.15,  # Rotor radius [m]
            # "ch": 0.04,  # Propeller chord [m]
            "a0": 6,  # Slope of the lift curve per radian [-]

            "max_IGE_ratio": 1.6,  # TODO: maximum IGE ratio
            "Rrad": 0.15,  # propeller radius
            "ch": 0.014,  # propeller chord
            "theta0": 0.26,  # incidence angle
            "thetatw": 0.045,  # twist angle
            "Jr": 0.6 * 1e-4,  # rotor inertia
            "t_max": 0.15,  # max torque
            "w_max": 1000,  # max rotor speed
            "h": 2.56e-2,  # CoG to rot. plane

            # for wind disturbances
            "Au": (0.23/0.2)**2*0.1,
            "Av": (0.23/0.2)**2*0.1,
            "Aw": (0.23/0.2)**2*0.2,
            "Cdx": 0.6,
            "Cdy": 0.6,
            "Cdz": 1,

            # Parameters from P. Pounds et al., 2010
            "sigma": 0.054,  # Solidity ratio [-]
            # "thetat": np.deg2rad(4.4),  # Blade tip angle [rad]
            "CT": 0.0047,  # Thrust coefficient [-]
        },

        # Physical properties by several authors
        "modelFrom": "OS4",

        "physPropBy": {
            # Prof. Taeyoung Lee's model for quadrotor UAV [1]
            "Taeyoung_Lee": {
                "J": np.diag([0.0820, 0.0845, 0.1377]),
                "m": 4.34,
                "d": 0.315,  # distance to each rotor from the center of mass
                "c": 8.004e-4,  # z-dir moment coefficient caused by rotor force
                "b": 1,
                "rotor_min": 0,
            },

            # G. P. Falconi's multicopter model [2-4]
            "GP_Falconi": {
                "J": np.diag([0.010007, 0.0102335, 0.0081]),
                "m": 0.64,
                "d": 0.215,  # distance to each rotor from the center of mass
                "c": 1.2864e-7,  # z-dir moment coefficient caused by rotor force
                "b": 6.546e-6,
                "rotor_min": 0,
            },

            # OS4
            "OS4": {
                "J": np.diag([0.0075, 0.0075, 0.013]),
                "m": 0.65,
                "d": 0.23,  # arm length
                "c": 0.75 * 1e-6,  # drag moment coefficient
                "b": 0.0000313,  # thrust coefficient
                "rotor_min": 0,
            },
        },
        "model_uncert": {
            "del_m": 0.1,
            "del_J": np.diag([-0.1, 0.1, 0.05]),
            "del_c": -0.05,
            "del_b": 0.05,
        },
    },

    # :::::: BLF :::::: #

    # ====== agents.BLF ====== #

    "agents.BLF": {

        # --- outerLoop --- #
        "oL": {
            "alp": np.array([3, 3, 1]),
            # "eps": np.array([8.406469764729502, 27.77393258460113, 20.06893194607595]),
            # "eps": np.array([1/5, 1/5, 1/5]),
            "eps": np.array([60, 70, 40]),
            "rho": np.array([0.5, 0.25]),
            "rho_k": 0.6,
            "gamma": np.array([2, 2, 2]),
        },
        # --- innerLoop --- #
        "iL": {
            "alp": np.array([3, 3, 1]),
            # "eps": np.array([1/25, 1/25, 1/25]),
            "eps": np.array([110, 250, 80]),
            "xi": np.array([-1, 1]) * 0.23 * 0.000313 * 1e6,
            "xi_psi": np.array([-1, 1]) * 2 * 0.75,
            "rho": np.deg2rad(np.array([45, 150])),
            "rho_psi": np.deg2rad([45, 180]),
            "c": np.array([200, 200]),
            "gamma": np.array([2, 2, 2]),
        },
        # --- gain K --- #
        "Kxy": np.array([1, 0.5, 0.417]),
        "Kang": np.array([10.5, 100, 0.97]),
        # "Kang": np.array([33.59647405167779, 182.67932574166815, 0]),
        # "Kxy": np.array([3, 2, 0]),
        # "Kang": np.array([3, 2, 0]),
        "theta": 0.7,

        # --- peaking-free --- #
        "pf.oL": {
            "l": np.array([50, 70, 80]),
            "alp": np.array([2.9, 2.9, 2]),
            "alp_y": np.array([2.9, 2.9, 2]),
            "bet": np.array([3.98, 0.993]),
            "rho": np.array([0.5, 0.25]),
            "rho_k": 0.6,
            "R": np.array([[0.6, 2],
                           [0.6, 3],
                           [1.8, 10]]),
        },
        "pf.iL": {
            "l": np.array([55, 60, 60]),
            "alp": np.array([2.9, 2.9, 2]),
            "alp_phi": np.array([2.9, 2.9, 30]),
            "alp_theta": np.array([2.9, 2.9, 40]),
            "alp_psi": np.array([2.9, 2.9, 2]),
            "bet": np.array([3.98, 0.993]),
            "xi": np.array([-1, 1]) * 0.23 * 0.000313 * 1e6,
            "xi_psi": np.array([-1, 1]) * 2 * 0.75,
            "rho": np.deg2rad(np.array([45, 150])),
            "rho_psi": np.deg2rad(np.array([45, 180])),
            "c": np.array([200, 200]),
            "dist_range_phi": 40,  # disturbance saturation value
            "dist_range_theta": 60,  # disturbance saturation value
            "dist_range_psi": 10,  # disturbance saturation value
        },
        "pf.Kxy": np.array([0.2651, 4.9938, 0.8745]),
        "pf.Kang": np.array([29.9895, 125.3485, 0.2621]),
        "pf.theta.pos": 0.8,
        "pf.theta.ang": 0.7,

    },

    # :::::: SIMULATION CONDITION :::::: #

    "simul_condi": {
        "dt": 0.01,
        "max_t": 30,
        "blade": False,
        "faultBias": False,  # not use
        "noise": False,  # Estimator real value noise
        "groundEffect": False,
        "hub": False,  # not use
        "gyro": False,
        "drygen": False,  # not use
        # "BLF": False,
        "BLF": True,
        "uncertainty": True,
        "ext_unc": True,
        "int_unc": True,
        # "uncertainty": False,
        # "ext_unc": False,
        # "int_unc": False,
    },

    "wind_dist": {
        "t1": 10,
        "t2": 20,
        "vx1": -10,
        "vx2": -10,
        "vy1": -12,
        "vy2": -12,
        "vz1": -5,
        "vz2": -8,
        "W20": 15,
    },


    # :::::: FTC EVALUATION ENV :::::: #

    # ====== env ====== #

    "parallel.max_workers": None,
    "episode.N": 100,
    "episode.range": {
        "pos": (-1, 1),
        "vel": (-1, 1),
        "omega": np.deg2rad((-5, 5)),
        "angle": np.deg2rad((-5, 5)),
    },
    "evaluation.cuttime": 5,
    "evaluation.threshold": 0.5,
    "env.kwargs": {
        "dt": 0.01,
        "max_t": 10,
    },
    "ref.pos": np.vstack((0, 0, -10)),
})

settings = fym.parser.parse(default_settings)


def _maximum_thrust(m, g):
    return m * g * 0.6371  # maximum thrust for each rotor [5]


def _set_maximum_rotor(settings):
    modelauthor = settings.models.multicopter.modelFrom
    g = settings.models.multicopter.physProp.g
    if modelauthor == "Taeyoung_Lee":
        m = settings.models.multicopter.physPropBy.Taeyoung_Lee.m
        rotor_max = {"models.multicopter.physPropBy.Taeyoung_Lee":
                     {"rotor_max": _maximum_thrust(m, g)}
                     }
    elif modelauthor == "GP_falconi":
        rotor_max = {"models.multicopter.physPropBy.GP_falconi":
                     {"rotor_max": 3e5}  # about 2 * m * g / b / 6
                     }
    elif modelauthor == "OS4":
        rotor_max = {"models.multicopter.physPropBy.OS4":
                     {"rotor_max": 1e6}
                     }
    fym.parser.update(settings, rotor_max)


_set_maximum_rotor(settings)


def load(key=None):
    if isinstance(key, str):
        chunks = key.split(".")
        if key.startswith("ftc"):
            chunks.pop(0)
        return reduce(lambda v, k: v.__dict__[k], chunks, settings)
    return settings


def set(d):
    fym.parser.update(settings, d)


def reset():
    fym.parser.update(settings, default_settings, prune=True)
