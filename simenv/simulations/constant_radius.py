import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from uraeus.nmbd.python import simulation
from uraeus.nmbd.python.engine.numerics.math_funcs import A, B

database_directory = os.path.abspath('../../')
sys.path.append(database_directory)

from uraeus_fsae.simenv.assemblies import asurt_FS17_v1 as num_assm
from controllers import speed_controller, stanley_controller

num_model = num_assm.num_model

dt = num_assm.dt
TR = 254

def generate_circular_path(radius, offset):
    theta  = np.deg2rad(np.linspace(0, 360, 360))
    x_data = radius * np.sin(theta) + offset[0]
    y_data = radius * np.cos(theta) + offset[1]
    radii = radius * np.ones((360,))
    return x_data, y_data, radii


x_data, y_data, radii = generate_circular_path(10.5, (0, -10.5))

path_data = np.zeros((360, 3))
path_data[:, 0] = -1e3 * x_data
path_data[:, 1] =  1e3 * y_data
path_data[:, 2] =  1e3 * radii

plt.figure(figsize=(10, 5))
plt.plot(path_data[:, 0], path_data[:, 1])
plt.grid()
plt.show()

logitudinal_controller = speed_controller(35, dt)
lateral_controller = stanley_controller(path_data, 25)


def terrain_state(x, y):
    local_normal = np.array([[0],[0],[1]], dtype=np.float64)
    hieght = 0
    return [local_normal, hieght]


def torque_function(t):
    P_ch = num_model.Subsystems.CH.P_rbs_chassis
    Rd = num_model.Subsystems.CH.Rd_rbs_chassis
    factor = logitudinal_controller.get_torque_factor(P_ch, Rd)
    return factor

def RR_Torque(t):
    factor = torque_function(t)
    torque = -factor*(70*9.81)*1e6*TR
    return torque

def RL_Torque(t):
    factor = torque_function(t)
    torque = -factor*(70*9.81)*1e6*TR
    return torque

def steering_function(t):
    R_ch = num_model.Subsystems.CH.R_rbs_chassis
    P_ch = num_model.Subsystems.CH.P_rbs_chassis
    Rd_ch = num_model.Subsystems.CH.Rd_rbs_chassis
    Pd_ch = num_model.Subsystems.CH.Pd_rbs_chassis

    rbar_ax1 = np.array([[-800], [0], [0]], dtype=np.float64)
    r_ax1 = R_ch + A(P_ch)@rbar_ax1
    vel = (A(P_ch).T @ (Rd_ch + B(P_ch, rbar_ax1)@Pd_ch))[0,0]

    delta = lateral_controller.get_steer_factor(r_ax1, P_ch, Pd_ch,  vel)

    travel = delta * 18
    #print('Travel = %s'%travel)
    return travel


def zero_func(t):
    return np.zeros((3,1), dtype=np.float64)


num_assm.terrain_data.get_state = terrain_state

num_assm.ST1_config.UF_mcs_rack_act = steering_function

num_assm.AX1_config.UF_far_drive = RR_Torque
num_assm.AX1_config.UF_fal_drive = RL_Torque

#num_assm.DR2_config.UF_far_drive = RR_Torque
#num_assm.DR2_config.UF_fal_drive = RL_Torque

num_assm.CH_config.UF_fas_aero_drag_F = zero_func
num_assm.CH_config.UF_fas_aero_drag_T = zero_func
# =============================================================================
#                       Setting and Starting Simulation
# =============================================================================

sim = simulation('sim', num_model, 'dds')
sim.set_time_array(15, dt)

# Getting Equilibrium results as initial conditions to this simulation
# ====================================================================
sim.set_initial_states('results/equilibrium_v4.npz')

sim.solve()

sim.save_as_csv('results', 'constant_radius_v8', 'pos')
sim.save_as_npz('results', 'constant_radius_v8')

#=============================================================================
#                       Plotting Simulation Results
# =============================================================================

import matplotlib.pyplot as plt

sim.soln.pos_dataframe.plot(x='CH.rbs_chassis.x', y='CH.rbs_chassis.y', grid=True)

sim.soln.vel_dataframe.plot(x='time', y='CH.rbs_chassis.x', grid=True)

sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)
sim.soln.vel_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)
sim.soln.acc_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)

sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e0', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e1', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e2', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e3', grid=True)

plt.show()
