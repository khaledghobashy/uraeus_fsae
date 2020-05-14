import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from uraeus.nmbd.python import simulation
from uraeus.nmbd.python.engine.numerics.math_funcs import A, B
from controllers import speed_controller, stanley_controller

database_directory = os.path.abspath('../../')
sys.path.append(database_directory)

from uraeus_fsae.simenv.assemblies import asurt_FS17_v1 as num_assm

num_model = num_assm.num_model

dt = num_assm.dt
TR = 254

#path_data = np.zeros((1000, 3))
#path_data[:, 0] = (10*1e3 / 2*np.pi) * np.linspace(0, -2*np.pi, 1000)
#path_data[:, 1] = (5*1e3 * np.sin(np.linspace(0, -2*np.pi, 1000)))
#path_data[:, 2] = np.gradient(path_data[:, 1] / ((25*1e3 / 2*np.pi)), np.linspace(0, -2*np.pi, 1000))

path_data = np.zeros((1000, 3))
path_data[:, 0] = 100*1e3 * np.linspace(0, -2*np.pi, 1000)
path_data[:, 1] = 10*1e3 * np.ones((1000,))
path_data[:, 2] = np.gradient(path_data[:, 1] / (100*1e3), path_data[:, 1])

plt.figure(figsize=(10, 5))
plt.plot(path_data[:,0], path_data[:,1]*1e-3)
plt.grid()

plt.figure(figsize=(10, 5))
plt.plot(path_data[:,0], path_data[:,2])
plt.grid()

plt.figure(figsize=(10, 5))
plt.plot(path_data[:,0], np.arctan(path_data[:,2]))
plt.grid()

plt.show()


controller = speed_controller(30, dt)
lateral_controller = stanley_controller(path_data)


def terrain_state(x, y):
    local_normal = np.array([[0],[0],[1]], dtype=np.float64)
    hieght = 0
    return [local_normal, hieght]


def torque_function(t):
    P_ch = num_model.Subsystems.CH.P_rbs_chassis
    Rd = num_model.Subsystems.CH.Rd_rbs_chassis
    factor = controller.get_torque_factor(P_ch, Rd)
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
    vel = (A(P_ch).T@(Rd_ch + B(P_ch, rbar_ax1)@Pd_ch))[0,0]

    delta = lateral_controller.get_steer_factor(r_ax1, P_ch, vel)

    travel = delta * 15
    print('Travel = %s'%travel)
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

sim.save_as_csv('results', 'sine_path_v8', 'pos')
sim.save_as_npz('results', 'sine_path_v8')

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
