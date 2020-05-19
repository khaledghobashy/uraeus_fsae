import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

from uraeus.nmbd.python import simulation
from uraeus.nmbd.python.engine.numerics.math_funcs import A, B
from controllers import speed_controller, stanley_controller

database_directory = os.path.abspath('../../')
sys.path.append(database_directory)

from uraeus_fsae.simenv.assemblies import asurt_FS17_v1 as num_assm

num_model = num_assm.num_model

dt = num_assm.dt
TR = 254

def iso_dlc(vehicle_width, x_offset=0):
    a = 1.1 * vehicle_width + 0.25
    b = 1.2 * vehicle_width + 0.25
    
    y_peak = (a/2 + b/2 + 1)
    
    xs = np.arange(0, x_offset, 1)
    ys = 0 * np.ones((len(xs),))
    
    x1 = np.arange(0, 10, 1) + x_offset
    y1 = 0 * np.ones((len(x1),))
    
    x2 = np.arange(55, 65, 1) + x_offset
    y2 = y_peak * np.ones((len(x2),))
    
    x3 = np.arange(100, 125, 1) + x_offset
    y3 = 0 * np.ones((len(x3),))
    
    xe = np.arange(x3[-1]+1, x3[-1] + x_offset, 1)
    ye = 0 * np.ones((len(xe),))
    
    x_data = np.concatenate([xs, x1, x2, x3, xe])
    y_data = np.concatenate([ys, y1, y2, y3, ye])
    
    tck = interpolate.splrep(x_data, y_data, k=5)#, s=1e-4)
    
    x_new = np.arange(min(x_data), max(x_data), 1)
    y_new = interpolate.splev(x_new, tck)

    return x_new, y_new

def plot_iso_dlc(vehicle_width, x_offset=0):
    a  = 1.1 * vehicle_width + 0.25
    b1 = 1.2 * vehicle_width + 0.25
    b2 = 1.3 * vehicle_width + 0.25
    
    x_polyons  = np.array([ 0, 7.5, 15, 45, 57.5, 70, 95, 110, 125])
    y_polyons1 = np.array([0, 0, 0, a+1, a+1, a+1, 0, 0, 0])
    y_polyons2 = np.array([a, a, a, a+1+b1, a+1+b1, a+1+b1, b2, b2, b2])
    
    x_data, y_data = iso_dlc(vehicle_width, x_offset)
    
    plt.figure(figsize=(15, 5))
    plt.plot(x_data[x_offset:125+x_offset]-x_offset, y_data[x_offset:125+x_offset]+(a/2))
    plt.plot(x_polyons, y_polyons1, 'o')
    plt.plot(x_polyons, y_polyons2, 'o')
    plt.grid()
    plt.show()


x_data, y_data = iso_dlc(1.2, 80)

path_data = np.zeros((len(x_data), 2))
path_data[:, 0] = -1e3 * x_data
path_data[:, 1] =  1e3 * y_data

plot_iso_dlc(1.2, 100)

logitudinal_controller = speed_controller(65, dt)
lateral_controller = stanley_controller(path_data, 25)


def terrain_state(x, y):
    local_normal = np.array([[0],[0],[1]], dtype=np.float64)
    hieght = 0
    return [local_normal, hieght]


def torque_function(t):
    P_ch = num_model.Subsystems.CH.P_rbs_chassis
    Rd = num_model.Subsystems.CH.Rd_rbs_chassis
    factor = logitudinal_controller.get_torque_factor(P_ch, Rd)# if t<5 else 0
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

    delta = lateral_controller.get_steer_factor(r_ax1, P_ch, vel)

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

sim.save_as_csv('results', 'dlc_auto_v10', 'pos')
sim.save_as_npz('results', 'dlc_auto_v10')

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


plt.figure(figsize=(10, 5))
plt.plot(range(len(lateral_controller.error_array)), lateral_controller.error_array)
plt.grid()

plt.show()
