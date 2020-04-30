import sys
import os

import numpy as np
import pandas as pd

from uraeus.nmbd.python import simulation
from uraeus.nmbd.python.engine.numerics.math_funcs import A

database_directory = os.path.abspath('../../')
sys.path.append(database_directory)

from uraeus_fsae.simenv.assemblies import asurt_FS17_v1 as num_assm

num_model = num_assm.num_model

dt = num_assm.dt
TR = 254

def terrain_state(x, y):
   
    t = num_model.topology.t
    amplitude = np.deg2rad(45)
    duration  = 15
    theta = amplitude*(1/duration)*t if t <=duration else amplitude

    #print('Table Angle = %s'%np.rad2deg(theta))

    local_normal = np.array([[0],[0],[1]])
    table_matrix = np.array([[1, 0, 0],
                             [0, np.cos(theta), -np.sin(theta)],
                             [0, np.sin(theta), np.cos(theta)]])

    table_normal = table_matrix @ local_normal

    pivot = -600
    horizontal_length = abs(y - pivot)
    hieght = horizontal_length * np.tan(theta)

    return [table_normal, hieght]



def FR_Torque(t):
    return 0

def FL_Torque(t):
    return 0

def RR_Torque(t):
    return 0

def RL_Torque(t):
    return 0

def steering_function(t):
    return 0

def zero_func(t):
    return np.zeros((3, 1), dtype=np.float64)


num_assm.terrain_data.get_state = terrain_state

num_assm.ST1_config.UF_mcs_rack_act = steering_function

num_assm.DR2_config.UF_far_drive = RR_Torque
num_assm.DR2_config.UF_fal_drive = RL_Torque

num_assm.CH_config.UF_fas_aero_drag_F = zero_func
num_assm.CH_config.UF_fas_aero_drag_T = zero_func

# =============================================================================
#                       Setting and Starting Simulation
# =============================================================================

sim = simulation('sim', num_model, 'dds')
sim.set_time_array(20, dt)

# Getting Equilibrium results as initial conditions to this simulation
# ====================================================================
sim.set_initial_states('results/equilibrium_v4.npz')

sim.solve()

sim.save_as_csv('results', 'tilt_table_45deg_20s_v1')
sim.save_as_npz('results', 'tilt_table_45deg_20s_v1')

# =============================================================================
#                       Plotting Simulation Results
# =============================================================================

import matplotlib.pyplot as plt

sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.x', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.y', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)

sim.soln.vel_dataframe.plot(x='time', y='CH.rbs_chassis.x', grid=True)
sim.soln.vel_dataframe.plot(x='time', y='CH.rbs_chassis.y', grid=True)
sim.soln.vel_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)

sim.soln.acc_dataframe.plot(x='time', y='CH.rbs_chassis.x', grid=True)
sim.soln.acc_dataframe.plot(x='time', y='CH.rbs_chassis.y', grid=True)
sim.soln.acc_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)

sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e0', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e1', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e2', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e3', grid=True)


sim.soln.pos_dataframe.plot(x='time', 
                            y=['AX1.rbr_hub.z', 'AX1.rbl_hub.z',
                               'AX2.rbr_hub.z', 'AX2.rbl_hub.z'], 
                            grid=True)


sim.soln.pos_dataframe.plot(x='time', 
                            y=['AX1.rbr_hub.e0', 'AX1.rbr_hub.e1',
                               'AX1.rbr_hub.e2', 'AX1.rbr_hub.e3'], 
                            grid=True)

plt.show()


