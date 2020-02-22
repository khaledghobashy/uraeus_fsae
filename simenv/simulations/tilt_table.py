import sys
import os

import numpy as np
import pandas as pd

from uraeus.nmbd.python import simulation
from uraeus.nmbd.python.engine.numerics.math_funcs import A

database_directory = os.path.abspath('../../')
sys.path.append(database_directory)

from uraeus_fsae.simenv.assemblies import asurt_FS17 as num_assm
from uraeus_fsae.simenv.assemblies.asurt_FS17 import num_model


dt = num_assm.dt
TR = 254

def terrain_state(x, y):
   
    t = num_model.topology.t
    amplitude = np.deg2rad(45)
    duration  = 15
    theta = amplitude*(1/duration)*t if t <=duration else amplitude

    print('Table Angle = %s'%np.rad2deg(theta))

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

num_assm.terrain_data.get_state = terrain_state


num_assm.AX1_config.UF_far_drive_T = FR_Torque
num_assm.AX1_config.UF_fal_drive_T = FL_Torque
num_assm.AX2_config.UF_far_drive_T = RR_Torque
num_assm.AX2_config.UF_fal_drive_T = RL_Torque

num_assm.ST1_config.UF_mcs_rack_act = steering_function

num_assm.AX1_config.UF_far_drive_F = lambda t: np.zeros((3,1), dtype=np.float64)
num_assm.AX1_config.UF_fal_drive_F = lambda t: np.zeros((3,1), dtype=np.float64)
num_assm.AX2_config.UF_far_drive_F = lambda t: np.zeros((3,1), dtype=np.float64)
num_assm.AX2_config.UF_fal_drive_F = lambda t: np.zeros((3,1), dtype=np.float64)

num_assm.CH_config.UF_fas_aero_drag_F = lambda t: np.zeros((3,1), dtype=np.float64)
num_assm.CH_config.UF_fas_aero_drag_T = lambda t: np.zeros((3,1), dtype=np.float64)

# =============================================================================
#                       Setting and Starting Simulation
# =============================================================================

# Getting Equilibrium results as initial conditions to this simulation
# ====================================================================
equlibrium_results = pd.read_csv('results/equilibrium_v1.csv', index_col=0)
q0 = equlibrium_results.iloc[-1][:-1][:,np.newaxis]

sim = simulation('sim', num_model, 'dds')

sim.soln.set_initial_states(q0, 0*q0)

sim.set_time_array(20, dt)
sim.solve()

sim.save_results('results', 'tilt_45d_20s_2')



# =============================================================================
#                       Plotting Simulation Results
# =============================================================================

import matplotlib.pyplot as plt

sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)
sim.soln.vel_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)
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
                            y=['AX1.rbr_hub.x', 'AX1.rbl_hub.x',
                               'AX2.rbr_hub.x', 'AX2.rbl_hub.x'], 
                            grid=True)

sim.soln.vel_dataframe.plot(x='time', 
                            y=['AX1.rbr_hub.x', 'AX1.rbl_hub.x',
                               'AX2.rbr_hub.x', 'AX2.rbl_hub.x'], 
                            grid=True)

sim.soln.pos_dataframe.plot(x='time', 
                            y=['AX1.rbr_hub.e0', 'AX1.rbr_hub.e1',
                               'AX1.rbr_hub.e2', 'AX1.rbr_hub.e3'], 
                            grid=True)

plt.show()


