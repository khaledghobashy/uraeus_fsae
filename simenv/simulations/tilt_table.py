import sys
import os

import numpy as np
import pandas as pd

from uraeus.nmbd.python import simulation
from uraeus.nmbd.python.engine.numerics.math_funcs import A

database_directory = r'C:\khaledghobashy\github'
sys.path.append(database_directory)

from imut_car_database.simulations.ST500.assemblies import num_assm as num_assm
from imut_car_database.simulations.ST500.assemblies.num_assm import num_model


def normalize(v):
    normalized = v/np.linalg.norm(v)
    return normalized

def terrain_state(x, y):
   
   t = num_model.topology.t
   amplitude = np.deg2rad(30)
   duration  = 15
   theta = amplitude*(1/duration)*t if t <=duration else amplitude

   n = np.array([[0],[0],[1]])
   v = normalize(np.cos(theta) * n)

   pivot = -1200
   l = abs(pivot - y)
   hieght = l * np.sin(theta)

   return [v, hieght]

num_assm.terrain_data.get_state = terrain_state


dt = num_assm.dt
TR = 546

def FR_Torque():
    return 0

def FL_Torque():
    return 0

def RR_Torque():
    return 0

def RL_Torque():
    return 0


def steering_function(t):
    return 0

num_assm.ST1_config.UF_mcs_steer_act = steering_function

num_assm.AX1_config.UF_far_drive_T = FR_Torque
num_assm.AX1_config.UF_fal_drive_T = FL_Torque
num_assm.AX2_config.UF_far_drive_T = RR_Torque
num_assm.AX2_config.UF_fal_drive_T = RL_Torque

num_assm.AX1_config.UF_far_drive_F = lambda : np.zeros((3,1), dtype=np.float64)
num_assm.AX1_config.UF_fal_drive_F = lambda : np.zeros((3,1), dtype=np.float64)
num_assm.AX2_config.UF_far_drive_F = lambda : np.zeros((3,1), dtype=np.float64)
num_assm.AX2_config.UF_fal_drive_F = lambda : np.zeros((3,1), dtype=np.float64)

num_assm.CH_config.UF_fas_aero_drag_F = lambda : np.zeros((3,1), dtype=np.float64)
num_assm.CH_config.UF_fas_aero_drag_T = lambda : np.zeros((3,1), dtype=np.float64)



equlibrium_results = pd.read_csv('results/equilibrium.csv', index_col=0)
q0 = np.array(equlibrium_results.iloc[-1][:-1]).reshape((num_model.topology.n, 1))

sim = simulation('sim', num_model, 'dds')

sim.soln.set_initial_states(q0, 0*q0)

sim.set_time_array(20, dt)
sim.solve()

sim.save_results('results', 'tilt_30d_20s_1')



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


