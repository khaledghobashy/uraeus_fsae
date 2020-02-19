import sys
import os

import numpy as np
import pandas as pd

from uraeus.nmbd.python import simulation
from uraeus.nmbd.python.engine.numerics.math_funcs import A

database_directory = os.path.abspath('../../')
print('Database Directory: %s'%database_directory)
sys.path.append(database_directory)

from uraeus_fsae.simenv.assemblies import asurt_FS16 as num_assm
from uraeus_fsae.simenv.assemblies.asurt_FS16 import num_model


dt = num_assm.dt
TR = 254

def terrain_state(x, y):
    local_normal = np.array([[0],[0],[1]])
    hieght = 0
    return [local_normal, hieght]



def drive_torque(P_hub, factor):
    local_torque = np.array([[0],
                             [-factor*(70*9.81)*1e6*TR],
                             [0]])
    global_torque = A(P_hub) @ local_torque
    return global_torque


def FR_Torque():
    return 0

def FL_Torque():
    return 0

def RR_Torque():
    factor = 1 if num_model.topology.t <= 3 else 0.5
    return drive_torque(num_model.Subsystems.AX2.P_rbr_upright, factor)

def RL_Torque():
    factor = 1 if num_model.topology.t <= 3 else 0.5
    return drive_torque(num_model.Subsystems.AX2.P_rbl_upright, factor)


def steering_function(t):
    
    t_dur = 2
    t_str = 4
    t_end = t_str + t_dur
    
    rotation  = 0
    amplitude = 22
    #if t >= t_str and t <= t_end:
        #rotation = amplitude*np.sin((2*np.pi/t_dur)*(t-t_str))
    
    return amplitude

num_assm.terrain_data.get_state = terrain_state


num_assm.AX1_config.UF_far_drive_T = FR_Torque
num_assm.AX1_config.UF_fal_drive_T = FL_Torque
num_assm.AX2_config.UF_far_drive_T = RR_Torque
num_assm.AX2_config.UF_fal_drive_T = RL_Torque

num_assm.ST1_config.UF_mcs_rack_act = steering_function

num_assm.AX1_config.UF_far_drive_F = lambda : np.zeros((3,1), dtype=np.float64)
num_assm.AX1_config.UF_fal_drive_F = lambda : np.zeros((3,1), dtype=np.float64)
num_assm.AX2_config.UF_far_drive_F = lambda : np.zeros((3,1), dtype=np.float64)
num_assm.AX2_config.UF_fal_drive_F = lambda : np.zeros((3,1), dtype=np.float64)

num_assm.CH_config.UF_fas_aero_drag_F = lambda : np.zeros((3,1), dtype=np.float64)
num_assm.CH_config.UF_fas_aero_drag_T = lambda : np.zeros((3,1), dtype=np.float64)



equlibrium_results = pd.read_csv('results/equilibrium_v1.csv', index_col=0)
q0 = equlibrium_results.iloc[-1][:-1][:,np.newaxis]

sim = simulation('sim', num_model, 'dds')

sim.soln.set_initial_states(q0, 0*q0)

sim.set_time_array(8, dt)
sim.solve()

sim.save_results('results', 'constant_steer_5')



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


