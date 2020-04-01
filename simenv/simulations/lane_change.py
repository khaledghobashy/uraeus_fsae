import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from uraeus.nmbd.python import simulation
from uraeus.nmbd.python.engine.numerics.math_funcs import A

database_directory = os.path.abspath('../../')
sys.path.append(database_directory)

from uraeus_fsae.simenv.assemblies import asurt_FS17_v1 as num_assm

num_model = num_assm.num_model

dt = num_assm.dt
TR = 254

def terrain_state(x, y):
    local_normal = np.array([[0],[0],[1]], dtype=np.float64)
    hieght = 0
    return [local_normal, hieght]


def FR_Torque(t):
    return 0

def FL_Torque(t):
    return 0

def RR_Torque(t):
    factor = 1 if t <= 3 else 0.2
    torque = -factor*(70*9.81)*1e6*TR
    return torque

def RL_Torque(t):
    factor = 1 if t <= 3 else 0.2
    torque = -factor*(70*9.81)*1e6*TR
    return torque

def steering_function(t):
    
    t_dur = 1
    t_str1 = 4
    t_end1 = t_str1 + t_dur

    t_str2 = t_end1 + 1.5
    t_end2 = t_str2 + t_dur
    
    travel = 0
    amplitude = 12
    if t >= t_str1 and t <= t_end1:
        travel = amplitude*np.sin((2*np.pi/t_dur)*(t-t_str1))
    
    if t >= t_str2 and t <= t_end2:
        travel = -amplitude*np.sin((2*np.pi/t_dur)*(t-t_str2))
        
    return travel


def zero_func(t):
    return np.zeros((3,1), dtype=np.float64)


num_assm.terrain_data.get_state = terrain_state

num_assm.ST1_config.UF_mcs_rack_act = steering_function

num_assm.DR2_config.UF_far_drive = RR_Torque
num_assm.DR2_config.UF_fal_drive = RL_Torque

num_assm.CH_config.UF_fas_aero_drag_F = zero_func
num_assm.CH_config.UF_fas_aero_drag_T = zero_func



sim = simulation('sim', num_model, 'dds')
sim.set_time_array(9, dt)

plt.plot(sim.soln.time_array, [steering_function(i) for i in sim.soln.time_array])
plt.show()

sim.set_initial_states('results/equilibrium_v4.npz')

sim.solve()

sim.save_as_csv('results', 'lanechange_v4', 'pos')
sim.save_as_npz('results', 'lanechange_v4')


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
