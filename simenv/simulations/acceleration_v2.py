import sys
import os

import numpy as np
import pandas as pd

from uraeus.nmbd.python import simulation
from uraeus.nmbd.python.engine.numerics.math_funcs import A

database_directory = os.path.abspath('../../')
sys.path.append(database_directory)

from uraeus_fsae.simenv.assemblies import asurt_FS17_v2 as num_assm

num_model = num_assm.num_model

dt = num_assm.dt
TR = 254

def FR_Torque(t):
    return 0

def FL_Torque(t):
    return 0

def RR_Torque(t):
    factor = 1 if t <= 4 else 0
    torque = -factor*(70*9.81)*1e6*TR
    return torque

def RL_Torque(t):
    factor = 1 if t <= 4 else 0
    torque = -factor*(70*9.81)*1e6*TR
    return torque

def steering_function(t):
    return 0

def zero_func(t):
    return np.zeros((3, 1), dtype=np.float64)

num_assm.ST1_config.UF_mcs_rack_act = steering_function

#num_assm.AX1_config.UF_far_drive = FR_Torque
#num_assm.AX1_config.UF_fal_drive = FL_Torque
#num_assm.AX2_config.UF_far_drive = RR_Torque
#num_assm.AX2_config.UF_fal_drive = RL_Torque

num_assm.DR2_config.UF_far_drive = RR_Torque
num_assm.DR2_config.UF_fal_drive = RL_Torque

num_assm.CH_config.UF_fas_aero_drag_F = zero_func
num_assm.CH_config.UF_fas_aero_drag_T = zero_func

# =============================================================================
#                       Setting and Starting Simulation
# =============================================================================

sim = simulation('sim', num_model, 'dds')
sim.set_time_array(5, dt)

# Getting Equilibrium results as initial conditions to this simulation
# ====================================================================
sim.set_initial_states('results/equilibrium_v4.npz')

sim.solve()

sim.save_as_csv('results', 'acceleration_v4')
sim.save_as_npz('results', 'acceleration_v4')

#=============================================================================
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


