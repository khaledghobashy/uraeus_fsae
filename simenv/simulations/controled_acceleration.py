import sys
import os

import numpy as np
import pandas as pd

from uraeus.nmbd.python import simulation
from uraeus.nmbd.python.engine.numerics.math_funcs import A
from controllers import speed_controller

database_directory = os.path.abspath('../../')
sys.path.append(database_directory)

from uraeus_fsae.simenv.assemblies import asurt_FS17_v1 as num_assm

num_model = num_assm.num_model

dt = num_assm.dt
TR = 254


controller = speed_controller(60, dt)

def torque_function(t):
    P_ch = num_model.Subsystems.CH.P_rbs_chassis
    Rd = num_model.Subsystems.CH.Rd_rbs_chassis
    if t >= 3:
        controller.desired_speed = 0
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
    return 0

def zero_func(t):
    return np.zeros((3, 1), dtype=np.float64)

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
sim.set_time_array(6, dt)

# Getting Equilibrium results as initial conditions to this simulation
# ====================================================================
sim.set_initial_states('results/equilibrium_v4.npz')

sim.solve()

sim.save_as_csv('results', 'acceleration_v10')
sim.save_as_npz('results', 'acceleration_v10')

#=============================================================================
#                       Plotting Simulation Results
# =============================================================================

import matplotlib.pyplot as plt

sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.x', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.y', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)

sim.soln.vel_dataframe.plot(x='time', y='CH.rbs_chassis.x', grid=True)
sim.soln.acc_dataframe.plot(x='time', y='CH.rbs_chassis.x', grid=True)

sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e0', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e1', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e2', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e3', grid=True)

plt.show()


