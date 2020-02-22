import sys
import os

import numpy as np
import pandas as pd

from uraeus.nmbd.python import simulation
from uraeus.nmbd.python.engine.numerics.math_funcs import A

database_directory = os.path.abspath('../../')
print('Database Directory: %s \n'%database_directory)
sys.path.append(database_directory)

from uraeus_fsae.simenv.assemblies import asurt_FS17 as num_assm
from uraeus_fsae.simenv.assemblies.asurt_FS17 import num_model

dt = num_assm.dt
TR = 254

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

num_assm.ST1_config.UF_mcs_rack_act = steering_function

num_assm.AX1_config.UF_far_drive_T = FR_Torque
num_assm.AX1_config.UF_fal_drive_T = FL_Torque
num_assm.AX2_config.UF_far_drive_T = RR_Torque
num_assm.AX2_config.UF_fal_drive_T = RL_Torque

num_assm.AX1_config.UF_far_drive_F = lambda t: np.zeros((3,1), dtype=np.float64)
num_assm.AX1_config.UF_fal_drive_F = lambda t: np.zeros((3,1), dtype=np.float64)
num_assm.AX2_config.UF_far_drive_F = lambda t: np.zeros((3,1), dtype=np.float64)
num_assm.AX2_config.UF_fal_drive_F = lambda t: np.zeros((3,1), dtype=np.float64)

num_assm.CH_config.UF_fas_aero_drag_F = lambda t: np.zeros((3,1), dtype=np.float64)
num_assm.CH_config.UF_fas_aero_drag_T = lambda t: np.zeros((3,1), dtype=np.float64)

print('INITIALIZED MODEL SUCCESSFULLY!')
print('===============================')
