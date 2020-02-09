import sys
import os

import numpy as np
import pandas as pd

from uraeus.nmbd.python import multibody_system, assembly
from uraeus.nmbd.python.engine.numerics.math_funcs import A
from uraeus.nmbd.python.engine.modules.vehicle_dynamics.tire_models import \
brush_model

from ..components.terrain_data import terrain

from ..subsystems.front_axle import AX1_config
from ..subsystems.rear_axle import AX2_config
from ..subsystems.steering import ST1_config
from ..subsystems.chassis import CH_config

assm = assembly(r'assemblies\rolling_chassis.json', r'C:\khaledghobashy\github\imut_car_database\numenv\python\templates')
num_model = multibody_system(assm)

num_model.Subsystems.AX1.config = AX1_config
num_model.Subsystems.AX2.config = AX2_config
num_model.Subsystems.ST1.config = ST1_config
num_model.Subsystems.CH.config  = CH_config


terrain_data = terrain()
tire_model = brush_model

dt = 1e-2


front_right_tire = tire_model()

def fr_tire_force():
    print('FR_Tire : \n=========')
    t  = num_model.topology.t
    R  = num_model.Subsystems.AX1.R_rbr_hub
    P  = num_model.Subsystems.AX1.P_rbr_hub
    Rd = num_model.Subsystems.AX1.Rd_rbr_hub
    Pd = num_model.Subsystems.AX1.Pd_rbr_hub

    wheel_states = [R, P, Rd, Pd]
    terrain_state = terrain_data.get_state(R[0,0], R[1,0])
    
    drive_torque = np.linalg.norm(AX1_config.UF_far_drive_T())
    if drive_torque == 0:
        front_right_tire.driven = 0

    front_right_tire.Eval_Forces(t, dt, wheel_states, drive_torque, terrain_state)
    force = front_right_tire.F
    torque_ratio = drive_torque / front_right_tire.My
    
    print('Drive Torque = %s'%drive_torque)
    print('Wheel Torque = %s'%torque_ratio)
    print('\n\n')
    return force


AX1_config.UF_far_tire_F = fr_tire_force
AX1_config.UF_far_tire_T = lambda : front_right_tire.M



front_left_tire = tire_model()

def fl_tire_force():
    print('FL_Tire : \n=========')
    t  = num_model.topology.t
    R  = num_model.Subsystems.AX1.R_rbl_hub
    P  = num_model.Subsystems.AX1.P_rbl_hub
    Rd = num_model.Subsystems.AX1.Rd_rbl_hub
    Pd = num_model.Subsystems.AX1.Pd_rbl_hub

    wheel_states = [R, P, Rd, Pd]
    terrain_state = terrain_data.get_state(R[0,0], R[1,0])

    drive_torque = np.linalg.norm(AX1_config.UF_fal_drive_T())
    if drive_torque == 0:
        front_left_tire.driven = 0

    front_left_tire.Eval_Forces(t, dt, wheel_states, drive_torque, terrain_state)
    force = front_left_tire.F
    torque_ratio = drive_torque / front_left_tire.My
    
    print('Drive Torque = %s'%drive_torque)
    print('Wheel Torque = %s'%torque_ratio)
    print('\n\n')
    return force


AX1_config.UF_fal_tire_F = fl_tire_force
AX1_config.UF_fal_tire_T = lambda : front_left_tire.M



rear_right_tire = tire_model()

def rr_tire_force():
    print('RR_Tire : \n=========')
    t  = num_model.topology.t
    R  = num_model.Subsystems.AX2.R_rbr_hub
    P  = num_model.Subsystems.AX2.P_rbr_hub
    Rd = num_model.Subsystems.AX2.Rd_rbr_hub
    Pd = num_model.Subsystems.AX2.Pd_rbr_hub

    wheel_states = [R, P, Rd, Pd]
    terrain_state = terrain_data.get_state(R[0,0], R[1,0])

    drive_torque = np.linalg.norm(AX2_config.UF_far_drive_T())
    if drive_torque == 0:
        rear_right_tire.driven = 0

    rear_right_tire.Eval_Forces(t, dt, wheel_states, drive_torque, terrain_state)
    force = rear_right_tire.F
    torque_ratio = drive_torque / rear_right_tire.My
    
    print('Drive Torque = %s'%drive_torque)
    print('Wheel Torque = %s'%torque_ratio)
    print('\n\n')
    return force

AX2_config.UF_far_tire_F = rr_tire_force
AX2_config.UF_far_tire_T = lambda : rear_right_tire.M


rear_left_tire = tire_model()

def rl_tire_force():
    print('RL_Tire : \n=========')
    t  = num_model.topology.t
    R  = num_model.Subsystems.AX2.R_rbl_hub
    P  = num_model.Subsystems.AX2.P_rbl_hub
    Rd = num_model.Subsystems.AX2.Rd_rbl_hub
    Pd = num_model.Subsystems.AX2.Pd_rbl_hub

    wheel_states = [R, P, Rd, Pd]
    terrain_state = terrain_data.get_state(R[0,0], R[1,0])

    drive_torque = np.linalg.norm(AX2_config.UF_fal_drive_T())
    if drive_torque == 0:
        rear_left_tire.driven = 0

    rear_left_tire.Eval_Forces(t, dt, wheel_states, drive_torque, terrain_state)
    force = rear_left_tire.F
    torque_ratio = drive_torque / rear_left_tire.My
    
    print('Drive Torque = %s'%drive_torque)
    print('Wheel Torque = %s'%torque_ratio)
    print('\n\n')
    return force


AX2_config.UF_fal_tire_F = rl_tire_force
AX2_config.UF_fal_tire_T = lambda : rear_left_tire.M


