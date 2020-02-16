import sys
import os

import numpy as np
import pandas as pd

from uraeus.nmbd.python import assembly, multibody_system
from uraeus.nmbd.python.engine.numerics.math_funcs import A
from uraeus.nmbd.python.engine.modules.vehicle_dynamics.tire_models import \
brush_model

from ..components.terrain_data import terrain

from ..subsystems.asurt_fs16.front_axle import AX1_config
from ..subsystems.asurt_fs16.rear_axle import AX2_config
from ..subsystems.asurt_fs16.steering import ST1_config
from ..subsystems.asurt_fs16.chassis import CH_config

templates_dir = os.path.abspath('../numenv/python/templates')
print('templates directory : %s'%templates_dir)

assm = assembly('assemblies/rolling_chassis_v1.json', templates_dir)
num_model = multibody_system(assm)

num_model.Subsystems.AX1.config = AX1_config
num_model.Subsystems.AX2.config = AX2_config
num_model.Subsystems.ST1.config = ST1_config
num_model.Subsystems.CH.config  = CH_config


terrain_data = terrain()
tire_model = brush_model

dt = 1e-2
TR = 254


front_right_tire = tire_model()
front_right_tire.mu = 1.3

front_right_tire.nominal_radius = TR
front_right_tire.kz = 250*1e6
front_right_tire.cz = 0.3*1e6

front_right_tire.a  = 100
front_right_tire.Iyy = 1*1e9

front_right_tire.cp   = 1500*1e3
front_right_tire.C_Fk = 400*1e9
front_right_tire.C_Fa = 400*1e9

def get_contact_point(R, P):
    u = np.array([[0], [0], [-TR]])
    point = R + (A(P) @ u)
    x, y, z = point.flat[:]
    print('Contact Point = %s'%((x,y),))
    return x, y

def fr_tire_force():
    print('FR_Tire : \n=========')
    t  = num_model.topology.t
    R  = num_model.Subsystems.AX1.R_rbr_hub
    P  = num_model.Subsystems.AX1.P_rbr_hub
    Rd = num_model.Subsystems.AX1.Rd_rbr_hub
    Pd = num_model.Subsystems.AX1.Pd_rbr_hub

    wheel_states = [R, P, Rd, Pd]
    x, y = get_contact_point(R, num_model.Subsystems.AX1.P_rbr_upright)
    terrain_state = terrain_data.get_state(x, y)
    
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
front_left_tire.mu = 1.3

front_left_tire.nominal_radius = TR
front_left_tire.kz = 250*1e6
front_left_tire.cz = 0.3*1e6

front_left_tire.a  = 100
front_left_tire.Iyy = 1*1e9

front_left_tire.cp   = 1500*1e3
front_left_tire.C_Fk = 400*1e9
front_left_tire.C_Fa = 400*1e9

def fl_tire_force():
    print('FL_Tire : \n=========')
    t  = num_model.topology.t
    R  = num_model.Subsystems.AX1.R_rbl_hub
    P  = num_model.Subsystems.AX1.P_rbl_hub
    Rd = num_model.Subsystems.AX1.Rd_rbl_hub
    Pd = num_model.Subsystems.AX1.Pd_rbl_hub

    wheel_states = [R, P, Rd, Pd]
    x, y = get_contact_point(R, num_model.Subsystems.AX1.P_rbl_upright)
    terrain_state = terrain_data.get_state(x, y)

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
rear_right_tire.mu = 1.3

rear_right_tire.nominal_radius = TR
rear_right_tire.kz = 250*1e6
rear_right_tire.cz = 0.3*1e6

rear_right_tire.a  = 100
rear_right_tire.Iyy = 1*1e9

rear_right_tire.cp   = 1500*1e3
rear_right_tire.C_Fk = 400*1e9
rear_right_tire.C_Fa = 400*1e9

def rr_tire_force():
    print('RR_Tire : \n=========')
    t  = num_model.topology.t
    R  = num_model.Subsystems.AX2.R_rbr_hub
    P  = num_model.Subsystems.AX2.P_rbr_hub
    Rd = num_model.Subsystems.AX2.Rd_rbr_hub
    Pd = num_model.Subsystems.AX2.Pd_rbr_hub

    wheel_states = [R, P, Rd, Pd]
    x, y = get_contact_point(R, num_model.Subsystems.AX2.P_rbr_upright)
    terrain_state = terrain_data.get_state(x, y)

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
rear_left_tire.mu = 1.3

rear_left_tire.nominal_radius = TR
rear_left_tire.kz = 250*1e6
rear_left_tire.cz = 0.3*1e6

rear_left_tire.a  = 100
rear_left_tire.Iyy = 1*1e9

rear_left_tire.cp   = 1500*1e3
rear_left_tire.C_Fk = 400*1e9
rear_left_tire.C_Fa = 400*1e9

def rl_tire_force():
    print('RL_Tire : \n=========')
    t  = num_model.topology.t
    R  = num_model.Subsystems.AX2.R_rbl_hub
    P  = num_model.Subsystems.AX2.P_rbl_hub
    Rd = num_model.Subsystems.AX2.Rd_rbl_hub
    Pd = num_model.Subsystems.AX2.Pd_rbl_hub

    wheel_states = [R, P, Rd, Pd]
    x, y = get_contact_point(R, num_model.Subsystems.AX2.P_rbl_upright)
    terrain_state = terrain_data.get_state(x, y)

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


