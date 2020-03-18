import sys
import os

import numpy as np

from uraeus.nmbd.python import assembly, multibody_system
from uraeus.nmbd.python.engine.numerics.math_funcs import A
from uraeus.nmbd.python.engine.modules.vehicle_dynamics.tire_models import \
brush_model

from ..components.terrain_data import terrain

from ..subsystems.asurt_fs17.front_axle import AX1_config
from ..subsystems.asurt_fs17.rear_axle import AX2_config
from ..subsystems.asurt_fs17.steering import ST1_config
from ..subsystems.asurt_fs17.chassis import CH_config
from ..subsystems.asurt_fs17.rear_drive_shafts_v1 import DR2_config


templates_dir = os.path.abspath('../numenv/python/templates')
print('templates directory : %s'%templates_dir)

assm = assembly('../symenv/assemblies/configurations/rolling_chassis_v1.json', templates_dir)
num_model = multibody_system(assm)

num_model.Subsystems.AX1.config = AX1_config
num_model.Subsystems.AX2.config = AX2_config
num_model.Subsystems.DR2.config = DR2_config
num_model.Subsystems.ST1.config = ST1_config
num_model.Subsystems.CH.config  = CH_config


terrain_data = terrain()
tire_model = brush_model

dt = 5e-3
TR = 254

LOG_STATES = True

def print_tire_states(tire_instance, drive_torque, name):
    torque_ratio = drive_torque / tire_instance.My
    print('%s : \n========='%name)
    print('Omega = %s'%tire_instance.Omega)
    print('V_WC = %s'%tire_instance.V_wc_SAE.T)
    print('V_x  = %s'%tire_instance.V_x)
    print('V_C  = %s'%tire_instance.V_C)
    print('V_sx = %s'%tire_instance.V_sx)
    print('V_sy = %s'%tire_instance.V_sy)
    print('Drive Torque = %s'%drive_torque)
    print('Torque Ratio = %s'%torque_ratio)
    print('\n\n')

def get_contact_point(R, P):
    u = np.array([[0], [0], [-TR]])
    point = R + (A(P) @ u)
    x, y, z = point.flat[:]
    #print('Contact Point = %s'%((x,y),))
    return x, y

# ================================================================================= #
# ================================================================================= #

# Tire properties
# ---------------
mu = 1.3
kz = 250 * 1e6
cz = 5 * 1e5

a = 100
cp = 1500 * 1e3
C_Fk = 400 * 1e9
C_Fa = 400 * 1e9

# ================================================================================= #
# ================================================================================= #

# Front Right Tire Instance
# =========================

front_right_tire = tire_model()

front_right_tire.mu = mu
front_right_tire.kz = kz
front_right_tire.cz = cz
front_right_tire.a  = a
front_right_tire.cp  = cp
front_right_tire.C_Fk = C_Fk
front_right_tire.C_Fa = C_Fa
front_right_tire.nominal_radius = TR

def fr_tire_force(t):
    R  = num_model.Subsystems.AX1.R_rbr_hub
    P  = num_model.Subsystems.AX1.P_rbr_hub
    Rd = num_model.Subsystems.AX1.Rd_rbr_hub
    Pd = num_model.Subsystems.AX1.Pd_rbr_hub

    wheel_states = [R, P, Rd, Pd]
    x, y = get_contact_point(R, num_model.Subsystems.AX1.P_rbr_upright)
    terrain_state = terrain_data.get_state(x, y)
    
    drive_torque = AX1_config.UF_far_drive(t)
    if drive_torque == 0:
        front_right_tire.driven = 0

    front_right_tire.Eval_Forces(t, dt, wheel_states, drive_torque, terrain_state)
    force = front_right_tire.F

    if LOG_STATES:
        print_tire_states(front_right_tire, drive_torque, 'FR_Tire')
    
    return force #np.array([[0], [0], [force[2,0]]])


AX1_config.UF_far_tire_F = fr_tire_force
AX1_config.UF_far_tire_T = lambda t: front_right_tire.M

# ================================================================================= #
# ================================================================================= #

# Front Left Tire Instance
# ========================

front_left_tire = tire_model()

front_left_tire.mu = mu
front_left_tire.kz = kz
front_left_tire.cz = cz
front_left_tire.a  = a
front_left_tire.cp = cp
front_left_tire.C_Fk = C_Fk
front_left_tire.C_Fa = C_Fa
front_left_tire.nominal_radius = TR

def fl_tire_force(t):
    R  = num_model.Subsystems.AX1.R_rbl_hub
    P  = num_model.Subsystems.AX1.P_rbl_hub
    Rd = num_model.Subsystems.AX1.Rd_rbl_hub
    Pd = num_model.Subsystems.AX1.Pd_rbl_hub

    wheel_states = [R, P, Rd, Pd]
    x, y = get_contact_point(R, num_model.Subsystems.AX1.P_rbl_upright)
    terrain_state = terrain_data.get_state(x, y)

    drive_torque = AX1_config.UF_fal_drive(t)
    if drive_torque == 0:
        front_left_tire.driven = 0

    front_left_tire.Eval_Forces(t, dt, wheel_states, drive_torque, terrain_state)
    force = front_left_tire.F

    if LOG_STATES:
        print_tire_states(front_left_tire, drive_torque, 'FL_Tire')
    
    return force #np.array([[0], [0], [force[2,0]]])


AX1_config.UF_fal_tire_F = fl_tire_force
AX1_config.UF_fal_tire_T = lambda t: front_left_tire.M

# ================================================================================= #
# ================================================================================= #

# Rear Right Tire Instance
# ========================

rear_right_tire = tire_model()

rear_right_tire.mu = mu
rear_right_tire.kz = kz
rear_right_tire.cz = cz
rear_right_tire.a  = a
rear_right_tire.cp = cp
rear_right_tire.C_Fk = C_Fk
rear_right_tire.C_Fa = C_Fa
rear_right_tire.nominal_radius = TR

def rr_tire_force(t):
    R  = num_model.Subsystems.AX2.R_rbr_hub
    P  = num_model.Subsystems.AX2.P_rbr_hub
    Rd = num_model.Subsystems.AX2.Rd_rbr_hub
    Pd = num_model.Subsystems.AX2.Pd_rbr_hub

    wheel_states = [R, P, Rd, Pd]
    x, y = get_contact_point(R, num_model.Subsystems.AX2.P_rbr_upright)
    terrain_state = terrain_data.get_state(x, y)

    drive_torque = AX2_config.UF_far_drive(t)
    if drive_torque == 0:
        rear_right_tire.driven = 0

    rear_right_tire.Eval_Forces(t, dt, wheel_states, drive_torque, terrain_state)
    force = rear_right_tire.F

    if LOG_STATES:
        print_tire_states(rear_right_tire, drive_torque, 'RR_Tire')
    
    return force #np.array([[0], [0], [force[2,0]]])

AX2_config.UF_far_tire_F = rr_tire_force
AX2_config.UF_far_tire_T = lambda t: rear_right_tire.M

# ================================================================================= #
# ================================================================================= #

rear_left_tire = tire_model()

rear_left_tire.mu = mu
rear_left_tire.kz = kz
rear_left_tire.cz = cz
rear_left_tire.a  = a
rear_left_tire.cp = cp
rear_left_tire.C_Fk = C_Fk
rear_left_tire.C_Fa = C_Fa
rear_left_tire.nominal_radius = TR

def rl_tire_force(t):
    R  = num_model.Subsystems.AX2.R_rbl_hub
    P  = num_model.Subsystems.AX2.P_rbl_hub
    Rd = num_model.Subsystems.AX2.Rd_rbl_hub
    Pd = num_model.Subsystems.AX2.Pd_rbl_hub

    wheel_states = [R, P, Rd, Pd]
    x, y = get_contact_point(R, num_model.Subsystems.AX2.P_rbl_upright)
    terrain_state = terrain_data.get_state(x, y)

    drive_torque = AX2_config.UF_fal_drive(t)
    if drive_torque == 0:
        rear_left_tire.driven = 0

    rear_left_tire.Eval_Forces(t, dt, wheel_states, drive_torque, terrain_state)
    force = rear_left_tire.F

    if LOG_STATES:
        print_tire_states(rear_left_tire, drive_torque, 'RL_Tire')
    
    return force #np.array([[0], [0], [force[2,0]]])


AX2_config.UF_fal_tire_F = rl_tire_force
AX2_config.UF_fal_tire_T = lambda t: rear_left_tire.M

