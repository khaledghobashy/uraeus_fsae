import sys
import os

import numpy as np
import pandas as pd
from uraeus.nmbd.python import configuration

from uraeus_fsae.numenv.python.templates import dwb_bellcrank_push
#from ..components.struts_data import stiffness_func, damping_func

json_file = os.path.abspath('../symenv/templates/double_wishbone_bellcrank/data/dwb_bellcrank_push_cfg.json')

AX1_config = configuration('front_axle')
AX1_config.construct_from_json(json_file)

# =============================================================================
#                                   Numerical Data
# =============================================================================

# Tire Radius
TR = 254

# Upper Control Arms
AX1_config.hpr_ucaf.flat[:] = [-235, 213, 89 + TR]
AX1_config.hpr_ucar.flat[:] = [ 170, 262, 61 + TR]
AX1_config.hpr_ucao.flat[:] = [   7, 466, 80 + TR]

# Lower Control Arms
AX1_config.hpr_lcaf.flat[:] = [-235, 213, -90 + TR]
AX1_config.hpr_lcar.flat[:] = [ 170, 262, -62 + TR]
AX1_config.hpr_lcao.flat[:] = [  -7, 483, -80 + TR]

# Tie-Rod
AX1_config.hpr_tri.flat[:] = [-122, 227, -122 + TR]
AX1_config.hpr_tro.flat[:] = [-122, 456, -132 + TR]

# Push-Rod
AX1_config.hpr_pushrod_rocker.flat[:] = [ 6.6, 347, 341 + TR]
AX1_config.hpr_pushrod_uca.flat[:] = [ 6.5, 412, 106 + TR]

# Struts
AX1_config.hpr_strut_chassis.flat[:] = [ 6.5, 22.5, 377 + TR]
AX1_config.hpr_strut_rocker.flat[:]  = [ 6.5, 263, 399 + TR]
AX1_config.pt1_far_strut = AX1_config.hpr_strut_chassis
AX1_config.pt2_far_strut = AX1_config.hpr_strut_rocker
AX1_config.s_strut_freelength = 255

# Bell-Crank
AX1_config.hpr_rocker_chassis.flat[:] = [ 6.5, 280, 320 + TR]

# Wheel Center
AX1_config.hpr_wc.flat[:]  = [0, 525, 0 + TR]
AX1_config.hpr_wc1.flat[:] = [0, 550, 0 + TR]
AX1_config.hpr_wc2.flat[:] = [0, 500, 0 + TR]
AX1_config.pt1_far_tire = AX1_config.hpr_wc

# Helpers
AX1_config.vcs_x.flat[:] = [1, 0, 0]
AX1_config.vcs_y.flat[:] = [0, 1, 0]
AX1_config.vcs_z.flat[:] = [0, 0, 1]

AX1_config.s_tire_radius = TR
AX1_config.s_hub_radius  = 0.3 * TR
AX1_config.s_links_ro    = 8
AX1_config.s_strut_inner = 15
AX1_config.s_strut_outer = 22
AX1_config.s_thickness   = 8

#AX1_config.ax1_far_drive = AX1_config.vcs_y

# Assembling the configuration
AX1_config.assemble()


# Overriding some configuration data
# ==================================

wheel_inertia =  np.array([[1*1e4, 0,      0 ],
                           [0    , 1*1e9,  0 ],
                           [0    , 0, 1*1e4  ]])


AX1_config.Jbar_rbr_hub = wheel_inertia
AX1_config.Jbar_rbl_hub = wheel_inertia

AX1_config.m_rbr_hub = 11*1e3
AX1_config.m_rbl_hub = 11*1e3


# =============================================================================
#                       Creating Struts Force Elements
# =============================================================================

def strut_spring(x):
    x = float(x)
    k = 75*1e6
    force = k * x if x >0 else 0
    return force

def strut_damping(v):
    v = v[0,0]
    force = 3*1e6 * v
    return force


AX1_config.UF_far_strut_Fs = strut_spring
AX1_config.UF_fal_strut_Fs = strut_spring
AX1_config.UF_far_strut_Fd = strut_damping
AX1_config.UF_fal_strut_Fd = strut_damping

AX1_config.export_json('config_inputs', 'AX1')
