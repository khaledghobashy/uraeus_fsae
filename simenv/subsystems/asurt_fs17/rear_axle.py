import sys
import os

import numpy as np
import pandas as pd
from uraeus.nmbd.python import configuration

from uraeus_fsae.numenv.python.templates import dwb_bellcrank_push
#from ..components.struts_data import stiffness_func, damping_func

json_file = os.path.abspath('../symenv/templates/double_wishbone_bellcrank/data/dwb_bellcrank_push_cfg.json')

AX2_config = configuration('rear_axle')
AX2_config.construct_from_json(json_file)

# =============================================================================
#                                   Numerical Data
# =============================================================================

# Tire Radius
TR = 254

# Wheel Base
WB = 1600

# Upper Control Arms
AX2_config.hpr_ucaf.flat[:] = [ -60 + WB, 250, 52 + TR]
AX2_config.hpr_ucar.flat[:] = [-247 + WB, 276, 25 + TR]
AX2_config.hpr_ucao.flat[:] = [   0 + WB, 462, 90 + TR]

# Lower Control Arms
AX2_config.hpr_lcaf.flat[:] = [ -60 + WB, 250, -84 + TR]
AX2_config.hpr_lcar.flat[:] = [-202 + WB, 269, -79 + TR]
AX2_config.hpr_lcao.flat[:] = [   0 + WB, 462, -90 + TR]

# Tie-Rod
AX2_config.hpr_tri.flat[:] = [-247 + WB, 276, -77 + TR]
AX2_config.hpr_tro.flat[:] = [-200 + WB, 462, -90 + TR]

# Push-Rod
AX2_config.hpr_pushrod_rocker.flat[:] = [-65 + WB, 336, 207 + TR]
AX2_config.hpr_pushrod_uca.flat[:]    = [-65 + WB, 381,  90 + TR]

# Bell-Crank
AX2_config.hpr_rocker_chassis.flat[:] = [-65 + WB, 289, 190 + TR]

# Struts
AX2_config.hpr_strut_chassis.flat[:] = [-65 + WB,  34, 202 + TR]
AX2_config.hpr_strut_rocker.flat[:]  = [-65 + WB, 270, 241 + TR]
AX2_config.s_strut_freelength = 255

AX2_config.pt1_far_strut = AX2_config.hpr_strut_chassis
AX2_config.pt2_far_strut = AX2_config.hpr_strut_rocker


# Wheel Center
AX2_config.hpr_wc.flat[:]  = [0 + WB, 525, 0 + TR]
AX2_config.hpr_wc1.flat[:] = [0 + WB, 550, 0 + TR]
AX2_config.hpr_wc2.flat[:] = [0 + WB, 500, 0 + TR]
AX2_config.pt1_far_tire = AX2_config.hpr_wc

# Helpers
AX2_config.vcs_x.flat[:] = [1, 0, 0]
AX2_config.vcs_y.flat[:] = [0, 1, 0]
AX2_config.vcs_z.flat[:] = [0, 0, 1]

AX2_config.s_tire_radius = TR
AX2_config.s_hub_radius  = 0.3 * TR
AX2_config.s_links_ro    = 12
AX2_config.s_strut_inner = 21
AX2_config.s_strut_outer = 25
AX2_config.s_thickness   = 20

#AX2_config.ax1_far_drive = AX2_config.vcs_y


# Loading data into the configuration instance
AX2_config.assemble()

# Overriding some configuration data
# ==================================
wheel_inertia =  np.array([[1*1e4, 0,      0 ],
                           [0    , 1*1e9,  0 ],
                           [0    , 0, 1*1e4  ]])


AX2_config.Jbar_rbr_hub = wheel_inertia
AX2_config.Jbar_rbl_hub = wheel_inertia

AX2_config.m_rbr_hub = 15*1e3
AX2_config.m_rbl_hub = 15*1e3

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


AX2_config.UF_far_strut_Fs = strut_spring
AX2_config.UF_fal_strut_Fs = strut_spring
AX2_config.UF_far_strut_Fd = strut_damping
AX2_config.UF_fal_strut_Fd = strut_damping


AX2_config.export_json('config_inputs', 'AX2')

