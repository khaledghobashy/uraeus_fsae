# standard library imports
import os

# uraeus imports
from uraeus.smbd.systems import template_topology, configuration

# getting directory of current file and specifying the directory
# where data will be saved
dir_name = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(dir_name, 'data')
database_directory = os.path.abspath(os.path.join(dir_name, '../../../'))

# ============================================================= #
#                       Symbolic Topology
# ============================================================= #

# Creating the symbolic topology as an instance of the
# standalone_topology class
project_name = 'double_wishbone_bellcrank'
sym_model = template_topology(project_name)

# Adding Bodies
# =============
sym_model.add_body('uca', mirror=True)
sym_model.add_body('lca', mirror=True)
sym_model.add_body('upright', mirror=True)
sym_model.add_body('tie_rod', mirror=True)
sym_model.add_body('hub', mirror=True)

sym_model.add_body('pushrod', mirror=True)
sym_model.add_body('rocker', mirror=True)
sym_model.add_body('upper_strut', mirror=True)
sym_model.add_body('lower_strut', mirror=True)

sym_model.add_body('steer', mirror=True, virtual=True)
sym_model.add_body('chassis', virtual=True)

# Adding Joints
# =============
sym_model.add_joint.spherical('uca_upright', 'rbr_uca', 'rbr_upright', mirror=True)
sym_model.add_joint.spherical('lca_upright', 'rbr_lca', 'rbr_upright', mirror=True)
sym_model.add_joint.spherical('tie_upright', 'rbr_tie_rod', 'rbr_upright', mirror=True)
sym_model.add_joint.spherical('prod_rocker', 'rbr_rocker', 'rbr_pushrod', mirror=True)

sym_model.add_joint.revolute('uca_chassis', 'rbr_uca', 'vbs_chassis', mirror=True)
sym_model.add_joint.revolute('lca_chassis', 'rbr_lca', 'vbs_chassis', mirror=True)
sym_model.add_joint.revolute('hub_bearing', 'rbr_upright', 'rbr_hub', mirror=True)
sym_model.add_joint.revolute('rocker_chassis', 'rbr_rocker', 'vbs_chassis', mirror=True)

sym_model.add_joint.universal('prod_uca', 'rbr_uca', 'rbr_pushrod', mirror=True)
sym_model.add_joint.universal('strut_chassis', 'rbr_upper_strut', 'vbs_chassis', mirror=True)
sym_model.add_joint.universal('strut_rocker', 'rbr_lower_strut', 'rbr_rocker', mirror=True)
sym_model.add_joint.universal('tie_steering', 'rbr_tie_rod', 'vbr_steer', mirror=True)

sym_model.add_joint.cylinderical('strut', 'rbr_upper_strut', 'rbr_lower_strut', mirror=True)

# Adding Actuators
# ================
#

# Adding Forces
# =============
sym_model.add_force.TSDA('strut', 'rbr_upper_strut', 'rbr_lower_strut', mirror=True)
sym_model.add_force.generic_load('tire', 'rbr_hub', mirror=True)
sym_model.add_force.local_torque('drive', 'rbr_hub', mirror=True)


# Assembling and Saving the model
sym_model.save(data_dir)
sym_model.assemble()

# ============================================================= #
#                     Symbolic Configuration
# ============================================================= #

# Symbolic configuration name.
config_name = '%s_cfg'%project_name

# Symbolic configuration instance.
sym_config = configuration(config_name, sym_model)

# Adding the desired set of UserInputs
# ====================================
# Adding UserInputs
# =================

# Upper Control Arm Points
# ========================
sym_config.add_point.UserInput('ucaf', mirror=True)
sym_config.add_point.UserInput('ucar', mirror=True)
sym_config.add_point.UserInput('ucao', mirror=True)

# Lower Control Arm Points
# ========================
sym_config.add_point.UserInput('lcaf', mirror=True)
sym_config.add_point.UserInput('lcar', mirror=True)
sym_config.add_point.UserInput('lcao', mirror=True)

# Tie-Rod Points
# ==============
sym_config.add_point.UserInput('tro', mirror=True)
sym_config.add_point.UserInput('tri', mirror=True)

# Bell-Crank Points
# =================
sym_config.add_point.UserInput('rocker_chassis', mirror=True)
sym_config.add_point.UserInput('pushrod_rocker', mirror=True)
sym_config.add_point.UserInput('pushrod_uca', mirror=True)

# Spring-Damper Points
# ====================
sym_config.add_point.UserInput('strut_chassis', mirror=True)
sym_config.add_point.UserInput('strut_rocker', mirror=True)

# Wheel Center Points
# ===================
sym_config.add_point.UserInput('wc', mirror=True)
sym_config.add_point.UserInput('wc1', mirror=True)
sym_config.add_point.UserInput('wc2', mirror=True)

# Guiding Global Axes
# ===================
sym_config.add_vector.UserInput('x')
sym_config.add_vector.UserInput('y')
sym_config.add_vector.UserInput('z')


# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# Coil-Spring FreeLength
sym_config.add_scalar.UserInput('strut_freelength')
sym_config.add_relation.Equal_to('far_strut_FL', ('s_strut_freelength',), mirror=True)

# OTHER POINTS
sym_config.add_point.Centered('strut_mid', ('hpr_strut_chassis', 'hpr_strut_rocker'), mirror=True)

# JOINTS CONFIGURATIONS

# UCA_Upright Spherical joint
# ===========================
sym_config.add_relation.Equal_to('pt1_jcr_uca_upright', ('hpr_ucao',), mirror=True)
sym_config.add_relation.Equal_to('ax1_jcr_uca_upright', ('vcs_z',), mirror=True)

# LCA_Upright Spherical joint
# ===========================
sym_config.add_relation.Equal_to('pt1_jcr_lca_upright', ('hpr_lcao',), mirror=True)
sym_config.add_relation.Equal_to('ax1_jcr_lca_upright', ('vcs_z',), mirror=True)

# TieRod_Upright Spherical joint
# ==============================
sym_config.add_relation.Equal_to('pt1_jcr_tie_upright', ('hpr_tro',), mirror=True)
sym_config.add_relation.Equal_to('ax1_jcr_tie_upright', ('hpr_tro',), mirror=True)

# PRod_Rocker Spherical joint
# ===========================
sym_config.add_relation.Equal_to('pt1_jcr_prod_rocker', ('hpr_pushrod_rocker',), mirror=True)
sym_config.add_relation.Equal_to('ax1_jcr_prod_rocker', ('hpr_pushrod_rocker',), mirror=True)

# Rocker-Chassis Revolute Joint:
# ==============================
sym_config.add_relation.Oriented('ax1_jcr_rocker_chassis', ('hpr_rocker_chassis','hpr_pushrod_rocker','hpr_strut_rocker'), mirror=True)
sym_config.add_relation.Equal_to('pt1_jcr_rocker_chassis', ('hpr_rocker_chassis',), mirror=True)

# PushRod-UCA Universal Joint:
# ============================
sym_config.add_relation.Equal_to('pt1_jcr_prod_uca', ('hpr_pushrod_uca',), mirror=True)
sym_config.add_relation.Oriented('ax1_jcr_prod_uca', ('hpr_pushrod_uca','hpr_pushrod_rocker'), mirror=True)
sym_config.add_relation.Oriented('ax2_jcr_prod_uca', ('hpr_pushrod_rocker','hpr_pushrod_uca'), mirror=True)

# Upper Control Arm Revolute Joint:
# =================================
sym_config.add_relation.Centered('pt1_jcr_uca_chassis', ('hpr_ucaf','hpr_ucar'), mirror=True)
sym_config.add_relation.Oriented('ax1_jcr_uca_chassis', ('hpr_ucaf','hpr_ucar'), mirror=True)

# Lower Control Arm Revolute Joint:
# ================================
sym_config.add_relation.Centered('pt1_jcr_lca_chassis', ('hpr_lcaf','hpr_lcar'), mirror=True)
sym_config.add_relation.Oriented('ax1_jcr_lca_chassis', ('hpr_lcaf','hpr_lcar'), mirror=True)

# Wheel Hub Revolute Joint:
# =========================
sym_config.add_relation.Equal_to('pt1_jcr_hub_bearing', ('hpr_wc',), mirror=True)
sym_config.add_relation.Equal_to('ax1_jcr_hub_bearing', ('vcs_y',), mirror=True)

# Strut-Chassis Universal Joint:
# ==============================
sym_config.add_relation.Equal_to('pt1_jcr_strut_chassis', ('hpr_strut_chassis',), mirror=True)
sym_config.add_relation.Oriented('ax1_jcr_strut_chassis', ('hpr_strut_chassis','hpr_strut_mid'), mirror=True)
sym_config.add_relation.Oriented('ax2_jcr_strut_chassis', ('hpr_strut_mid','hpr_strut_chassis'), mirror=True)

# Strut-Rocker Universal Joint:
# =============================
sym_config.add_relation.Equal_to('pt1_jcr_strut_rocker', ('hpr_strut_rocker',), mirror=True)
sym_config.add_relation.Oriented('ax1_jcr_strut_rocker', ('hpr_strut_chassis','hpr_strut_rocker'), mirror=True)
sym_config.add_relation.Oriented('ax2_jcr_strut_rocker', ('hpr_strut_rocker','hpr_strut_chassis'), mirror=True)

# Tie-Steer Universal Joint:
# ==========================
sym_config.add_relation.Equal_to('pt1_jcr_tie_steering', ('hpr_tri',), mirror=True)
sym_config.add_relation.Oriented('ax1_jcr_tie_steering', ('hpr_tri','hpr_tro'), mirror=True)
sym_config.add_relation.Oriented('ax2_jcr_tie_steering', ('hpr_tro','hpr_tri'), mirror=True)

# Strut Cylinderical Joint:
# =========================
sym_config.add_relation.Equal_to('pt1_jcr_strut', ('hpr_strut_mid',), mirror=True)
sym_config.add_relation.Oriented('ax1_jcr_strut', ('hpr_strut_rocker','hpr_strut_chassis'), mirror=True)

# Strut Force Points:
# ===================
sym_config.add_relation.Equal_to('pt1_far_strut', ('hpr_strut_chassis',), mirror=True)
sym_config.add_relation.Equal_to('pt2_far_strut', ('hpr_strut_rocker',), mirror=True)

# Wheel Torque Axis:
# ==================
sym_config.add_relation.Equal_to('ax1_far_drive', ('vcs_y',), mirror=True)



# Creating Geometries
# ===================
# GEOMETRIES
# ==========
sym_config.add_scalar.UserInput('links_ro')
sym_config.add_scalar.UserInput('strut_outer')
sym_config.add_scalar.UserInput('strut_inner')
sym_config.add_scalar.UserInput('thickness')
sym_config.add_scalar.UserInput('hub_radius')
sym_config.add_scalar.UserInput('tire_radius')


# Upper Control Arm
# =================
#sym_config.add_geometry.Triangular_Prism('uca', ('hpr_ucaf','hpr_ucar','hpr_ucao','s_thickness'), mirror=True)
sym_config.add_geometry.Cylinder_Geometry('uca_c1', ('hpr_ucaf', 'hpr_ucao','s_links_ro'), mirror=True)
sym_config.add_geometry.Cylinder_Geometry('uca_c2', ('hpr_ucar', 'hpr_ucao','s_links_ro'), mirror=True)
sym_config.add_geometry.Composite_Geometry('uca', ('gmr_uca_c1', 'gmr_uca_c2'), mirror=True)
sym_config.assign_geometry_to_body('rbr_uca', 'gmr_uca', mirror=True)

# Lower Control Arm
# =================
#sym_config.add_geometry.Triangular_Prism('lca', ('hpr_lcaf','hpr_lcar','hpr_lcao','s_thickness'), mirror=True)
sym_config.add_geometry.Cylinder_Geometry('lca_c1', ('hpr_lcaf', 'hpr_lcao','s_links_ro'), mirror=True)
sym_config.add_geometry.Cylinder_Geometry('lca_c2', ('hpr_lcar', 'hpr_lcao','s_links_ro'), mirror=True)
sym_config.add_geometry.Composite_Geometry('lca', ('gmr_lca_c1', 'gmr_lca_c2'), mirror=True)
sym_config.assign_geometry_to_body('rbr_lca', 'gmr_lca', mirror=True)

# Bell-Crank
# ==========
sym_config.add_geometry.Triangular_Prism('rocker', ('hpr_strut_rocker','hpr_pushrod_rocker','hpr_rocker_chassis','s_thickness'), mirror=True)
sym_config.assign_geometry_to_body('rbr_rocker','gmr_rocker',mirror=True)

# Wheel Upright
# =============
sym_config.add_geometry.Triangular_Prism('upright', ('hpr_ucao','hpr_wc','hpr_lcao','s_thickness'), mirror=True)
sym_config.assign_geometry_to_body('rbr_upright', 'gmr_upright', mirror=True)

# Coil-Over Upper Part
# ====================
sym_config.add_geometry.Cylinder_Geometry('upper_strut', ('hpr_strut_chassis','hpr_strut_mid','s_strut_outer') , mirror=True)
sym_config.assign_geometry_to_body('rbr_upper_strut', 'gmr_upper_strut', mirror=True)

# Coil-Over Lower Part
# ====================
sym_config.add_geometry.Cylinder_Geometry('lower_strut', ('hpr_strut_mid','hpr_strut_rocker','s_strut_inner'), mirror=True)
sym_config.assign_geometry_to_body('rbr_lower_strut', 'gmr_lower_strut', mirror=True)

# TieRod
# ======
sym_config.add_geometry.Cylinder_Geometry('tie_rod', ('hpr_tri','hpr_tro','s_links_ro'), mirror=True)
sym_config.assign_geometry_to_body('rbr_tie_rod','gmr_tie_rod',mirror=True)

# PushRod
# =======
sym_config.add_geometry.Cylinder_Geometry('pushrod', ('hpr_pushrod_uca','hpr_pushrod_rocker','s_links_ro'), mirror=True)
sym_config.assign_geometry_to_body('rbr_pushrod', 'gmr_pushrod', mirror=True)

# Wheel Hub
# =========
sym_config.add_geometry.Cylinder_Geometry('hub_cyl', ('hpr_wc2','hpr_wc','s_hub_radius'), mirror=True)
sym_config.add_geometry.Cylinder_Geometry('tire', ('hpr_wc1','hpr_wc','s_tire_radius'), mirror=True)
sym_config.add_geometry.Composite_Geometry('hub', ('gmr_hub_cyl', 'gmr_tire'), mirror=True)
sym_config.assign_geometry_to_body('rbr_hub', 'gmr_hub', mirror=True)


# Exporing the configuration as a JSON file
sym_config.export_JSON_file(data_dir)

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import templatebased_project
project = templatebased_project(database_directory)

project.write_topology_code(sym_model)
