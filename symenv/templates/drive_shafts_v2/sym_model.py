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
project_name = 'drive_shafts_v2'
sym_model = template_topology(project_name)

# Adding Bodies
# =============

# Drive Shafts
sym_model.add_body('inner_shaft', mirror=True)
sym_model.add_body('coupling_inner', mirror=True)
sym_model.add_body('coupling_outer', mirror=True)

# Helper Bodies
sym_model.add_body('differential', virtual=True)
sym_model.add_body('wheel_hub', virtual=True, mirror=True)

# Adding Joints
# =============
sym_model.add_joint.revolute('diff_joint', 'rbr_inner_shaft', 'vbs_differential', mirror=True)
sym_model.add_joint.universal('inner_cv', 'rbr_inner_shaft', 'rbr_coupling_inner', mirror=True)
sym_model.add_joint.translational('coupling_trans', 'rbr_coupling_inner', 'rbr_coupling_outer', mirror=True)
sym_model.add_joint.universal('outer_cv', 'rbr_coupling_outer', 'vbr_wheel_hub', mirror=True)

# Adding Forces
# =============
sym_model.add_force.local_torque('drive', 'rbr_inner_shaft', mirror=True)


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
sym_config.add_point.UserInput('inner_cv', mirror=True)
sym_config.add_point.UserInput('outer_cv', mirror=True)
sym_config.add_point.UserInput('diff_input', mirror=True)

sym_config.add_vector.UserInput('x')
sym_config.add_vector.UserInput('y')
sym_config.add_vector.UserInput('z')

# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================
sym_config.add_point.Centered('coupling_mid', ('hpr_inner_cv', 'hpr_outer_cv'), mirror=True)

# Inner CV Joint:
# ===============
sym_config.add_relation.Equal_to('pt1_jcr_inner_cv', ('hpr_inner_cv',), mirror=True)
sym_config.add_relation.Oriented('ax1_jcr_inner_cv', ('hpr_inner_cv', 'hpr_diff_input'), mirror=True)
sym_config.add_relation.Oriented('ax2_jcr_inner_cv', ('hpr_outer_cv', 'hpr_inner_cv'), mirror=True)

# Outer CV Joint:
# ===============
sym_config.add_relation.Equal_to('pt1_jcr_outer_cv', ('hpr_outer_cv',), mirror=True)
sym_config.add_relation.Oriented('ax1_jcr_outer_cv', ('hpr_outer_cv', 'hpr_inner_cv'), mirror=True)
sym_config.add_relation.Equal_to('ax2_jcr_outer_cv', ('vcs_y',), mirror=True)

# Coupling Trans Joint:
# ====================
sym_config.add_relation.Equal_to('pt1_jcr_coupling_trans', ('hpr_coupling_mid',), mirror=True)
sym_config.add_relation.Oriented('ax1_jcr_coupling_trans', ('hpr_outer_cv', 'hpr_inner_cv'), mirror=True)

# Differential Joint:
# ===================
sym_config.add_relation.Equal_to('pt1_jcr_diff_joint', ('hpr_diff_input',), mirror=True)
sym_config.add_relation.Equal_to('ax1_jcr_diff_joint', ('vcs_y',), mirror=True)

# Drive Torque:
# =============
sym_config.add_relation.Equal_to('ax1_far_drive', ('vcs_y',), mirror=True)

# Creating Geometries
# ===================
sym_config.add_scalar.UserInput('shafts_radius')

sym_config.add_geometry.Cylinder_Geometry('inner_shaft', ('hpr_diff_input', 'hpr_inner_cv','s_shafts_radius'), mirror=True)
sym_config.assign_geometry_to_body('rbr_inner_shaft', 'gmr_inner_shaft', mirror=True)

sym_config.add_geometry.Cylinder_Geometry('coupling_inner', ('hpr_inner_cv', 'hpr_coupling_mid','s_shafts_radius'), mirror=True)
sym_config.assign_geometry_to_body('rbr_coupling_inner', 'gmr_coupling_inner', mirror=True)

sym_config.add_geometry.Cylinder_Geometry('coupling_outer', ('hpr_coupling_mid', 'hpr_outer_cv','s_shafts_radius'), mirror=True)
sym_config.assign_geometry_to_body('rbr_coupling_outer', 'gmr_coupling_outer', mirror=True)


# Exporing the configuration as a JSON file
sym_config.export_JSON_file(data_dir)

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import templatebased_project
project = templatebased_project(database_directory)

project.write_topology_code(sym_model)
