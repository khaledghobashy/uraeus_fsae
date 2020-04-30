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
project_name = 'parallel_link_steering'
sym_model = template_topology(project_name)

# Adding Bodies
# =============
sym_model.add_body('coupler')
sym_model.add_body('rocker', mirror=True)

sym_model.add_body('chassis', virtual=True)


# Adding Joints
# =============
sym_model.add_joint.revolute('rocker_chassis', 'rbr_rocker', 'vbs_chassis', mirror=True)
sym_model.add_joint.universal('rocker_uni', 'rbr_rocker', 'rbs_coupler')
sym_model.add_joint.spherical('rocker_sph', 'rbl_rocker', 'rbs_coupler')


# Adding Actuators
# ================
sym_model.add_actuator.rotational_actuator('steer_act', 'jcl_rocker_chassis')


# Adding Forces
# =============
#

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
sym_config.add_point.UserInput('rocker_chassis', mirror=True)
sym_config.add_point.UserInput('rocker_coupler', mirror=True)

sym_config.add_vector.UserInput('x')
sym_config.add_vector.UserInput('y')
sym_config.add_vector.UserInput('z')


# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# Revolute Joint:
# ===============
sym_config.add_relation.Equal_to('pt1_jcr_rocker_chassis', ('hpr_rocker_chassis',), mirror=True)
sym_config.add_relation.Equal_to('ax1_jcr_rocker_chassis', ('vcs_z',), mirror=True)

# Spherical Joint:
# ================
sym_config.add_relation.Equal_to('pt1_jcs_rocker_sph', ('hpl_rocker_coupler',))
sym_config.add_relation.Equal_to('ax1_jcs_rocker_sph', ('vcs_z',))

# Universal Joint:
# ================
sym_config.add_relation.Equal_to('pt1_jcs_rocker_uni', ('hpr_rocker_coupler',))
sym_config.add_relation.Oriented('ax1_jcs_rocker_uni', ('hpr_rocker_coupler', 'hpl_rocker_coupler'))
sym_config.add_relation.Oriented('ax2_jcs_rocker_uni', ('hpl_rocker_coupler', 'hpr_rocker_coupler'))

# Creating Geometries
# ===================
sym_config.add_scalar.UserInput('links_radius')

sym_config.add_geometry.Cylinder_Geometry('coupler', ('hpr_rocker_coupler', 'hpl_rocker_coupler','s_links_radius'))
sym_config.assign_geometry_to_body('rbs_coupler', 'gms_coupler')

sym_config.add_geometry.Cylinder_Geometry('rocker', ('hpr_rocker_coupler', 'hpr_rocker_chassis','s_links_radius'), mirror=True)
sym_config.assign_geometry_to_body('rbr_rocker', 'gmr_rocker', mirror=True)


# Exporing the configuration as a JSON file
sym_config.export_JSON_file(data_dir)

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import templatebased_project
project = templatebased_project(database_directory)

project.write_topology_code(sym_model)
