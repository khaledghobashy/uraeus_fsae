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
# TODO : User Declared
project_name = NotImplemented
sym_model = template_topology(project_name)

# Adding Bodies
# =============
# TODO
#sym_model.add_body('body')

# Adding Joints
# =============
# TODO
#sym_model.add_joint.revolute('a', 'ground', 'rbs_body')

# Adding Actuators
# ================
# TODO
#sym_model.add_actuator.rotational_actuator('act', 'jcs_a')

# Adding Forces
# =============
# TODO


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
# TODO :eg
#sym_config.add_point.UserInput('p1')
#sym_config.add_point.UserInput('p2')

#sym_config.add_vector.UserInput('v')


# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# TODO :eg
# Revolute Joint (a) location and orientation
#sym_config.add_relation.Equal_to('pt1_jcs_a', ('hps_p1',))
#sym_config.add_relation.Equal_to('ax1_jcs_a', ('vcs_v',))

# Creating Geometries
# ===================
# TODO :eg
#sym_config.add_scalar.UserInput('radius')

#sym_config.add_geometry.Sphere_Geometry('body', ('hps_p2', 's_radius'))
#sym_config.assign_geometry_to_body('rbs_body', 'gms_body')

# Exporing the configuration as a JSON file
sym_config.export_JSON_file(data_dir)

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import templatebased_project
project = templatebased_project(database_directory)

project.write_topology_code(sym_model)
