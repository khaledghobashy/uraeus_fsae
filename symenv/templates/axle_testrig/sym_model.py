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
project_name = 'front_axle_testrig'
sym_model = template_topology(project_name)

# Adding Bodies
# =============
sym_model.add_body('wheel_hub', virtual=True, mirror=True)
sym_model.add_body('wheel_upright', virtual=True, mirror=True)

# Adding Joints
# =============
sym_model.add_joint.revolute('hub_bearing', 'vbr_wheel_hub', 'vbr_wheel_upright', virtual=True, mirror=True)

# Adding Actuators
# ================
sym_model.add_actuator.rotational_actuator('wheel_lock', 'jcr_hub_bearing', mirror=True)

# Adding Forces
# =============


# Assembling and Saving the model
sym_model.assemble()
sym_model.save(data_dir)

# ============================================================= #
#                     Symbolic Configuration
# ============================================================= #

# Symbolic configuration name.
config_name = '%s_cfg'%project_name

# Symbolic configuration instance.
sym_config = configuration(config_name, sym_model)

# Adding the desired set of UserInputs
# ====================================
sym_config.add_vector.UserInput('y')


# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

sym_config.add_relation.Equal_to('ax1_jcr_hub_bearing', ('vcs_y',), mirror=True)

# Creating Geometries
# ===================

# Exporing the configuration as a JSON file
sym_config.export_JSON_file(data_dir)

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import templatebased_project
project = templatebased_project(database_directory)

project.write_topology_code(sym_model)
