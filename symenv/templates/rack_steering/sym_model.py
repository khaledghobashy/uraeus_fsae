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
project_name = 'rack_steering'
sym_model = template_topology(project_name)

# Adding Bodies
# =============
sym_model.add_body('rack')
sym_model.add_body('chassis', virtual=True)

# Adding Joints
# =============
sym_model.add_joint.translational('rack', 'rbs_rack', 'vbs_chassis')

# Adding Actuators
# ================
sym_model.add_actuator.translational_actuator('rack_act', 'jcs_rack')

# Adding Forces
# =============
# TODO


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
sym_config.add_point.UserInput('rack_end', mirror=True)
sym_config.add_vector.UserInput('y')

# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# Translational Joint:
sym_config.add_relation.Centered('pt1_jcs_rack', ('hpr_rack_end', 'hpl_rack_end'))
sym_config.add_relation.Equal_to('ax1_jcs_rack', ('vcs_y',))


# Creating Geometries
# ===================
sym_config.add_scalar.UserInput('rack_radius')

sym_config.add_geometry.Cylinder_Geometry('rack', ('hpr_rack_end','hpl_rack_end','s_rack_radius'))
sym_config.assign_geometry_to_body('rbs_rack', 'gms_rack')


# Exporing the configuration as a JSON file
sym_config.export_JSON_file(data_dir)

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import templatebased_project
project = templatebased_project(database_directory)

project.write_topology_code(sym_model)
