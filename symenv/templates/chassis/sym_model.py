
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
model_name = 'chassis'
sym_model = template_topology(model_name)

# Adding Bodies
# =============
sym_model.add_body('chassis')

# Adding Forces
# =============
sym_model.add_force.generic_load('aero_drag', 'rbs_chassis')

# Assembling and Saving the model
sym_model.save(data_dir)
sym_model.assemble()

# ============================================================= #
#                     Symbolic Configuration
# ============================================================= #

# Symbolic configuration name.
config_name = '%s_cfg'%model_name
sym_config = configuration(config_name, sym_model)

# Adding UserInputs
# =================
sym_config.add_point.UserInput('CG')

# Defining Relations between original topology inputs
# and our desired UserInputs.
# ===================================================

# Aero Drag Force:
sym_config.add_relation.Equal_to('pt1_fas_aero_drag', ('hps_CG',))

# Creating Geometries
# ===================
sym_config.add_scalar.UserInput('CG_radius')

sym_config.add_geometry.Sphere_Geometry('CG', ('hps_CG','s_CG_radius'))
sym_config.assign_geometry_to_body('rbs_chassis', 'gms_CG', False)
sym_config.add_relation.Equal_to('R_rbs_chassis', ('gms_CG.R',))
sym_config.add_relation.Equal_to('P_rbs_chassis', ('gms_CG.P',))

# Exporing the configuration as a JSON file
sym_config.export_JSON_file(data_dir)

# ============================================================= #
#                     Code Generation
# ============================================================= #

from uraeus.nmbd.python import templatebased_project

project = templatebased_project(database_directory)
project.write_topology_code(sym_model)
