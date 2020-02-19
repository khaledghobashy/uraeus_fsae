import os
database_directory = os.path.abspath('../../')

from uraeus.nmbd.python import templatebased_project

project = templatebased_project(database_directory)
project.create_dirs()

stpl_file = os.path.join(database_directory, 'symenv/templates/objects/dwb_bellcrank_push.stpl')
project.write_topology_code(stpl_file)
