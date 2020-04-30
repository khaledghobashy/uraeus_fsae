import sys
import os

import numpy as np
import pandas as pd
from uraeus.nmbd.python import configuration

json_file = os.path.abspath('../symenv/templates/rack_steering/data/rack_steering_cfg.json')

ST1_config = configuration('front_steer')
ST1_config.construct_from_json(json_file)

# =============================================================================
#                                   Numerical Data
# =============================================================================

# Tire Radius
TR = 254

ST1_config.hpr_rack_end.flat[:] = [-122, 227, -122 + TR]
ST1_config.vcs_y.flat[:] = [0, 1, 0]
ST1_config.s_rack_radius = 12

# Loading data into the configuration instance
ST1_config.assemble()

ST1_config.export_json('config_inputs', 'ST1')
