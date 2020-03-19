import sys
import os

import numpy as np
import pandas as pd
from uraeus.nmbd.python import configuration

json_file = os.path.abspath('../symenv/templates/drive_shafts_v1/data/drive_shafts_v1_cfg.json')

DR2_config = configuration('DR2_v1')
DR2_config.construct_from_json(json_file)

# =============================================================================
#                                   Numerical Data
# =============================================================================

# Wheel Base
WB = 1600
TR = 254

DR2_config.hpr_diff_input.flat[:] = [WB - 0, 50, TR + 20]
DR2_config.hpr_inner_cv.flat[:] = [WB - 0, 100, TR + 20]
DR2_config.hpr_outer_cv.flat[:] = [WB, 525 - 100, TR]

# Helpers
DR2_config.vcs_x.flat[:] = [1, 0, 0]
DR2_config.vcs_y.flat[:] = [0, 1, 0]
DR2_config.vcs_z.flat[:] = [0, 0, 1]

DR2_config.s_shafts_radius = 15

# Loading data into the configuration instance
DR2_config.assemble()

DR2_config.export_json('config_inputs', 'DR2')

