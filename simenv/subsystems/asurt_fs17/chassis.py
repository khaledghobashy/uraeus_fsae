import sys
import os

import numpy as np
import pandas as pd
from uraeus.nmbd.python import configuration

json_file = os.path.abspath('../symenv/templates/chassis/data/chassis_cfg.json')

CH_config = configuration('chassis')
CH_config.construct_from_json(json_file)

# =============================================================================
#                                   Numerical Data
# =============================================================================

# Wheel Base
WB = 1600

CH_config.hps_CG.flat[:] = [WB/2, 0, 300]

CH_config.s_CG_radius = 80

# Loading data into the configuration instance
CH_config.assemble()

CH_config.m_rbs_chassis = 250*1e3
CH_config.Jbar_rbs_chassis = np.array([[120*1e9 , 0, 0   ],
                                       [0   , 150*1e9 ,0 ],
                                       [0   , 0, 150*1e9 ]])





CH_config.export_json('config_inputs', 'CH')

