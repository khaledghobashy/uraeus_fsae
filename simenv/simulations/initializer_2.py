import sys
import os

import numpy as np
import pandas as pd

from uraeus.nmbd.python import simulation
from uraeus.nmbd.python.engine.numerics.math_funcs import A

database_directory = os.path.abspath('../../')
print('Database Directory: %s \n'%database_directory)
sys.path.append(database_directory)

from uraeus_fsae.simenv.assemblies import asurt_FS17_v2 as num_assm

num_model = num_assm.num_model

print('INITIALIZED MODEL SUCCESSFULLY!')
print('===============================')
