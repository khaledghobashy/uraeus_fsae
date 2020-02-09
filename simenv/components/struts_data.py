
import os
import pandas as pd
from scipy.interpolate import interp1d

file_name = os.path.dirname(os.path.abspath(__file__))
data_dir  = os.path.abspath(os.path.join(file_name, os.path.pardir))


adiabatic_data = pd.read_csv(os.path.join(data_dir, 'components/force_elements', 'adiabatic.csv'), header=0)
damping_data   = pd.read_csv(os.path.join(data_dir, 'components/force_elements', 'damping.csv'), header=0)


stiffness_boundries = (min(adiabatic_data['y']*1e3*1e6), max(adiabatic_data['y']*1e3*1e6))
damping_boundries   = (min(damping_data['y']*1e3*1e6), max(damping_data['y']*1e3*1e6))


stiffness_func = interp1d(adiabatic_data['x'], adiabatic_data['y']*1e3*1e6,
                          fill_value = stiffness_boundries, bounds_error=False)


damping_func = interp1d(damping_data['x']*1e3, damping_data['y']*1e3*1e6,
                        fill_value = damping_boundries, bounds_error=False)



