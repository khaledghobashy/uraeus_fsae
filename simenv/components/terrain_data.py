
import os
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline


def normalize(v):
    normalized = v/np.linalg.norm(v)
    return normalized

normal = np.array([[0], [0], [1]], dtype=np.float64)

class terrain(object):

   def __init__(self):
      pass

   def get_state(self, x, y):
      return [normal, 0]


terrain_data = pd.read_csv(os.path.abspath('components/terrains/terrain_3.csv'))
scale = 20
x = np.unique(terrain_data['Px']) * scale
y = np.unique(terrain_data['Py']) * scale

dim = len(x)

z = np.zeros((dim,dim))
for i in range(dim):
    z[i,:] = terrain_data['Pz'][i*dim:(i+1)*dim] - 20
z = z.T * scale


interpolator = RectBivariateSpline(x, y, z)
normal = np.array([[0],[0],[1]])

def terrain_state(x, y):
   hieght = interpolator(x, y)
   values = [normal, hieght[0,0]]
   print(values)
   return values
