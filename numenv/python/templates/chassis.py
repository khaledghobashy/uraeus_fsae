
import numpy as np
from numpy import cos, sin
from numpy.linalg import multi_dot
from scipy.misc import derivative

from uraeus.nmbd.python.engine.numerics.math_funcs import A, B, G, E, triad, skew




class topology(object):

    def __init__(self,prefix=''):
        self.t = 0.0
        self.prefix = (prefix if prefix=='' else prefix+'.')
        self.config = None

        self.indicies_map = {'vbs_ground': 0, 'rbs_chassis': 1}

        self.n  = 7
        self.nc = 1
        self.nrows = 1
        self.ncols = 2*1
        self.rows = np.arange(self.nrows)

        reactions_indicies = []
        self.reactions_indicies = ['%s%s'%(self.prefix,i) for i in reactions_indicies]

    
    def initialize(self):
        self.t = 0
        self.assemble(self.indicies_map, {}, 0)
        self.set_initial_states()
        self.eval_constants()

    def assemble(self, indicies_map, interface_map, rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map, interface_map)
        self.rows += self.rows_offset
        self.jac_rows = np.array([0, 0])
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.rbs_chassis*2, self.rbs_chassis*2+1])

    def set_initial_states(self):
        self.q0  = np.concatenate([self.config.R_rbs_chassis,
        self.config.P_rbs_chassis])
        self.qd0 = np.concatenate([self.config.Rd_rbs_chassis,
        self.config.Pd_rbs_chassis])

        self.set_gen_coordinates(self.q0)
        self.set_gen_velocities(self.qd0)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbs_chassis = indicies_map[p + 'rbs_chassis']
        self.vbs_ground = indicies_map[interface_map[p + 'vbs_ground']]

    
    def eval_constants(self):
        config = self.config

        self.F_rbs_chassis_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_chassis]], dtype=np.float64)

        self.ubar_rbs_chassis_fas_aero_drag = (multi_dot([A(config.P_rbs_chassis).T,config.pt1_fas_aero_drag]) + -1*multi_dot([A(config.P_rbs_chassis).T,config.R_rbs_chassis]))
        self.ubar_vbs_ground_fas_aero_drag = (multi_dot([A(config.P_vbs_ground).T,config.pt1_fas_aero_drag]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))

    
    def set_gen_coordinates(self,q):
        self.R_rbs_chassis = q[0:3,0:1]
        self.P_rbs_chassis = q[3:7,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbs_chassis = qd[0:3,0:1]
        self.Pd_rbs_chassis = qd[3:7,0:1]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_rbs_chassis = qdd[0:3,0:1]
        self.Pdd_rbs_chassis = qdd[3:7,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        pass

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.P_rbs_chassis

        self.pos_eq_blocks = ((-1*np.eye(1, dtype=np.float64) + multi_dot([x0.T,x0])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

    

        self.vel_eq_blocks = (np.zeros((1,1),dtype=np.float64),)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbs_chassis

        self.acc_eq_blocks = (2*multi_dot([a0.T,a0]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

    

        self.jac_eq_blocks = (np.zeros((1,3),dtype=np.float64),
        2*self.P_rbs_chassis.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = G(self.P_rbs_chassis)

        self.mass_eq_blocks = (config.m_rbs_chassis*np.eye(3, dtype=np.float64),
        4*multi_dot([m0.T,config.Jbar_rbs_chassis,m0]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_chassis)
        f1 = self.P_rbs_chassis

        self.frc_eq_blocks = ((config.UF_fas_aero_drag_F() + self.F_rbs_chassis_gravity),
        (8*multi_dot([f0.T,config.Jbar_rbs_chassis,f0,f1]) + 2*multi_dot([G(f1).T,(config.UF_fas_aero_drag_T() + multi_dot([skew(multi_dot([A(f1),self.ubar_rbs_chassis_fas_aero_drag])).T,config.UF_fas_aero_drag_F()]))])),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

    

        self.reactions = {}

