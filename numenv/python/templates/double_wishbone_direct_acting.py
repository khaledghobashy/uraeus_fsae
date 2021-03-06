
import numpy as np
from numpy import cos, sin
from scipy.misc import derivative

from uraeus.nmbd.python.engine.numerics.math_funcs import A, B, G, E, triad, skew, multi_dot

# CONSTANTS
F64_DTYPE = np.float64

I1 = np.eye(1, dtype=F64_DTYPE)
I2 = np.eye(2, dtype=F64_DTYPE)
I3 = np.eye(3, dtype=F64_DTYPE)
I4 = np.eye(4, dtype=F64_DTYPE)

Z1x1 = np.zeros((1,1), F64_DTYPE)
Z1x3 = np.zeros((1,3), F64_DTYPE)
Z3x1 = np.zeros((3,1), F64_DTYPE)
Z3x4 = np.zeros((3,4), F64_DTYPE)
Z4x1 = np.zeros((4,1), F64_DTYPE)
Z4x3 = np.zeros((4,3), F64_DTYPE)



class topology(object):

    def __init__(self,prefix=''):
        self.t = 0.0
        self.prefix = (prefix if prefix=='' else prefix+'.')
        self.config = None

        self.indicies_map = {'vbs_ground': 0, 'rbr_uca': 1, 'rbl_uca': 2, 'rbr_lca': 3, 'rbl_lca': 4, 'rbr_upright': 5, 'rbl_upright': 6, 'rbr_upper_strut': 7, 'rbl_upper_strut': 8, 'rbr_lower_strut': 9, 'rbl_lower_strut': 10, 'rbr_tie_rod': 11, 'rbl_tie_rod': 12, 'rbr_hub': 13, 'rbl_hub': 14, 'vbr_steer': 15, 'vbl_steer': 16, 'vbs_chassis': 17}

        self.n  = 98
        self.nc = 94
        self.nrows = 58
        self.ncols = 2*14
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_rbr_uca_jcr_uca_upright', 'T_rbr_uca_jcr_uca_upright', 'F_rbr_uca_jcr_uca_chassis', 'T_rbr_uca_jcr_uca_chassis', 'F_rbl_uca_jcl_uca_upright', 'T_rbl_uca_jcl_uca_upright', 'F_rbl_uca_jcl_uca_chassis', 'T_rbl_uca_jcl_uca_chassis', 'F_rbr_lca_jcr_lca_upright', 'T_rbr_lca_jcr_lca_upright', 'F_rbr_lca_jcr_lca_chassis', 'T_rbr_lca_jcr_lca_chassis', 'F_rbl_lca_jcl_lca_upright', 'T_rbl_lca_jcl_lca_upright', 'F_rbl_lca_jcl_lca_chassis', 'T_rbl_lca_jcl_lca_chassis', 'F_rbr_upright_jcr_hub_bearing', 'T_rbr_upright_jcr_hub_bearing', 'F_rbl_upright_jcl_hub_bearing', 'T_rbl_upright_jcl_hub_bearing', 'F_rbr_upper_strut_jcr_strut_chassis', 'T_rbr_upper_strut_jcr_strut_chassis', 'F_rbr_upper_strut_jcr_strut', 'T_rbr_upper_strut_jcr_strut', 'F_rbr_upper_strut_far_strut', 'T_rbr_upper_strut_far_strut', 'F_rbl_upper_strut_jcl_strut_chassis', 'T_rbl_upper_strut_jcl_strut_chassis', 'F_rbl_upper_strut_jcl_strut', 'T_rbl_upper_strut_jcl_strut', 'F_rbl_upper_strut_fal_strut', 'T_rbl_upper_strut_fal_strut', 'F_rbr_lower_strut_jcr_strut_lca', 'T_rbr_lower_strut_jcr_strut_lca', 'F_rbl_lower_strut_jcl_strut_lca', 'T_rbl_lower_strut_jcl_strut_lca', 'F_rbr_tie_rod_jcr_tie_upright', 'T_rbr_tie_rod_jcr_tie_upright', 'F_rbr_tie_rod_jcr_tie_steering', 'T_rbr_tie_rod_jcr_tie_steering', 'F_rbl_tie_rod_jcl_tie_upright', 'T_rbl_tie_rod_jcl_tie_upright', 'F_rbl_tie_rod_jcl_tie_steering', 'T_rbl_tie_rod_jcl_tie_steering']
        self.reactions_indicies = ['%s%s'%(self.prefix,i) for i in reactions_indicies]

    
    def initialize(self, q, qd, qdd, lgr):
        self.t = 0
        self.assemble(self.indicies_map, {}, 0)
        self._set_states_arrays(q, qd, qdd, lgr)
        self._map_states_arrays()
        self.set_initial_states()
        self.eval_constants()

    def assemble(self, indicies_map, interface_map, rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map, interface_map)
        self.rows += self.rows_offset
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.rbr_uca*2, self.rbr_uca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.vbr_steer*2, self.vbr_steer*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.vbr_steer*2, self.vbr_steer*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.vbl_steer*2, self.vbl_steer*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.vbl_steer*2, self.vbl_steer*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbl_hub*2, self.rbl_hub*2+1], dtype=np.intc)

    def _set_states_arrays(self, q, qd, qdd, lgr):
        self._q = q
        self._qd = qd
        self._qdd = qdd
        self._lgr = lgr

    def _map_states_arrays(self):
        self._map_gen_coordinates()
        self._map_gen_velocities()
        self._map_gen_accelerations()
        self._map_lagrange_multipliers()

    def set_initial_states(self):
        np.concatenate([self.config.R_rbr_uca,
        self.config.P_rbr_uca,
        self.config.R_rbl_uca,
        self.config.P_rbl_uca,
        self.config.R_rbr_lca,
        self.config.P_rbr_lca,
        self.config.R_rbl_lca,
        self.config.P_rbl_lca,
        self.config.R_rbr_upright,
        self.config.P_rbr_upright,
        self.config.R_rbl_upright,
        self.config.P_rbl_upright,
        self.config.R_rbr_upper_strut,
        self.config.P_rbr_upper_strut,
        self.config.R_rbl_upper_strut,
        self.config.P_rbl_upper_strut,
        self.config.R_rbr_lower_strut,
        self.config.P_rbr_lower_strut,
        self.config.R_rbl_lower_strut,
        self.config.P_rbl_lower_strut,
        self.config.R_rbr_tie_rod,
        self.config.P_rbr_tie_rod,
        self.config.R_rbl_tie_rod,
        self.config.P_rbl_tie_rod,
        self.config.R_rbr_hub,
        self.config.P_rbr_hub,
        self.config.R_rbl_hub,
        self.config.P_rbl_hub], out=self._q)

        np.concatenate([self.config.Rd_rbr_uca,
        self.config.Pd_rbr_uca,
        self.config.Rd_rbl_uca,
        self.config.Pd_rbl_uca,
        self.config.Rd_rbr_lca,
        self.config.Pd_rbr_lca,
        self.config.Rd_rbl_lca,
        self.config.Pd_rbl_lca,
        self.config.Rd_rbr_upright,
        self.config.Pd_rbr_upright,
        self.config.Rd_rbl_upright,
        self.config.Pd_rbl_upright,
        self.config.Rd_rbr_upper_strut,
        self.config.Pd_rbr_upper_strut,
        self.config.Rd_rbl_upper_strut,
        self.config.Pd_rbl_upper_strut,
        self.config.Rd_rbr_lower_strut,
        self.config.Pd_rbr_lower_strut,
        self.config.Rd_rbl_lower_strut,
        self.config.Pd_rbl_lower_strut,
        self.config.Rd_rbr_tie_rod,
        self.config.Pd_rbr_tie_rod,
        self.config.Rd_rbl_tie_rod,
        self.config.Pd_rbl_tie_rod,
        self.config.Rd_rbr_hub,
        self.config.Pd_rbr_hub,
        self.config.Rd_rbl_hub,
        self.config.Pd_rbl_hub], out=self._qd)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbr_uca = indicies_map[p + 'rbr_uca']
        self.rbl_uca = indicies_map[p + 'rbl_uca']
        self.rbr_lca = indicies_map[p + 'rbr_lca']
        self.rbl_lca = indicies_map[p + 'rbl_lca']
        self.rbr_upright = indicies_map[p + 'rbr_upright']
        self.rbl_upright = indicies_map[p + 'rbl_upright']
        self.rbr_upper_strut = indicies_map[p + 'rbr_upper_strut']
        self.rbl_upper_strut = indicies_map[p + 'rbl_upper_strut']
        self.rbr_lower_strut = indicies_map[p + 'rbr_lower_strut']
        self.rbl_lower_strut = indicies_map[p + 'rbl_lower_strut']
        self.rbr_tie_rod = indicies_map[p + 'rbr_tie_rod']
        self.rbl_tie_rod = indicies_map[p + 'rbl_tie_rod']
        self.rbr_hub = indicies_map[p + 'rbr_hub']
        self.rbl_hub = indicies_map[p + 'rbl_hub']
        self.vbr_steer = indicies_map[interface_map[p + 'vbr_steer']]
        self.vbl_steer = indicies_map[interface_map[p + 'vbl_steer']]
        self.vbs_ground = indicies_map[interface_map[p + 'vbs_ground']]
        self.vbs_chassis = indicies_map[interface_map[p + 'vbs_chassis']]

    
    def eval_constants(self):
        config = self.config

        self.F_rbr_uca_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_uca]], dtype=np.float64)
        self.F_rbl_uca_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_uca]], dtype=np.float64)
        self.F_rbr_lca_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_lca]], dtype=np.float64)
        self.F_rbl_lca_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_lca]], dtype=np.float64)
        self.F_rbr_upright_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_upright]], dtype=np.float64)
        self.F_rbl_upright_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_upright]], dtype=np.float64)
        self.F_rbr_upper_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_upper_strut]], dtype=np.float64)
        self.T_rbr_upper_strut_far_strut = Z3x1
        self.T_rbr_lower_strut_far_strut = Z3x1
        self.F_rbl_upper_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_upper_strut]], dtype=np.float64)
        self.T_rbl_upper_strut_fal_strut = Z3x1
        self.T_rbl_lower_strut_fal_strut = Z3x1
        self.F_rbr_lower_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_lower_strut]], dtype=np.float64)
        self.F_rbl_lower_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_lower_strut]], dtype=np.float64)
        self.F_rbr_tie_rod_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_tie_rod]], dtype=np.float64)
        self.F_rbl_tie_rod_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_tie_rod]], dtype=np.float64)
        self.F_rbr_hub_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_hub]], dtype=np.float64)
        self.F_rbl_hub_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_hub]], dtype=np.float64)

        self.Mbar_rbr_uca_jcr_uca_upright = multi_dot([A(config.P_rbr_uca).T,triad(config.ax1_jcr_uca_upright)])
        self.Mbar_rbr_upright_jcr_uca_upright = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_uca_upright)])
        self.ubar_rbr_uca_jcr_uca_upright = (multi_dot([A(config.P_rbr_uca).T,config.pt1_jcr_uca_upright]) + (-1) * multi_dot([A(config.P_rbr_uca).T,config.R_rbr_uca]))
        self.ubar_rbr_upright_jcr_uca_upright = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_uca_upright]) + (-1) * multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.Mbar_rbr_uca_jcr_uca_chassis = multi_dot([A(config.P_rbr_uca).T,triad(config.ax1_jcr_uca_chassis)])
        self.Mbar_vbs_chassis_jcr_uca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcr_uca_chassis)])
        self.ubar_rbr_uca_jcr_uca_chassis = (multi_dot([A(config.P_rbr_uca).T,config.pt1_jcr_uca_chassis]) + (-1) * multi_dot([A(config.P_rbr_uca).T,config.R_rbr_uca]))
        self.ubar_vbs_chassis_jcr_uca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_uca_chassis]) + (-1) * multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_uca_jcl_uca_upright = multi_dot([A(config.P_rbl_uca).T,triad(config.ax1_jcl_uca_upright)])
        self.Mbar_rbl_upright_jcl_uca_upright = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_uca_upright)])
        self.ubar_rbl_uca_jcl_uca_upright = (multi_dot([A(config.P_rbl_uca).T,config.pt1_jcl_uca_upright]) + (-1) * multi_dot([A(config.P_rbl_uca).T,config.R_rbl_uca]))
        self.ubar_rbl_upright_jcl_uca_upright = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_uca_upright]) + (-1) * multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.Mbar_rbl_uca_jcl_uca_chassis = multi_dot([A(config.P_rbl_uca).T,triad(config.ax1_jcl_uca_chassis)])
        self.Mbar_vbs_chassis_jcl_uca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcl_uca_chassis)])
        self.ubar_rbl_uca_jcl_uca_chassis = (multi_dot([A(config.P_rbl_uca).T,config.pt1_jcl_uca_chassis]) + (-1) * multi_dot([A(config.P_rbl_uca).T,config.R_rbl_uca]))
        self.ubar_vbs_chassis_jcl_uca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_uca_chassis]) + (-1) * multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbr_lca_jcr_lca_upright = multi_dot([A(config.P_rbr_lca).T,triad(config.ax1_jcr_lca_upright)])
        self.Mbar_rbr_upright_jcr_lca_upright = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_lca_upright)])
        self.ubar_rbr_lca_jcr_lca_upright = (multi_dot([A(config.P_rbr_lca).T,config.pt1_jcr_lca_upright]) + (-1) * multi_dot([A(config.P_rbr_lca).T,config.R_rbr_lca]))
        self.ubar_rbr_upright_jcr_lca_upright = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_lca_upright]) + (-1) * multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.Mbar_rbr_lca_jcr_lca_chassis = multi_dot([A(config.P_rbr_lca).T,triad(config.ax1_jcr_lca_chassis)])
        self.Mbar_vbs_chassis_jcr_lca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcr_lca_chassis)])
        self.ubar_rbr_lca_jcr_lca_chassis = (multi_dot([A(config.P_rbr_lca).T,config.pt1_jcr_lca_chassis]) + (-1) * multi_dot([A(config.P_rbr_lca).T,config.R_rbr_lca]))
        self.ubar_vbs_chassis_jcr_lca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_lca_chassis]) + (-1) * multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_lca_jcl_lca_upright = multi_dot([A(config.P_rbl_lca).T,triad(config.ax1_jcl_lca_upright)])
        self.Mbar_rbl_upright_jcl_lca_upright = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_lca_upright)])
        self.ubar_rbl_lca_jcl_lca_upright = (multi_dot([A(config.P_rbl_lca).T,config.pt1_jcl_lca_upright]) + (-1) * multi_dot([A(config.P_rbl_lca).T,config.R_rbl_lca]))
        self.ubar_rbl_upright_jcl_lca_upright = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_lca_upright]) + (-1) * multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.Mbar_rbl_lca_jcl_lca_chassis = multi_dot([A(config.P_rbl_lca).T,triad(config.ax1_jcl_lca_chassis)])
        self.Mbar_vbs_chassis_jcl_lca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcl_lca_chassis)])
        self.ubar_rbl_lca_jcl_lca_chassis = (multi_dot([A(config.P_rbl_lca).T,config.pt1_jcl_lca_chassis]) + (-1) * multi_dot([A(config.P_rbl_lca).T,config.R_rbl_lca]))
        self.ubar_vbs_chassis_jcl_lca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_lca_chassis]) + (-1) * multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbr_upright_jcr_hub_bearing = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_hub_bearing)])
        self.Mbar_rbr_hub_jcr_hub_bearing = multi_dot([A(config.P_rbr_hub).T,triad(config.ax1_jcr_hub_bearing)])
        self.ubar_rbr_upright_jcr_hub_bearing = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_hub_bearing]) + (-1) * multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.ubar_rbr_hub_jcr_hub_bearing = (multi_dot([A(config.P_rbr_hub).T,config.pt1_jcr_hub_bearing]) + (-1) * multi_dot([A(config.P_rbr_hub).T,config.R_rbr_hub]))
        self.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_hub_bearing)])
        self.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([A(config.P_rbl_hub).T,triad(config.ax1_jcl_hub_bearing)])
        self.ubar_rbl_upright_jcl_hub_bearing = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_hub_bearing]) + (-1) * multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.ubar_rbl_hub_jcl_hub_bearing = (multi_dot([A(config.P_rbl_hub).T,config.pt1_jcl_hub_bearing]) + (-1) * multi_dot([A(config.P_rbl_hub).T,config.R_rbl_hub]))
        self.Mbar_rbr_upper_strut_jcr_strut_chassis = multi_dot([A(config.P_rbr_upper_strut).T,triad(config.ax1_jcr_strut_chassis)])
        self.Mbar_vbs_chassis_jcr_strut_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax2_jcr_strut_chassis,triad(config.ax1_jcr_strut_chassis)[0:3,1:2])])
        self.ubar_rbr_upper_strut_jcr_strut_chassis = (multi_dot([A(config.P_rbr_upper_strut).T,config.pt1_jcr_strut_chassis]) + (-1) * multi_dot([A(config.P_rbr_upper_strut).T,config.R_rbr_upper_strut]))
        self.ubar_vbs_chassis_jcr_strut_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_strut_chassis]) + (-1) * multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbr_upper_strut_jcr_strut = multi_dot([A(config.P_rbr_upper_strut).T,triad(config.ax1_jcr_strut)])
        self.Mbar_rbr_lower_strut_jcr_strut = multi_dot([A(config.P_rbr_lower_strut).T,triad(config.ax1_jcr_strut)])
        self.ubar_rbr_upper_strut_jcr_strut = (multi_dot([A(config.P_rbr_upper_strut).T,config.pt1_jcr_strut]) + (-1) * multi_dot([A(config.P_rbr_upper_strut).T,config.R_rbr_upper_strut]))
        self.ubar_rbr_lower_strut_jcr_strut = (multi_dot([A(config.P_rbr_lower_strut).T,config.pt1_jcr_strut]) + (-1) * multi_dot([A(config.P_rbr_lower_strut).T,config.R_rbr_lower_strut]))
        self.ubar_rbr_upper_strut_far_strut = (multi_dot([A(config.P_rbr_upper_strut).T,config.pt1_far_strut]) + (-1) * multi_dot([A(config.P_rbr_upper_strut).T,config.R_rbr_upper_strut]))
        self.ubar_rbr_lower_strut_far_strut = (multi_dot([A(config.P_rbr_lower_strut).T,config.pt2_far_strut]) + (-1) * multi_dot([A(config.P_rbr_lower_strut).T,config.R_rbr_lower_strut]))
        self.Mbar_rbl_upper_strut_jcl_strut_chassis = multi_dot([A(config.P_rbl_upper_strut).T,triad(config.ax1_jcl_strut_chassis)])
        self.Mbar_vbs_chassis_jcl_strut_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax2_jcl_strut_chassis,triad(config.ax1_jcl_strut_chassis)[0:3,1:2])])
        self.ubar_rbl_upper_strut_jcl_strut_chassis = (multi_dot([A(config.P_rbl_upper_strut).T,config.pt1_jcl_strut_chassis]) + (-1) * multi_dot([A(config.P_rbl_upper_strut).T,config.R_rbl_upper_strut]))
        self.ubar_vbs_chassis_jcl_strut_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_strut_chassis]) + (-1) * multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_upper_strut_jcl_strut = multi_dot([A(config.P_rbl_upper_strut).T,triad(config.ax1_jcl_strut)])
        self.Mbar_rbl_lower_strut_jcl_strut = multi_dot([A(config.P_rbl_lower_strut).T,triad(config.ax1_jcl_strut)])
        self.ubar_rbl_upper_strut_jcl_strut = (multi_dot([A(config.P_rbl_upper_strut).T,config.pt1_jcl_strut]) + (-1) * multi_dot([A(config.P_rbl_upper_strut).T,config.R_rbl_upper_strut]))
        self.ubar_rbl_lower_strut_jcl_strut = (multi_dot([A(config.P_rbl_lower_strut).T,config.pt1_jcl_strut]) + (-1) * multi_dot([A(config.P_rbl_lower_strut).T,config.R_rbl_lower_strut]))
        self.ubar_rbl_upper_strut_fal_strut = (multi_dot([A(config.P_rbl_upper_strut).T,config.pt1_fal_strut]) + (-1) * multi_dot([A(config.P_rbl_upper_strut).T,config.R_rbl_upper_strut]))
        self.ubar_rbl_lower_strut_fal_strut = (multi_dot([A(config.P_rbl_lower_strut).T,config.pt2_fal_strut]) + (-1) * multi_dot([A(config.P_rbl_lower_strut).T,config.R_rbl_lower_strut]))
        self.Mbar_rbr_lower_strut_jcr_strut_lca = multi_dot([A(config.P_rbr_lower_strut).T,triad(config.ax1_jcr_strut_lca)])
        self.Mbar_rbr_lca_jcr_strut_lca = multi_dot([A(config.P_rbr_lca).T,triad(config.ax2_jcr_strut_lca,triad(config.ax1_jcr_strut_lca)[0:3,1:2])])
        self.ubar_rbr_lower_strut_jcr_strut_lca = (multi_dot([A(config.P_rbr_lower_strut).T,config.pt1_jcr_strut_lca]) + (-1) * multi_dot([A(config.P_rbr_lower_strut).T,config.R_rbr_lower_strut]))
        self.ubar_rbr_lca_jcr_strut_lca = (multi_dot([A(config.P_rbr_lca).T,config.pt1_jcr_strut_lca]) + (-1) * multi_dot([A(config.P_rbr_lca).T,config.R_rbr_lca]))
        self.Mbar_rbl_lower_strut_jcl_strut_lca = multi_dot([A(config.P_rbl_lower_strut).T,triad(config.ax1_jcl_strut_lca)])
        self.Mbar_rbl_lca_jcl_strut_lca = multi_dot([A(config.P_rbl_lca).T,triad(config.ax2_jcl_strut_lca,triad(config.ax1_jcl_strut_lca)[0:3,1:2])])
        self.ubar_rbl_lower_strut_jcl_strut_lca = (multi_dot([A(config.P_rbl_lower_strut).T,config.pt1_jcl_strut_lca]) + (-1) * multi_dot([A(config.P_rbl_lower_strut).T,config.R_rbl_lower_strut]))
        self.ubar_rbl_lca_jcl_strut_lca = (multi_dot([A(config.P_rbl_lca).T,config.pt1_jcl_strut_lca]) + (-1) * multi_dot([A(config.P_rbl_lca).T,config.R_rbl_lca]))
        self.Mbar_rbr_tie_rod_jcr_tie_upright = multi_dot([A(config.P_rbr_tie_rod).T,triad(config.ax1_jcr_tie_upright)])
        self.Mbar_rbr_upright_jcr_tie_upright = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_tie_upright)])
        self.ubar_rbr_tie_rod_jcr_tie_upright = (multi_dot([A(config.P_rbr_tie_rod).T,config.pt1_jcr_tie_upright]) + (-1) * multi_dot([A(config.P_rbr_tie_rod).T,config.R_rbr_tie_rod]))
        self.ubar_rbr_upright_jcr_tie_upright = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_tie_upright]) + (-1) * multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.Mbar_rbr_tie_rod_jcr_tie_steering = multi_dot([A(config.P_rbr_tie_rod).T,triad(config.ax1_jcr_tie_steering)])
        self.Mbar_vbr_steer_jcr_tie_steering = multi_dot([A(config.P_vbr_steer).T,triad(config.ax2_jcr_tie_steering,triad(config.ax1_jcr_tie_steering)[0:3,1:2])])
        self.ubar_rbr_tie_rod_jcr_tie_steering = (multi_dot([A(config.P_rbr_tie_rod).T,config.pt1_jcr_tie_steering]) + (-1) * multi_dot([A(config.P_rbr_tie_rod).T,config.R_rbr_tie_rod]))
        self.ubar_vbr_steer_jcr_tie_steering = (multi_dot([A(config.P_vbr_steer).T,config.pt1_jcr_tie_steering]) + (-1) * multi_dot([A(config.P_vbr_steer).T,config.R_vbr_steer]))
        self.Mbar_rbl_tie_rod_jcl_tie_upright = multi_dot([A(config.P_rbl_tie_rod).T,triad(config.ax1_jcl_tie_upright)])
        self.Mbar_rbl_upright_jcl_tie_upright = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_tie_upright)])
        self.ubar_rbl_tie_rod_jcl_tie_upright = (multi_dot([A(config.P_rbl_tie_rod).T,config.pt1_jcl_tie_upright]) + (-1) * multi_dot([A(config.P_rbl_tie_rod).T,config.R_rbl_tie_rod]))
        self.ubar_rbl_upright_jcl_tie_upright = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_tie_upright]) + (-1) * multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.Mbar_rbl_tie_rod_jcl_tie_steering = multi_dot([A(config.P_rbl_tie_rod).T,triad(config.ax1_jcl_tie_steering)])
        self.Mbar_vbl_steer_jcl_tie_steering = multi_dot([A(config.P_vbl_steer).T,triad(config.ax2_jcl_tie_steering,triad(config.ax1_jcl_tie_steering)[0:3,1:2])])
        self.ubar_rbl_tie_rod_jcl_tie_steering = (multi_dot([A(config.P_rbl_tie_rod).T,config.pt1_jcl_tie_steering]) + (-1) * multi_dot([A(config.P_rbl_tie_rod).T,config.R_rbl_tie_rod]))
        self.ubar_vbl_steer_jcl_tie_steering = (multi_dot([A(config.P_vbl_steer).T,config.pt1_jcl_tie_steering]) + (-1) * multi_dot([A(config.P_vbl_steer).T,config.R_vbl_steer]))
        self.ubar_rbr_hub_far_tire = (multi_dot([A(config.P_rbr_hub).T,config.pt1_far_tire]) + (-1) * multi_dot([A(config.P_rbr_hub).T,config.R_rbr_hub]))
        self.ubar_vbs_ground_far_tire = (multi_dot([A(config.P_vbs_ground).T,config.pt1_far_tire]) + (-1) * multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))
        self.ubar_rbl_hub_fal_tire = (multi_dot([A(config.P_rbl_hub).T,config.pt1_fal_tire]) + (-1) * multi_dot([A(config.P_rbl_hub).T,config.R_rbl_hub]))
        self.ubar_vbs_ground_fal_tire = (multi_dot([A(config.P_vbs_ground).T,config.pt1_fal_tire]) + (-1) * multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))

    
    def _map_gen_coordinates(self):
        q = self._q
        self.R_rbr_uca = q[0:3]
        self.P_rbr_uca = q[3:7]
        self.R_rbl_uca = q[7:10]
        self.P_rbl_uca = q[10:14]
        self.R_rbr_lca = q[14:17]
        self.P_rbr_lca = q[17:21]
        self.R_rbl_lca = q[21:24]
        self.P_rbl_lca = q[24:28]
        self.R_rbr_upright = q[28:31]
        self.P_rbr_upright = q[31:35]
        self.R_rbl_upright = q[35:38]
        self.P_rbl_upright = q[38:42]
        self.R_rbr_upper_strut = q[42:45]
        self.P_rbr_upper_strut = q[45:49]
        self.R_rbl_upper_strut = q[49:52]
        self.P_rbl_upper_strut = q[52:56]
        self.R_rbr_lower_strut = q[56:59]
        self.P_rbr_lower_strut = q[59:63]
        self.R_rbl_lower_strut = q[63:66]
        self.P_rbl_lower_strut = q[66:70]
        self.R_rbr_tie_rod = q[70:73]
        self.P_rbr_tie_rod = q[73:77]
        self.R_rbl_tie_rod = q[77:80]
        self.P_rbl_tie_rod = q[80:84]
        self.R_rbr_hub = q[84:87]
        self.P_rbr_hub = q[87:91]
        self.R_rbl_hub = q[91:94]
        self.P_rbl_hub = q[94:98]

    
    def _map_gen_velocities(self):
        qd = self._qd
        self.Rd_rbr_uca = qd[0:3]
        self.Pd_rbr_uca = qd[3:7]
        self.Rd_rbl_uca = qd[7:10]
        self.Pd_rbl_uca = qd[10:14]
        self.Rd_rbr_lca = qd[14:17]
        self.Pd_rbr_lca = qd[17:21]
        self.Rd_rbl_lca = qd[21:24]
        self.Pd_rbl_lca = qd[24:28]
        self.Rd_rbr_upright = qd[28:31]
        self.Pd_rbr_upright = qd[31:35]
        self.Rd_rbl_upright = qd[35:38]
        self.Pd_rbl_upright = qd[38:42]
        self.Rd_rbr_upper_strut = qd[42:45]
        self.Pd_rbr_upper_strut = qd[45:49]
        self.Rd_rbl_upper_strut = qd[49:52]
        self.Pd_rbl_upper_strut = qd[52:56]
        self.Rd_rbr_lower_strut = qd[56:59]
        self.Pd_rbr_lower_strut = qd[59:63]
        self.Rd_rbl_lower_strut = qd[63:66]
        self.Pd_rbl_lower_strut = qd[66:70]
        self.Rd_rbr_tie_rod = qd[70:73]
        self.Pd_rbr_tie_rod = qd[73:77]
        self.Rd_rbl_tie_rod = qd[77:80]
        self.Pd_rbl_tie_rod = qd[80:84]
        self.Rd_rbr_hub = qd[84:87]
        self.Pd_rbr_hub = qd[87:91]
        self.Rd_rbl_hub = qd[91:94]
        self.Pd_rbl_hub = qd[94:98]

    
    def _map_gen_accelerations(self):
        qdd = self._qdd
        self.Rdd_rbr_uca = qdd[0:3]
        self.Pdd_rbr_uca = qdd[3:7]
        self.Rdd_rbl_uca = qdd[7:10]
        self.Pdd_rbl_uca = qdd[10:14]
        self.Rdd_rbr_lca = qdd[14:17]
        self.Pdd_rbr_lca = qdd[17:21]
        self.Rdd_rbl_lca = qdd[21:24]
        self.Pdd_rbl_lca = qdd[24:28]
        self.Rdd_rbr_upright = qdd[28:31]
        self.Pdd_rbr_upright = qdd[31:35]
        self.Rdd_rbl_upright = qdd[35:38]
        self.Pdd_rbl_upright = qdd[38:42]
        self.Rdd_rbr_upper_strut = qdd[42:45]
        self.Pdd_rbr_upper_strut = qdd[45:49]
        self.Rdd_rbl_upper_strut = qdd[49:52]
        self.Pdd_rbl_upper_strut = qdd[52:56]
        self.Rdd_rbr_lower_strut = qdd[56:59]
        self.Pdd_rbr_lower_strut = qdd[59:63]
        self.Rdd_rbl_lower_strut = qdd[63:66]
        self.Pdd_rbl_lower_strut = qdd[66:70]
        self.Rdd_rbr_tie_rod = qdd[70:73]
        self.Pdd_rbr_tie_rod = qdd[73:77]
        self.Rdd_rbl_tie_rod = qdd[77:80]
        self.Pdd_rbl_tie_rod = qdd[80:84]
        self.Rdd_rbr_hub = qdd[84:87]
        self.Pdd_rbr_hub = qdd[87:91]
        self.Rdd_rbl_hub = qdd[91:94]
        self.Pdd_rbl_hub = qdd[94:98]

    
    def _map_lagrange_multipliers(self):
        Lambda = self._lgr
        self.L_jcr_uca_upright = Lambda[0:3]
        self.L_jcr_uca_chassis = Lambda[3:8]
        self.L_jcl_uca_upright = Lambda[8:11]
        self.L_jcl_uca_chassis = Lambda[11:16]
        self.L_jcr_lca_upright = Lambda[16:19]
        self.L_jcr_lca_chassis = Lambda[19:24]
        self.L_jcl_lca_upright = Lambda[24:27]
        self.L_jcl_lca_chassis = Lambda[27:32]
        self.L_jcr_hub_bearing = Lambda[32:37]
        self.L_jcl_hub_bearing = Lambda[37:42]
        self.L_jcr_strut_chassis = Lambda[42:46]
        self.L_jcr_strut = Lambda[46:50]
        self.L_jcl_strut_chassis = Lambda[50:54]
        self.L_jcl_strut = Lambda[54:58]
        self.L_jcr_strut_lca = Lambda[58:62]
        self.L_jcl_strut_lca = Lambda[62:66]
        self.L_jcr_tie_upright = Lambda[66:69]
        self.L_jcr_tie_steering = Lambda[69:73]
        self.L_jcl_tie_upright = Lambda[73:76]
        self.L_jcl_tie_steering = Lambda[76:80]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_rbr_uca
        x1 = self.R_rbr_upright
        x2 = (-1) * x1
        x3 = self.P_rbr_uca
        x4 = A(x3)
        x5 = self.P_rbr_upright
        x6 = A(x5)
        x7 = (-1) * self.R_vbs_chassis
        x8 = A(self.P_vbs_chassis)
        x9 = x4.T
        x10 = self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        x11 = self.R_rbl_uca
        x12 = self.R_rbl_upright
        x13 = (-1) * x12
        x14 = self.P_rbl_uca
        x15 = A(x14)
        x16 = self.P_rbl_upright
        x17 = A(x16)
        x18 = x15.T
        x19 = self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        x20 = self.R_rbr_lca
        x21 = self.P_rbr_lca
        x22 = A(x21)
        x23 = x22.T
        x24 = self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        x25 = self.R_rbl_lca
        x26 = self.P_rbl_lca
        x27 = A(x26)
        x28 = x27.T
        x29 = self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        x30 = self.P_rbr_hub
        x31 = A(x30)
        x32 = x6.T
        x33 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        x34 = self.P_rbl_hub
        x35 = A(x34)
        x36 = x17.T
        x37 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        x38 = self.R_rbr_upper_strut
        x39 = self.P_rbr_upper_strut
        x40 = A(x39)
        x41 = x40.T
        x42 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1].T
        x43 = self.P_rbr_lower_strut
        x44 = A(x43)
        x45 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        x46 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2].T
        x47 = self.R_rbr_lower_strut
        x48 = (x38 + (-1) * x47 + multi_dot([x40,self.ubar_rbr_upper_strut_jcr_strut]) + (-1) * multi_dot([x44,self.ubar_rbr_lower_strut_jcr_strut]))
        x49 = self.R_rbl_upper_strut
        x50 = self.P_rbl_upper_strut
        x51 = A(x50)
        x52 = x51.T
        x53 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1].T
        x54 = self.P_rbl_lower_strut
        x55 = A(x54)
        x56 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        x57 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2].T
        x58 = self.R_rbl_lower_strut
        x59 = (x49 + (-1) * x58 + multi_dot([x51,self.ubar_rbl_upper_strut_jcl_strut]) + (-1) * multi_dot([x55,self.ubar_rbl_lower_strut_jcl_strut]))
        x60 = self.R_rbr_tie_rod
        x61 = self.P_rbr_tie_rod
        x62 = A(x61)
        x63 = A(self.P_vbr_steer)
        x64 = self.R_rbl_tie_rod
        x65 = self.P_rbl_tie_rod
        x66 = A(x65)
        x67 = A(self.P_vbl_steer)
        x68 = (-1) * I1

        self.pos_eq_blocks = ((x0 + x2 + multi_dot([x4,self.ubar_rbr_uca_jcr_uca_upright]) + (-1) * multi_dot([x6,self.ubar_rbr_upright_jcr_uca_upright])),
        (x0 + x7 + multi_dot([x4,self.ubar_rbr_uca_jcr_uca_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcr_uca_chassis])),
        multi_dot([self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1].T,x9,x8,x10]),
        multi_dot([self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2].T,x9,x8,x10]),
        (x11 + x13 + multi_dot([x15,self.ubar_rbl_uca_jcl_uca_upright]) + (-1) * multi_dot([x17,self.ubar_rbl_upright_jcl_uca_upright])),
        (x11 + x7 + multi_dot([x15,self.ubar_rbl_uca_jcl_uca_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcl_uca_chassis])),
        multi_dot([self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1].T,x18,x8,x19]),
        multi_dot([self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2].T,x18,x8,x19]),
        (x20 + x2 + multi_dot([x22,self.ubar_rbr_lca_jcr_lca_upright]) + (-1) * multi_dot([x6,self.ubar_rbr_upright_jcr_lca_upright])),
        (x20 + x7 + multi_dot([x22,self.ubar_rbr_lca_jcr_lca_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcr_lca_chassis])),
        multi_dot([self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1].T,x23,x8,x24]),
        multi_dot([self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2].T,x23,x8,x24]),
        (x25 + x13 + multi_dot([x27,self.ubar_rbl_lca_jcl_lca_upright]) + (-1) * multi_dot([x17,self.ubar_rbl_upright_jcl_lca_upright])),
        (x25 + x7 + multi_dot([x27,self.ubar_rbl_lca_jcl_lca_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcl_lca_chassis])),
        multi_dot([self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1].T,x28,x8,x29]),
        multi_dot([self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2].T,x28,x8,x29]),
        (x1 + (-1) * self.R_rbr_hub + multi_dot([x6,self.ubar_rbr_upright_jcr_hub_bearing]) + (-1) * multi_dot([x31,self.ubar_rbr_hub_jcr_hub_bearing])),
        multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x32,x31,x33]),
        multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x32,x31,x33]),
        (x12 + (-1) * self.R_rbl_hub + multi_dot([x17,self.ubar_rbl_upright_jcl_hub_bearing]) + (-1) * multi_dot([x35,self.ubar_rbl_hub_jcl_hub_bearing])),
        multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x36,x35,x37]),
        multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x36,x35,x37]),
        (x38 + x7 + multi_dot([x40,self.ubar_rbr_upper_strut_jcr_strut_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcr_strut_chassis])),
        multi_dot([self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1].T,x41,x8,self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]]),
        multi_dot([x42,x41,x44,x45]),
        multi_dot([x46,x41,x44,x45]),
        multi_dot([x42,x41,x48]),
        multi_dot([x46,x41,x48]),
        (x49 + x7 + multi_dot([x51,self.ubar_rbl_upper_strut_jcl_strut_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcl_strut_chassis])),
        multi_dot([self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1].T,x52,x8,self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]]),
        multi_dot([x53,x52,x55,x56]),
        multi_dot([x57,x52,x55,x56]),
        multi_dot([x53,x52,x59]),
        multi_dot([x57,x52,x59]),
        (x47 + (-1) * x20 + multi_dot([x44,self.ubar_rbr_lower_strut_jcr_strut_lca]) + (-1) * multi_dot([x22,self.ubar_rbr_lca_jcr_strut_lca])),
        multi_dot([self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1].T,x44.T,x22,self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]]),
        (x58 + (-1) * x25 + multi_dot([x55,self.ubar_rbl_lower_strut_jcl_strut_lca]) + (-1) * multi_dot([x27,self.ubar_rbl_lca_jcl_strut_lca])),
        multi_dot([self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1].T,x55.T,x27,self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]]),
        (x60 + x2 + multi_dot([x62,self.ubar_rbr_tie_rod_jcr_tie_upright]) + (-1) * multi_dot([x6,self.ubar_rbr_upright_jcr_tie_upright])),
        (x60 + (-1) * self.R_vbr_steer + multi_dot([x62,self.ubar_rbr_tie_rod_jcr_tie_steering]) + (-1) * multi_dot([x63,self.ubar_vbr_steer_jcr_tie_steering])),
        multi_dot([self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x62.T,x63,self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]]),
        (x64 + x13 + multi_dot([x66,self.ubar_rbl_tie_rod_jcl_tie_upright]) + (-1) * multi_dot([x17,self.ubar_rbl_upright_jcl_tie_upright])),
        (x64 + (-1) * self.R_vbl_steer + multi_dot([x66,self.ubar_rbl_tie_rod_jcl_tie_steering]) + (-1) * multi_dot([x67,self.ubar_vbl_steer_jcl_tie_steering])),
        multi_dot([self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1].T,x66.T,x67,self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]]),
        (x68 + multi_dot([x3.T,x3])),
        (x68 + multi_dot([x14.T,x14])),
        (x68 + multi_dot([x21.T,x21])),
        (x68 + multi_dot([x26.T,x26])),
        (x68 + multi_dot([x5.T,x5])),
        (x68 + multi_dot([x16.T,x16])),
        (x68 + multi_dot([x39.T,x39])),
        (x68 + multi_dot([x50.T,x50])),
        (x68 + multi_dot([x43.T,x43])),
        (x68 + multi_dot([x54.T,x54])),
        (x68 + multi_dot([x61.T,x61])),
        (x68 + multi_dot([x65.T,x65])),
        (x68 + multi_dot([x30.T,x30])),
        (x68 + multi_dot([x34.T,x34])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = Z3x1
        v1 = Z1x1

        self.vel_eq_blocks = (v0,
        v0,
        v1,
        v1,
        v0,
        v0,
        v1,
        v1,
        v0,
        v0,
        v1,
        v1,
        v0,
        v0,
        v1,
        v1,
        v0,
        v1,
        v1,
        v0,
        v1,
        v1,
        v0,
        v1,
        v1,
        v1,
        v1,
        v1,
        v0,
        v1,
        v1,
        v1,
        v1,
        v1,
        v0,
        v1,
        v0,
        v1,
        v0,
        v0,
        v1,
        v0,
        v0,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbr_uca
        a1 = self.Pd_rbr_upright
        a2 = self.Pd_vbs_chassis
        a3 = self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        a4 = self.P_rbr_uca
        a5 = A(a4).T
        a6 = self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        a7 = B(a2,a6)
        a8 = a6.T
        a9 = self.P_vbs_chassis
        a10 = A(a9).T
        a11 = a0.T
        a12 = B(a9,a6)
        a13 = self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        a14 = self.Pd_rbl_uca
        a15 = self.Pd_rbl_upright
        a16 = self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        a17 = a16.T
        a18 = self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        a19 = self.P_rbl_uca
        a20 = A(a19).T
        a21 = B(a2,a16)
        a22 = a14.T
        a23 = B(a9,a16)
        a24 = self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        a25 = self.Pd_rbr_lca
        a26 = self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        a27 = a26.T
        a28 = self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        a29 = self.P_rbr_lca
        a30 = A(a29).T
        a31 = B(a2,a26)
        a32 = a25.T
        a33 = B(a9,a26)
        a34 = self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        a35 = self.Pd_rbl_lca
        a36 = self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        a37 = self.P_rbl_lca
        a38 = A(a37).T
        a39 = self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        a40 = B(a2,a39)
        a41 = a39.T
        a42 = a35.T
        a43 = B(a9,a39)
        a44 = self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        a45 = self.Pd_rbr_hub
        a46 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        a47 = a46.T
        a48 = self.P_rbr_hub
        a49 = A(a48).T
        a50 = self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a51 = self.P_rbr_upright
        a52 = A(a51).T
        a53 = B(a45,a46)
        a54 = a1.T
        a55 = B(a48,a46)
        a56 = self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a57 = self.Pd_rbl_hub
        a58 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        a59 = a58.T
        a60 = self.P_rbl_hub
        a61 = A(a60).T
        a62 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a63 = self.P_rbl_upright
        a64 = A(a63).T
        a65 = B(a57,a58)
        a66 = a15.T
        a67 = B(a60,a58)
        a68 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a69 = self.Pd_rbr_upper_strut
        a70 = self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        a71 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        a72 = self.P_rbr_upper_strut
        a73 = A(a72).T
        a74 = a69.T
        a75 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        a76 = a75.T
        a77 = self.P_rbr_lower_strut
        a78 = A(a77).T
        a79 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        a80 = B(a69,a79)
        a81 = a79.T
        a82 = self.Pd_rbr_lower_strut
        a83 = B(a82,a75)
        a84 = B(a72,a79).T
        a85 = B(a77,a75)
        a86 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        a87 = B(a69,a86)
        a88 = a86.T
        a89 = B(a72,a86).T
        a90 = self.ubar_rbr_upper_strut_jcr_strut
        a91 = self.ubar_rbr_lower_strut_jcr_strut
        a92 = (multi_dot([B(a69,a90),a69]) + (-1) * multi_dot([B(a82,a91),a82]))
        a93 = (self.Rd_rbr_upper_strut + (-1) * self.Rd_rbr_lower_strut + multi_dot([B(a72,a90),a69]) + (-1) * multi_dot([B(a77,a91),a82]))
        a94 = (self.R_rbr_upper_strut.T + (-1) * self.R_rbr_lower_strut.T + multi_dot([a90.T,a73]) + (-1) * multi_dot([a91.T,a78]))
        a95 = self.Pd_rbl_upper_strut
        a96 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        a97 = self.P_rbl_upper_strut
        a98 = A(a97).T
        a99 = self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        a100 = a95.T
        a101 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        a102 = a101.T
        a103 = self.Pd_rbl_lower_strut
        a104 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        a105 = B(a103,a104)
        a106 = a104.T
        a107 = self.P_rbl_lower_strut
        a108 = A(a107).T
        a109 = B(a95,a101)
        a110 = B(a97,a101).T
        a111 = B(a107,a104)
        a112 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        a113 = a112.T
        a114 = B(a95,a112)
        a115 = B(a97,a112).T
        a116 = self.ubar_rbl_upper_strut_jcl_strut
        a117 = self.ubar_rbl_lower_strut_jcl_strut
        a118 = (multi_dot([B(a95,a116),a95]) + (-1) * multi_dot([B(a103,a117),a103]))
        a119 = (self.Rd_rbl_upper_strut + (-1) * self.Rd_rbl_lower_strut + multi_dot([B(a97,a116),a95]) + (-1) * multi_dot([B(a107,a117),a103]))
        a120 = (self.R_rbl_upper_strut.T + (-1) * self.R_rbl_lower_strut.T + multi_dot([a116.T,a98]) + (-1) * multi_dot([a117.T,a108]))
        a121 = self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a122 = self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        a123 = a82.T
        a124 = self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a125 = self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        a126 = a103.T
        a127 = self.Pd_rbr_tie_rod
        a128 = self.Pd_vbr_steer
        a129 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        a130 = self.P_rbr_tie_rod
        a131 = self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        a132 = self.P_vbr_steer
        a133 = a127.T
        a134 = self.Pd_rbl_tie_rod
        a135 = self.Pd_vbl_steer
        a136 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        a137 = self.P_rbl_tie_rod
        a138 = self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        a139 = self.P_vbl_steer
        a140 = a134.T

        self.acc_eq_blocks = ((multi_dot([B(a0,self.ubar_rbr_uca_jcr_uca_upright),a0]) + (-1) * multi_dot([B(a1,self.ubar_rbr_upright_jcr_uca_upright),a1])),
        (multi_dot([B(a0,self.ubar_rbr_uca_jcr_uca_chassis),a0]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcr_uca_chassis),a2])),
        (multi_dot([a3.T,a5,a7,a2]) + multi_dot([a8,a10,B(a0,a3),a0]) + (2) * multi_dot([a11,B(a4,a3).T,a12,a2])),
        (multi_dot([a13.T,a5,a7,a2]) + multi_dot([a8,a10,B(a0,a13),a0]) + (2) * multi_dot([a11,B(a4,a13).T,a12,a2])),
        (multi_dot([B(a14,self.ubar_rbl_uca_jcl_uca_upright),a14]) + (-1) * multi_dot([B(a15,self.ubar_rbl_upright_jcl_uca_upright),a15])),
        (multi_dot([B(a14,self.ubar_rbl_uca_jcl_uca_chassis),a14]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcl_uca_chassis),a2])),
        (multi_dot([a17,a10,B(a14,a18),a14]) + multi_dot([a18.T,a20,a21,a2]) + (2) * multi_dot([a22,B(a19,a18).T,a23,a2])),
        (multi_dot([a17,a10,B(a14,a24),a14]) + multi_dot([a24.T,a20,a21,a2]) + (2) * multi_dot([a22,B(a19,a24).T,a23,a2])),
        (multi_dot([B(a25,self.ubar_rbr_lca_jcr_lca_upright),a25]) + (-1) * multi_dot([B(a1,self.ubar_rbr_upright_jcr_lca_upright),a1])),
        (multi_dot([B(a25,self.ubar_rbr_lca_jcr_lca_chassis),a25]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcr_lca_chassis),a2])),
        (multi_dot([a27,a10,B(a25,a28),a25]) + multi_dot([a28.T,a30,a31,a2]) + (2) * multi_dot([a32,B(a29,a28).T,a33,a2])),
        (multi_dot([a27,a10,B(a25,a34),a25]) + multi_dot([a34.T,a30,a31,a2]) + (2) * multi_dot([a32,B(a29,a34).T,a33,a2])),
        (multi_dot([B(a35,self.ubar_rbl_lca_jcl_lca_upright),a35]) + (-1) * multi_dot([B(a15,self.ubar_rbl_upright_jcl_lca_upright),a15])),
        (multi_dot([B(a35,self.ubar_rbl_lca_jcl_lca_chassis),a35]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcl_lca_chassis),a2])),
        (multi_dot([a36.T,a38,a40,a2]) + multi_dot([a41,a10,B(a35,a36),a35]) + (2) * multi_dot([a42,B(a37,a36).T,a43,a2])),
        (multi_dot([a44.T,a38,a40,a2]) + multi_dot([a41,a10,B(a35,a44),a35]) + (2) * multi_dot([a42,B(a37,a44).T,a43,a2])),
        (multi_dot([B(a1,self.ubar_rbr_upright_jcr_hub_bearing),a1]) + (-1) * multi_dot([B(a45,self.ubar_rbr_hub_jcr_hub_bearing),a45])),
        (multi_dot([a47,a49,B(a1,a50),a1]) + multi_dot([a50.T,a52,a53,a45]) + (2) * multi_dot([a54,B(a51,a50).T,a55,a45])),
        (multi_dot([a47,a49,B(a1,a56),a1]) + multi_dot([a56.T,a52,a53,a45]) + (2) * multi_dot([a54,B(a51,a56).T,a55,a45])),
        (multi_dot([B(a15,self.ubar_rbl_upright_jcl_hub_bearing),a15]) + (-1) * multi_dot([B(a57,self.ubar_rbl_hub_jcl_hub_bearing),a57])),
        (multi_dot([a59,a61,B(a15,a62),a15]) + multi_dot([a62.T,a64,a65,a57]) + (2) * multi_dot([a66,B(a63,a62).T,a67,a57])),
        (multi_dot([a59,a61,B(a15,a68),a15]) + multi_dot([a68.T,a64,a65,a57]) + (2) * multi_dot([a66,B(a63,a68).T,a67,a57])),
        (multi_dot([B(a69,self.ubar_rbr_upper_strut_jcr_strut_chassis),a69]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcr_strut_chassis),a2])),
        (multi_dot([a70.T,a10,B(a69,a71),a69]) + multi_dot([a71.T,a73,B(a2,a70),a2]) + (2) * multi_dot([a74,B(a72,a71).T,B(a9,a70),a2])),
        (multi_dot([a76,a78,a80,a69]) + multi_dot([a81,a73,a83,a82]) + (2) * multi_dot([a74,a84,a85,a82])),
        (multi_dot([a76,a78,a87,a69]) + multi_dot([a88,a73,a83,a82]) + (2) * multi_dot([a74,a89,a85,a82])),
        (multi_dot([a81,a73,a92]) + (2) * multi_dot([a74,a84,a93]) + multi_dot([a94,a80,a69])),
        (multi_dot([a88,a73,a92]) + (2) * multi_dot([a74,a89,a93]) + multi_dot([a94,a87,a69])),
        (multi_dot([B(a95,self.ubar_rbl_upper_strut_jcl_strut_chassis),a95]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcl_strut_chassis),a2])),
        (multi_dot([a96.T,a98,B(a2,a99),a2]) + multi_dot([a99.T,a10,B(a95,a96),a95]) + (2) * multi_dot([a100,B(a97,a96).T,B(a9,a99),a2])),
        (multi_dot([a102,a98,a105,a103]) + multi_dot([a106,a108,a109,a95]) + (2) * multi_dot([a100,a110,a111,a103])),
        (multi_dot([a113,a98,a105,a103]) + multi_dot([a106,a108,a114,a95]) + (2) * multi_dot([a100,a115,a111,a103])),
        (multi_dot([a102,a98,a118]) + (2) * multi_dot([a100,a110,a119]) + multi_dot([a120,a109,a95])),
        (multi_dot([a113,a98,a118]) + (2) * multi_dot([a100,a115,a119]) + multi_dot([a120,a114,a95])),
        (multi_dot([B(a82,self.ubar_rbr_lower_strut_jcr_strut_lca),a82]) + (-1) * multi_dot([B(a25,self.ubar_rbr_lca_jcr_strut_lca),a25])),
        (multi_dot([a121.T,a78,B(a25,a122),a25]) + multi_dot([a122.T,a30,B(a82,a121),a82]) + (2) * multi_dot([a123,B(a77,a121).T,B(a29,a122),a25])),
        (multi_dot([B(a103,self.ubar_rbl_lower_strut_jcl_strut_lca),a103]) + (-1) * multi_dot([B(a35,self.ubar_rbl_lca_jcl_strut_lca),a35])),
        (multi_dot([a124.T,a108,B(a35,a125),a35]) + multi_dot([a125.T,a38,B(a103,a124),a103]) + (2) * multi_dot([a126,B(a107,a124).T,B(a37,a125),a35])),
        (multi_dot([B(a127,self.ubar_rbr_tie_rod_jcr_tie_upright),a127]) + (-1) * multi_dot([B(a1,self.ubar_rbr_upright_jcr_tie_upright),a1])),
        (multi_dot([B(a127,self.ubar_rbr_tie_rod_jcr_tie_steering),a127]) + (-1) * multi_dot([B(a128,self.ubar_vbr_steer_jcr_tie_steering),a128])),
        (multi_dot([a129.T,A(a130).T,B(a128,a131),a128]) + multi_dot([a131.T,A(a132).T,B(a127,a129),a127]) + (2) * multi_dot([a133,B(a130,a129).T,B(a132,a131),a128])),
        (multi_dot([B(a134,self.ubar_rbl_tie_rod_jcl_tie_upright),a134]) + (-1) * multi_dot([B(a15,self.ubar_rbl_upright_jcl_tie_upright),a15])),
        (multi_dot([B(a134,self.ubar_rbl_tie_rod_jcl_tie_steering),a134]) + (-1) * multi_dot([B(a135,self.ubar_vbl_steer_jcl_tie_steering),a135])),
        (multi_dot([a136.T,A(a137).T,B(a135,a138),a135]) + multi_dot([a138.T,A(a139).T,B(a134,a136),a134]) + (2) * multi_dot([a140,B(a137,a136).T,B(a139,a138),a135])),
        (2) * multi_dot([a11,a0]),
        (2) * multi_dot([a22,a14]),
        (2) * multi_dot([a32,a25]),
        (2) * multi_dot([a42,a35]),
        (2) * multi_dot([a54,a1]),
        (2) * multi_dot([a66,a15]),
        (2) * multi_dot([a74,a69]),
        (2) * multi_dot([a100,a95]),
        (2) * multi_dot([a123,a82]),
        (2) * multi_dot([a126,a103]),
        (2) * multi_dot([a133,a127]),
        (2) * multi_dot([a140,a134]),
        (2) * multi_dot([a45.T,a45]),
        (2) * multi_dot([a57.T,a57]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = I3
        j1 = self.P_rbr_uca
        j2 = (-1) * j0
        j3 = self.P_rbr_upright
        j4 = Z1x3
        j5 = self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        j6 = j5.T
        j7 = self.P_vbs_chassis
        j8 = A(j7).T
        j9 = self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        j10 = self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        j11 = A(j1).T
        j12 = B(j7,j5)
        j13 = self.P_rbl_uca
        j14 = self.P_rbl_upright
        j15 = self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        j16 = j15.T
        j17 = self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        j18 = self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        j19 = A(j13).T
        j20 = B(j7,j15)
        j21 = self.P_rbr_lca
        j22 = self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        j23 = j22.T
        j24 = self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        j25 = self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        j26 = A(j21).T
        j27 = B(j7,j22)
        j28 = self.P_rbl_lca
        j29 = self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        j30 = j29.T
        j31 = self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        j32 = self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        j33 = A(j28).T
        j34 = B(j7,j29)
        j35 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        j36 = j35.T
        j37 = self.P_rbr_hub
        j38 = A(j37).T
        j39 = self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j40 = self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j41 = A(j3).T
        j42 = B(j37,j35)
        j43 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        j44 = j43.T
        j45 = self.P_rbl_hub
        j46 = A(j45).T
        j47 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j48 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j49 = A(j14).T
        j50 = B(j45,j43)
        j51 = self.P_rbr_upper_strut
        j52 = self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        j53 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        j54 = A(j51).T
        j55 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        j56 = j55.T
        j57 = self.P_rbr_lower_strut
        j58 = A(j57).T
        j59 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        j60 = B(j51,j59)
        j61 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        j62 = B(j51,j61)
        j63 = j59.T
        j64 = multi_dot([j63,j54])
        j65 = self.ubar_rbr_upper_strut_jcr_strut
        j66 = B(j51,j65)
        j67 = self.ubar_rbr_lower_strut_jcr_strut
        j68 = (self.R_rbr_upper_strut.T + (-1) * self.R_rbr_lower_strut.T + multi_dot([j65.T,j54]) + (-1) * multi_dot([j67.T,j58]))
        j69 = j61.T
        j70 = multi_dot([j69,j54])
        j71 = B(j57,j55)
        j72 = B(j57,j67)
        j73 = self.P_rbl_upper_strut
        j74 = self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        j75 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        j76 = A(j73).T
        j77 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        j78 = j77.T
        j79 = self.P_rbl_lower_strut
        j80 = A(j79).T
        j81 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        j82 = B(j73,j81)
        j83 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        j84 = B(j73,j83)
        j85 = j81.T
        j86 = multi_dot([j85,j76])
        j87 = self.ubar_rbl_upper_strut_jcl_strut
        j88 = B(j73,j87)
        j89 = self.ubar_rbl_lower_strut_jcl_strut
        j90 = (self.R_rbl_upper_strut.T + (-1) * self.R_rbl_lower_strut.T + multi_dot([j87.T,j76]) + (-1) * multi_dot([j89.T,j80]))
        j91 = j83.T
        j92 = multi_dot([j91,j76])
        j93 = B(j79,j77)
        j94 = B(j79,j89)
        j95 = self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        j96 = self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j97 = self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        j98 = self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j99 = self.P_rbr_tie_rod
        j100 = self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        j101 = self.P_vbr_steer
        j102 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        j103 = self.P_rbl_tie_rod
        j104 = self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        j105 = self.P_vbl_steer
        j106 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]

        self.jac_eq_blocks = (j0,
        B(j1,self.ubar_rbr_uca_jcr_uca_upright),
        j2,
        (-1) * B(j3,self.ubar_rbr_upright_jcr_uca_upright),
        j0,
        B(j1,self.ubar_rbr_uca_jcr_uca_chassis),
        j2,
        (-1) * B(j7,self.ubar_vbs_chassis_jcr_uca_chassis),
        j4,
        multi_dot([j6,j8,B(j1,j9)]),
        j4,
        multi_dot([j9.T,j11,j12]),
        j4,
        multi_dot([j6,j8,B(j1,j10)]),
        j4,
        multi_dot([j10.T,j11,j12]),
        j0,
        B(j13,self.ubar_rbl_uca_jcl_uca_upright),
        j2,
        (-1) * B(j14,self.ubar_rbl_upright_jcl_uca_upright),
        j0,
        B(j13,self.ubar_rbl_uca_jcl_uca_chassis),
        j2,
        (-1) * B(j7,self.ubar_vbs_chassis_jcl_uca_chassis),
        j4,
        multi_dot([j16,j8,B(j13,j17)]),
        j4,
        multi_dot([j17.T,j19,j20]),
        j4,
        multi_dot([j16,j8,B(j13,j18)]),
        j4,
        multi_dot([j18.T,j19,j20]),
        j0,
        B(j21,self.ubar_rbr_lca_jcr_lca_upright),
        j2,
        (-1) * B(j3,self.ubar_rbr_upright_jcr_lca_upright),
        j0,
        B(j21,self.ubar_rbr_lca_jcr_lca_chassis),
        j2,
        (-1) * B(j7,self.ubar_vbs_chassis_jcr_lca_chassis),
        j4,
        multi_dot([j23,j8,B(j21,j24)]),
        j4,
        multi_dot([j24.T,j26,j27]),
        j4,
        multi_dot([j23,j8,B(j21,j25)]),
        j4,
        multi_dot([j25.T,j26,j27]),
        j0,
        B(j28,self.ubar_rbl_lca_jcl_lca_upright),
        j2,
        (-1) * B(j14,self.ubar_rbl_upright_jcl_lca_upright),
        j0,
        B(j28,self.ubar_rbl_lca_jcl_lca_chassis),
        j2,
        (-1) * B(j7,self.ubar_vbs_chassis_jcl_lca_chassis),
        j4,
        multi_dot([j30,j8,B(j28,j31)]),
        j4,
        multi_dot([j31.T,j33,j34]),
        j4,
        multi_dot([j30,j8,B(j28,j32)]),
        j4,
        multi_dot([j32.T,j33,j34]),
        j0,
        B(j3,self.ubar_rbr_upright_jcr_hub_bearing),
        j2,
        (-1) * B(j37,self.ubar_rbr_hub_jcr_hub_bearing),
        j4,
        multi_dot([j36,j38,B(j3,j39)]),
        j4,
        multi_dot([j39.T,j41,j42]),
        j4,
        multi_dot([j36,j38,B(j3,j40)]),
        j4,
        multi_dot([j40.T,j41,j42]),
        j0,
        B(j14,self.ubar_rbl_upright_jcl_hub_bearing),
        j2,
        (-1) * B(j45,self.ubar_rbl_hub_jcl_hub_bearing),
        j4,
        multi_dot([j44,j46,B(j14,j47)]),
        j4,
        multi_dot([j47.T,j49,j50]),
        j4,
        multi_dot([j44,j46,B(j14,j48)]),
        j4,
        multi_dot([j48.T,j49,j50]),
        j0,
        B(j51,self.ubar_rbr_upper_strut_jcr_strut_chassis),
        j2,
        (-1) * B(j7,self.ubar_vbs_chassis_jcr_strut_chassis),
        j4,
        multi_dot([j52.T,j8,B(j51,j53)]),
        j4,
        multi_dot([j53.T,j54,B(j7,j52)]),
        j4,
        multi_dot([j56,j58,j60]),
        j4,
        multi_dot([j63,j54,j71]),
        j4,
        multi_dot([j56,j58,j62]),
        j4,
        multi_dot([j69,j54,j71]),
        j64,
        (multi_dot([j63,j54,j66]) + multi_dot([j68,j60])),
        (-1) * j64,
        (-1) * multi_dot([j63,j54,j72]),
        j70,
        (multi_dot([j69,j54,j66]) + multi_dot([j68,j62])),
        (-1) * j70,
        (-1) * multi_dot([j69,j54,j72]),
        j0,
        B(j73,self.ubar_rbl_upper_strut_jcl_strut_chassis),
        j2,
        (-1) * B(j7,self.ubar_vbs_chassis_jcl_strut_chassis),
        j4,
        multi_dot([j74.T,j8,B(j73,j75)]),
        j4,
        multi_dot([j75.T,j76,B(j7,j74)]),
        j4,
        multi_dot([j78,j80,j82]),
        j4,
        multi_dot([j85,j76,j93]),
        j4,
        multi_dot([j78,j80,j84]),
        j4,
        multi_dot([j91,j76,j93]),
        j86,
        (multi_dot([j85,j76,j88]) + multi_dot([j90,j82])),
        (-1) * j86,
        (-1) * multi_dot([j85,j76,j94]),
        j92,
        (multi_dot([j91,j76,j88]) + multi_dot([j90,j84])),
        (-1) * j92,
        (-1) * multi_dot([j91,j76,j94]),
        j2,
        (-1) * B(j21,self.ubar_rbr_lca_jcr_strut_lca),
        j0,
        B(j57,self.ubar_rbr_lower_strut_jcr_strut_lca),
        j4,
        multi_dot([j96.T,j58,B(j21,j95)]),
        j4,
        multi_dot([j95.T,j26,B(j57,j96)]),
        j2,
        (-1) * B(j28,self.ubar_rbl_lca_jcl_strut_lca),
        j0,
        B(j79,self.ubar_rbl_lower_strut_jcl_strut_lca),
        j4,
        multi_dot([j98.T,j80,B(j28,j97)]),
        j4,
        multi_dot([j97.T,j33,B(j79,j98)]),
        j2,
        (-1) * B(j3,self.ubar_rbr_upright_jcr_tie_upright),
        j0,
        B(j99,self.ubar_rbr_tie_rod_jcr_tie_upright),
        j0,
        B(j99,self.ubar_rbr_tie_rod_jcr_tie_steering),
        j2,
        (-1) * B(j101,self.ubar_vbr_steer_jcr_tie_steering),
        j4,
        multi_dot([j100.T,A(j101).T,B(j99,j102)]),
        j4,
        multi_dot([j102.T,A(j99).T,B(j101,j100)]),
        j2,
        (-1) * B(j14,self.ubar_rbl_upright_jcl_tie_upright),
        j0,
        B(j103,self.ubar_rbl_tie_rod_jcl_tie_upright),
        j0,
        B(j103,self.ubar_rbl_tie_rod_jcl_tie_steering),
        j2,
        (-1) * B(j105,self.ubar_vbl_steer_jcl_tie_steering),
        j4,
        multi_dot([j104.T,A(j105).T,B(j103,j106)]),
        j4,
        multi_dot([j106.T,A(j103).T,B(j105,j104)]),
        j4,
        (2) * j1.T,
        j4,
        (2) * j13.T,
        j4,
        (2) * j21.T,
        j4,
        (2) * j28.T,
        j4,
        (2) * j3.T,
        j4,
        (2) * j14.T,
        j4,
        (2) * j51.T,
        j4,
        (2) * j73.T,
        j4,
        (2) * j57.T,
        j4,
        (2) * j79.T,
        j4,
        (2) * j99.T,
        j4,
        (2) * j103.T,
        j4,
        (2) * j37.T,
        j4,
        (2) * j45.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = I3
        m1 = G(self.P_rbr_uca)
        m2 = G(self.P_rbl_uca)
        m3 = G(self.P_rbr_lca)
        m4 = G(self.P_rbl_lca)
        m5 = G(self.P_rbr_upright)
        m6 = G(self.P_rbl_upright)
        m7 = G(self.P_rbr_upper_strut)
        m8 = G(self.P_rbl_upper_strut)
        m9 = G(self.P_rbr_lower_strut)
        m10 = G(self.P_rbl_lower_strut)
        m11 = G(self.P_rbr_tie_rod)
        m12 = G(self.P_rbl_tie_rod)
        m13 = G(self.P_rbr_hub)
        m14 = G(self.P_rbl_hub)

        self.mass_eq_blocks = (config.m_rbr_uca * m0,
        (4) * multi_dot([m1.T,config.Jbar_rbr_uca,m1]),
        config.m_rbl_uca * m0,
        (4) * multi_dot([m2.T,config.Jbar_rbl_uca,m2]),
        config.m_rbr_lca * m0,
        (4) * multi_dot([m3.T,config.Jbar_rbr_lca,m3]),
        config.m_rbl_lca * m0,
        (4) * multi_dot([m4.T,config.Jbar_rbl_lca,m4]),
        config.m_rbr_upright * m0,
        (4) * multi_dot([m5.T,config.Jbar_rbr_upright,m5]),
        config.m_rbl_upright * m0,
        (4) * multi_dot([m6.T,config.Jbar_rbl_upright,m6]),
        config.m_rbr_upper_strut * m0,
        (4) * multi_dot([m7.T,config.Jbar_rbr_upper_strut,m7]),
        config.m_rbl_upper_strut * m0,
        (4) * multi_dot([m8.T,config.Jbar_rbl_upper_strut,m8]),
        config.m_rbr_lower_strut * m0,
        (4) * multi_dot([m9.T,config.Jbar_rbr_lower_strut,m9]),
        config.m_rbl_lower_strut * m0,
        (4) * multi_dot([m10.T,config.Jbar_rbl_lower_strut,m10]),
        config.m_rbr_tie_rod * m0,
        (4) * multi_dot([m11.T,config.Jbar_rbr_tie_rod,m11]),
        config.m_rbl_tie_rod * m0,
        (4) * multi_dot([m12.T,config.Jbar_rbl_tie_rod,m12]),
        config.m_rbr_hub * m0,
        (4) * multi_dot([m13.T,config.Jbar_rbr_hub,m13]),
        config.m_rbl_hub * m0,
        (4) * multi_dot([m14.T,config.Jbar_rbl_hub,m14]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbr_uca)
        f1 = G(self.Pd_rbl_uca)
        f2 = G(self.Pd_rbr_lca)
        f3 = G(self.Pd_rbl_lca)
        f4 = G(self.Pd_rbr_upright)
        f5 = G(self.Pd_rbl_upright)
        f6 = self.R_rbr_upper_strut
        f7 = self.R_rbr_lower_strut
        f8 = self.ubar_rbr_upper_strut_far_strut
        f9 = self.P_rbr_upper_strut
        f10 = A(f9)
        f11 = self.ubar_rbr_lower_strut_far_strut
        f12 = self.P_rbr_lower_strut
        f13 = A(f12)
        f14 = (f6.T + (-1) * f7.T + multi_dot([f8.T,f10.T]) + (-1) * multi_dot([f11.T,f13.T]))
        f15 = multi_dot([f10,f8])
        f16 = multi_dot([f13,f11])
        f17 = (f6 + (-1) * f7 + f15 + (-1) * f16)
        f18 = ((multi_dot([f14,f17]))**(1.0/2.0))[0]
        f19 = 1.0/f18
        f20 = config.UF_far_strut_Fs((config.far_strut_FL + (-1 * f18)))
        f21 = self.Pd_rbr_upper_strut
        f22 = self.Pd_rbr_lower_strut
        f23 = config.UF_far_strut_Fd((-1 * 1.0/f18) * multi_dot([f14,(self.Rd_rbr_upper_strut + (-1) * self.Rd_rbr_lower_strut + multi_dot([B(f9,f8),f21]) + (-1) * multi_dot([B(f12,f11),f22]))]))
        f24 = (f19 * (f20 + f23)) * f17
        f25 = G(f21)
        f26 = (2 * f20)
        f27 = (2 * f23)
        f28 = self.R_rbl_upper_strut
        f29 = self.R_rbl_lower_strut
        f30 = self.ubar_rbl_upper_strut_fal_strut
        f31 = self.P_rbl_upper_strut
        f32 = A(f31)
        f33 = self.ubar_rbl_lower_strut_fal_strut
        f34 = self.P_rbl_lower_strut
        f35 = A(f34)
        f36 = (f28.T + (-1) * f29.T + multi_dot([f30.T,f32.T]) + (-1) * multi_dot([f33.T,f35.T]))
        f37 = multi_dot([f32,f30])
        f38 = multi_dot([f35,f33])
        f39 = (f28 + (-1) * f29 + f37 + (-1) * f38)
        f40 = ((multi_dot([f36,f39]))**(1.0/2.0))[0]
        f41 = 1.0/f40
        f42 = config.UF_fal_strut_Fs((config.fal_strut_FL + (-1 * f40)))
        f43 = self.Pd_rbl_upper_strut
        f44 = self.Pd_rbl_lower_strut
        f45 = config.UF_fal_strut_Fd((-1 * 1.0/f40) * multi_dot([f36,(self.Rd_rbl_upper_strut + (-1) * self.Rd_rbl_lower_strut + multi_dot([B(f31,f30),f43]) + (-1) * multi_dot([B(f34,f33),f44]))]))
        f46 = (f41 * (f42 + f45)) * f39
        f47 = G(f43)
        f48 = (2 * f42)
        f49 = (2 * f45)
        f50 = Z3x1
        f51 = Z4x1
        f52 = G(f22)
        f53 = G(f44)
        f54 = G(self.Pd_rbr_tie_rod)
        f55 = G(self.Pd_rbl_tie_rod)
        f56 = t
        f57 = config.UF_far_tire_F(f56)
        f58 = G(self.Pd_rbr_hub)
        f59 = self.P_rbr_hub
        f60 = config.UF_fal_tire_F(f56)
        f61 = G(self.Pd_rbl_hub)
        f62 = self.P_rbl_hub

        self.frc_eq_blocks = (self.F_rbr_uca_gravity,
        (8) * multi_dot([f0.T,config.Jbar_rbr_uca,f0,self.P_rbr_uca]),
        self.F_rbl_uca_gravity,
        (8) * multi_dot([f1.T,config.Jbar_rbl_uca,f1,self.P_rbl_uca]),
        self.F_rbr_lca_gravity,
        (8) * multi_dot([f2.T,config.Jbar_rbr_lca,f2,self.P_rbr_lca]),
        self.F_rbl_lca_gravity,
        (8) * multi_dot([f3.T,config.Jbar_rbl_lca,f3,self.P_rbl_lca]),
        self.F_rbr_upright_gravity,
        (8) * multi_dot([f4.T,config.Jbar_rbr_upright,f4,self.P_rbr_upright]),
        self.F_rbl_upright_gravity,
        (8) * multi_dot([f5.T,config.Jbar_rbl_upright,f5,self.P_rbl_upright]),
        (self.F_rbr_upper_strut_gravity + f24),
        ((8) * multi_dot([f25.T,config.Jbar_rbr_upper_strut,f25,f9]) + (f19 * ((-1 * f26) + (-1 * f27))) * multi_dot([E(f9).T,skew(f15).T,f17])),
        (self.F_rbl_upper_strut_gravity + f46),
        ((8) * multi_dot([f47.T,config.Jbar_rbl_upper_strut,f47,f31]) + (f41 * ((-1 * f48) + (-1 * f49))) * multi_dot([E(f31).T,skew(f37).T,f39])),
        (self.F_rbr_lower_strut_gravity + f50 + (-1) * f24),
        (f51 + (8) * multi_dot([f52.T,config.Jbar_rbr_lower_strut,f52,f12]) + (f19 * (f26 + f27)) * multi_dot([E(f12).T,skew(f16).T,f17])),
        (self.F_rbl_lower_strut_gravity + f50 + (-1) * f46),
        (f51 + (8) * multi_dot([f53.T,config.Jbar_rbl_lower_strut,f53,f34]) + (f41 * (f48 + f49)) * multi_dot([E(f34).T,skew(f38).T,f39])),
        self.F_rbr_tie_rod_gravity,
        (8) * multi_dot([f54.T,config.Jbar_rbr_tie_rod,f54,self.P_rbr_tie_rod]),
        self.F_rbl_tie_rod_gravity,
        (8) * multi_dot([f55.T,config.Jbar_rbl_tie_rod,f55,self.P_rbl_tie_rod]),
        (self.F_rbr_hub_gravity + f57),
        ((8) * multi_dot([f58.T,config.Jbar_rbr_hub,f58,f59]) + (2) * multi_dot([E(f59).T,(config.UF_far_tire_T(f56) + multi_dot([skew(multi_dot([A(f59),self.ubar_rbr_hub_far_tire])),f57]))])),
        (self.F_rbl_hub_gravity + f60),
        ((8) * multi_dot([f61.T,config.Jbar_rbl_hub,f61,f62]) + (2) * multi_dot([E(f62).T,(config.UF_fal_tire_T(f56) + multi_dot([skew(multi_dot([A(f62),self.ubar_rbl_hub_fal_tire])),f60]))])),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbr_uca_jcr_uca_upright = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbr_uca,self.ubar_rbr_uca_jcr_uca_upright).T]]),self.L_jcr_uca_upright])
        self.F_rbr_uca_jcr_uca_upright = Q_rbr_uca_jcr_uca_upright[0:3]
        Te_rbr_uca_jcr_uca_upright = Q_rbr_uca_jcr_uca_upright[3:7]
        self.T_rbr_uca_jcr_uca_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_uca),self.ubar_rbr_uca_jcr_uca_upright])),self.F_rbr_uca_jcr_uca_upright]) + (0.5) * multi_dot([E(self.P_rbr_uca),Te_rbr_uca_jcr_uca_upright]))
        Q_rbr_uca_jcr_uca_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbr_uca,self.ubar_rbr_uca_jcr_uca_chassis).T,multi_dot([B(self.P_rbr_uca,self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]]),multi_dot([B(self.P_rbr_uca,self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]])]]),self.L_jcr_uca_chassis])
        self.F_rbr_uca_jcr_uca_chassis = Q_rbr_uca_jcr_uca_chassis[0:3]
        Te_rbr_uca_jcr_uca_chassis = Q_rbr_uca_jcr_uca_chassis[3:7]
        self.T_rbr_uca_jcr_uca_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_uca),self.ubar_rbr_uca_jcr_uca_chassis])),self.F_rbr_uca_jcr_uca_chassis]) + (0.5) * multi_dot([E(self.P_rbr_uca),Te_rbr_uca_jcr_uca_chassis]))
        Q_rbl_uca_jcl_uca_upright = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbl_uca,self.ubar_rbl_uca_jcl_uca_upright).T]]),self.L_jcl_uca_upright])
        self.F_rbl_uca_jcl_uca_upright = Q_rbl_uca_jcl_uca_upright[0:3]
        Te_rbl_uca_jcl_uca_upright = Q_rbl_uca_jcl_uca_upright[3:7]
        self.T_rbl_uca_jcl_uca_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_uca),self.ubar_rbl_uca_jcl_uca_upright])),self.F_rbl_uca_jcl_uca_upright]) + (0.5) * multi_dot([E(self.P_rbl_uca),Te_rbl_uca_jcl_uca_upright]))
        Q_rbl_uca_jcl_uca_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbl_uca,self.ubar_rbl_uca_jcl_uca_chassis).T,multi_dot([B(self.P_rbl_uca,self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]]),multi_dot([B(self.P_rbl_uca,self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]])]]),self.L_jcl_uca_chassis])
        self.F_rbl_uca_jcl_uca_chassis = Q_rbl_uca_jcl_uca_chassis[0:3]
        Te_rbl_uca_jcl_uca_chassis = Q_rbl_uca_jcl_uca_chassis[3:7]
        self.T_rbl_uca_jcl_uca_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_uca),self.ubar_rbl_uca_jcl_uca_chassis])),self.F_rbl_uca_jcl_uca_chassis]) + (0.5) * multi_dot([E(self.P_rbl_uca),Te_rbl_uca_jcl_uca_chassis]))
        Q_rbr_lca_jcr_lca_upright = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbr_lca,self.ubar_rbr_lca_jcr_lca_upright).T]]),self.L_jcr_lca_upright])
        self.F_rbr_lca_jcr_lca_upright = Q_rbr_lca_jcr_lca_upright[0:3]
        Te_rbr_lca_jcr_lca_upright = Q_rbr_lca_jcr_lca_upright[3:7]
        self.T_rbr_lca_jcr_lca_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_lca),self.ubar_rbr_lca_jcr_lca_upright])),self.F_rbr_lca_jcr_lca_upright]) + (0.5) * multi_dot([E(self.P_rbr_lca),Te_rbr_lca_jcr_lca_upright]))
        Q_rbr_lca_jcr_lca_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbr_lca,self.ubar_rbr_lca_jcr_lca_chassis).T,multi_dot([B(self.P_rbr_lca,self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]]),multi_dot([B(self.P_rbr_lca,self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]])]]),self.L_jcr_lca_chassis])
        self.F_rbr_lca_jcr_lca_chassis = Q_rbr_lca_jcr_lca_chassis[0:3]
        Te_rbr_lca_jcr_lca_chassis = Q_rbr_lca_jcr_lca_chassis[3:7]
        self.T_rbr_lca_jcr_lca_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_lca),self.ubar_rbr_lca_jcr_lca_chassis])),self.F_rbr_lca_jcr_lca_chassis]) + (0.5) * multi_dot([E(self.P_rbr_lca),Te_rbr_lca_jcr_lca_chassis]))
        Q_rbl_lca_jcl_lca_upright = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbl_lca,self.ubar_rbl_lca_jcl_lca_upright).T]]),self.L_jcl_lca_upright])
        self.F_rbl_lca_jcl_lca_upright = Q_rbl_lca_jcl_lca_upright[0:3]
        Te_rbl_lca_jcl_lca_upright = Q_rbl_lca_jcl_lca_upright[3:7]
        self.T_rbl_lca_jcl_lca_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_lca),self.ubar_rbl_lca_jcl_lca_upright])),self.F_rbl_lca_jcl_lca_upright]) + (0.5) * multi_dot([E(self.P_rbl_lca),Te_rbl_lca_jcl_lca_upright]))
        Q_rbl_lca_jcl_lca_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbl_lca,self.ubar_rbl_lca_jcl_lca_chassis).T,multi_dot([B(self.P_rbl_lca,self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]]),multi_dot([B(self.P_rbl_lca,self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]])]]),self.L_jcl_lca_chassis])
        self.F_rbl_lca_jcl_lca_chassis = Q_rbl_lca_jcl_lca_chassis[0:3]
        Te_rbl_lca_jcl_lca_chassis = Q_rbl_lca_jcl_lca_chassis[3:7]
        self.T_rbl_lca_jcl_lca_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_lca),self.ubar_rbl_lca_jcl_lca_chassis])),self.F_rbl_lca_jcl_lca_chassis]) + (0.5) * multi_dot([E(self.P_rbl_lca),Te_rbl_lca_jcl_lca_chassis]))
        Q_rbr_upright_jcr_hub_bearing = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbr_upright,self.ubar_rbr_upright_jcr_hub_bearing).T,multi_dot([B(self.P_rbr_upright,self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]).T,A(self.P_rbr_hub),self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]]),multi_dot([B(self.P_rbr_upright,self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]).T,A(self.P_rbr_hub),self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]])]]),self.L_jcr_hub_bearing])
        self.F_rbr_upright_jcr_hub_bearing = Q_rbr_upright_jcr_hub_bearing[0:3]
        Te_rbr_upright_jcr_hub_bearing = Q_rbr_upright_jcr_hub_bearing[3:7]
        self.T_rbr_upright_jcr_hub_bearing = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_upright),self.ubar_rbr_upright_jcr_hub_bearing])),self.F_rbr_upright_jcr_hub_bearing]) + (0.5) * multi_dot([E(self.P_rbr_upright),Te_rbr_upright_jcr_hub_bearing]))
        Q_rbl_upright_jcl_hub_bearing = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbl_upright,self.ubar_rbl_upright_jcl_hub_bearing).T,multi_dot([B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]).T,A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]]),multi_dot([B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]).T,A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]])]]),self.L_jcl_hub_bearing])
        self.F_rbl_upright_jcl_hub_bearing = Q_rbl_upright_jcl_hub_bearing[0:3]
        Te_rbl_upright_jcl_hub_bearing = Q_rbl_upright_jcl_hub_bearing[3:7]
        self.T_rbl_upright_jcl_hub_bearing = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_upright),self.ubar_rbl_upright_jcl_hub_bearing])),self.F_rbl_upright_jcl_hub_bearing]) + (0.5) * multi_dot([E(self.P_rbl_upright),Te_rbl_upright_jcl_hub_bearing]))
        Q_rbr_upper_strut_jcr_strut_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut_chassis).T,multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]])]]),self.L_jcr_strut_chassis])
        self.F_rbr_upper_strut_jcr_strut_chassis = Q_rbr_upper_strut_jcr_strut_chassis[0:3]
        Te_rbr_upper_strut_jcr_strut_chassis = Q_rbr_upper_strut_jcr_strut_chassis[3:7]
        self.T_rbr_upper_strut_jcr_strut_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut_chassis])),self.F_rbr_upper_strut_jcr_strut_chassis]) + (0.5) * multi_dot([E(self.P_rbr_upper_strut),Te_rbr_upper_strut_jcr_strut_chassis]))
        Q_rbr_upper_strut_jcr_strut = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]]),multi_dot([A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]])],[multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]).T,A(self.P_rbr_lower_strut),self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]]),multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]).T,A(self.P_rbr_lower_strut),self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]]),(multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]).T,((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut]) + self.R_rbr_upper_strut)]) + multi_dot([B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut).T,A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]])),(multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]).T,((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut]) + self.R_rbr_upper_strut)]) + multi_dot([B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut).T,A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]]))]]),self.L_jcr_strut])
        self.F_rbr_upper_strut_jcr_strut = Q_rbr_upper_strut_jcr_strut[0:3]
        Te_rbr_upper_strut_jcr_strut = Q_rbr_upper_strut_jcr_strut[3:7]
        self.T_rbr_upper_strut_jcr_strut = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut])),self.F_rbr_upper_strut_jcr_strut]) + (0.5) * multi_dot([E(self.P_rbr_upper_strut),Te_rbr_upper_strut_jcr_strut]))
        self.F_rbr_upper_strut_far_strut = (1.0/((multi_dot([((-1) * self.R_rbr_lower_strut.T + multi_dot([self.ubar_rbr_upper_strut_far_strut.T,A(self.P_rbr_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbr_lower_strut_far_strut.T,A(self.P_rbr_lower_strut).T]) + self.R_rbr_upper_strut.T),((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_far_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_far_strut]) + self.R_rbr_upper_strut)]))**(1.0/2.0))[0] * (config.UF_far_strut_Fd((-1 * 1.0/((multi_dot([((-1) * self.R_rbr_lower_strut.T + multi_dot([self.ubar_rbr_upper_strut_far_strut.T,A(self.P_rbr_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbr_lower_strut_far_strut.T,A(self.P_rbr_lower_strut).T]) + self.R_rbr_upper_strut.T),((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_far_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_far_strut]) + self.R_rbr_upper_strut)]))**(1.0/2.0))[0]) * multi_dot([((-1) * self.R_rbr_lower_strut.T + multi_dot([self.ubar_rbr_upper_strut_far_strut.T,A(self.P_rbr_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbr_lower_strut_far_strut.T,A(self.P_rbr_lower_strut).T]) + self.R_rbr_upper_strut.T),((-1) * self.Rd_rbr_lower_strut + multi_dot([B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_far_strut),self.Pd_rbr_upper_strut]) + (-1) * multi_dot([B(self.P_rbr_lower_strut,self.ubar_rbr_lower_strut_far_strut),self.Pd_rbr_lower_strut]) + self.Rd_rbr_upper_strut)])) + config.UF_far_strut_Fs((config.far_strut_FL + (-1 * ((multi_dot([((-1) * self.R_rbr_lower_strut.T + multi_dot([self.ubar_rbr_upper_strut_far_strut.T,A(self.P_rbr_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbr_lower_strut_far_strut.T,A(self.P_rbr_lower_strut).T]) + self.R_rbr_upper_strut.T),((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_far_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_far_strut]) + self.R_rbr_upper_strut)]))**(1.0/2.0))[0]))))) * ((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_far_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_far_strut]) + self.R_rbr_upper_strut)
        self.T_rbr_upper_strut_far_strut = Z3x1
        Q_rbl_upper_strut_jcl_strut_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut_chassis).T,multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]])]]),self.L_jcl_strut_chassis])
        self.F_rbl_upper_strut_jcl_strut_chassis = Q_rbl_upper_strut_jcl_strut_chassis[0:3]
        Te_rbl_upper_strut_jcl_strut_chassis = Q_rbl_upper_strut_jcl_strut_chassis[3:7]
        self.T_rbl_upper_strut_jcl_strut_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut_chassis])),self.F_rbl_upper_strut_jcl_strut_chassis]) + (0.5) * multi_dot([E(self.P_rbl_upper_strut),Te_rbl_upper_strut_jcl_strut_chassis]))
        Q_rbl_upper_strut_jcl_strut = (-1) * multi_dot([np.bmat([[Z1x3.T,Z1x3.T,multi_dot([A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]]),multi_dot([A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]])],[multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]).T,A(self.P_rbl_lower_strut),self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]]),multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]).T,A(self.P_rbl_lower_strut),self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]]),(multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]).T,((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut]) + self.R_rbl_upper_strut)]) + multi_dot([B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut).T,A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]])),(multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]).T,((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut]) + self.R_rbl_upper_strut)]) + multi_dot([B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut).T,A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]]))]]),self.L_jcl_strut])
        self.F_rbl_upper_strut_jcl_strut = Q_rbl_upper_strut_jcl_strut[0:3]
        Te_rbl_upper_strut_jcl_strut = Q_rbl_upper_strut_jcl_strut[3:7]
        self.T_rbl_upper_strut_jcl_strut = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut])),self.F_rbl_upper_strut_jcl_strut]) + (0.5) * multi_dot([E(self.P_rbl_upper_strut),Te_rbl_upper_strut_jcl_strut]))
        self.F_rbl_upper_strut_fal_strut = (1.0/((multi_dot([((-1) * self.R_rbl_lower_strut.T + multi_dot([self.ubar_rbl_upper_strut_fal_strut.T,A(self.P_rbl_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbl_lower_strut_fal_strut.T,A(self.P_rbl_lower_strut).T]) + self.R_rbl_upper_strut.T),((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_fal_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_fal_strut]) + self.R_rbl_upper_strut)]))**(1.0/2.0))[0] * (config.UF_fal_strut_Fd((-1 * 1.0/((multi_dot([((-1) * self.R_rbl_lower_strut.T + multi_dot([self.ubar_rbl_upper_strut_fal_strut.T,A(self.P_rbl_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbl_lower_strut_fal_strut.T,A(self.P_rbl_lower_strut).T]) + self.R_rbl_upper_strut.T),((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_fal_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_fal_strut]) + self.R_rbl_upper_strut)]))**(1.0/2.0))[0]) * multi_dot([((-1) * self.R_rbl_lower_strut.T + multi_dot([self.ubar_rbl_upper_strut_fal_strut.T,A(self.P_rbl_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbl_lower_strut_fal_strut.T,A(self.P_rbl_lower_strut).T]) + self.R_rbl_upper_strut.T),((-1) * self.Rd_rbl_lower_strut + multi_dot([B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_fal_strut),self.Pd_rbl_upper_strut]) + (-1) * multi_dot([B(self.P_rbl_lower_strut,self.ubar_rbl_lower_strut_fal_strut),self.Pd_rbl_lower_strut]) + self.Rd_rbl_upper_strut)])) + config.UF_fal_strut_Fs((config.fal_strut_FL + (-1 * ((multi_dot([((-1) * self.R_rbl_lower_strut.T + multi_dot([self.ubar_rbl_upper_strut_fal_strut.T,A(self.P_rbl_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbl_lower_strut_fal_strut.T,A(self.P_rbl_lower_strut).T]) + self.R_rbl_upper_strut.T),((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_fal_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_fal_strut]) + self.R_rbl_upper_strut)]))**(1.0/2.0))[0]))))) * ((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_fal_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_fal_strut]) + self.R_rbl_upper_strut)
        self.T_rbl_upper_strut_fal_strut = Z3x1
        Q_rbr_lower_strut_jcr_strut_lca = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbr_lower_strut,self.ubar_rbr_lower_strut_jcr_strut_lca).T,multi_dot([B(self.P_rbr_lower_strut,self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]).T,A(self.P_rbr_lca),self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]])]]),self.L_jcr_strut_lca])
        self.F_rbr_lower_strut_jcr_strut_lca = Q_rbr_lower_strut_jcr_strut_lca[0:3]
        Te_rbr_lower_strut_jcr_strut_lca = Q_rbr_lower_strut_jcr_strut_lca[3:7]
        self.T_rbr_lower_strut_jcr_strut_lca = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut_lca])),self.F_rbr_lower_strut_jcr_strut_lca]) + (0.5) * multi_dot([E(self.P_rbr_lower_strut),Te_rbr_lower_strut_jcr_strut_lca]))
        Q_rbl_lower_strut_jcl_strut_lca = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbl_lower_strut,self.ubar_rbl_lower_strut_jcl_strut_lca).T,multi_dot([B(self.P_rbl_lower_strut,self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]).T,A(self.P_rbl_lca),self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]])]]),self.L_jcl_strut_lca])
        self.F_rbl_lower_strut_jcl_strut_lca = Q_rbl_lower_strut_jcl_strut_lca[0:3]
        Te_rbl_lower_strut_jcl_strut_lca = Q_rbl_lower_strut_jcl_strut_lca[3:7]
        self.T_rbl_lower_strut_jcl_strut_lca = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut_lca])),self.F_rbl_lower_strut_jcl_strut_lca]) + (0.5) * multi_dot([E(self.P_rbl_lower_strut),Te_rbl_lower_strut_jcl_strut_lca]))
        Q_rbr_tie_rod_jcr_tie_upright = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbr_tie_rod,self.ubar_rbr_tie_rod_jcr_tie_upright).T]]),self.L_jcr_tie_upright])
        self.F_rbr_tie_rod_jcr_tie_upright = Q_rbr_tie_rod_jcr_tie_upright[0:3]
        Te_rbr_tie_rod_jcr_tie_upright = Q_rbr_tie_rod_jcr_tie_upright[3:7]
        self.T_rbr_tie_rod_jcr_tie_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_tie_rod),self.ubar_rbr_tie_rod_jcr_tie_upright])),self.F_rbr_tie_rod_jcr_tie_upright]) + (0.5) * multi_dot([E(self.P_rbr_tie_rod),Te_rbr_tie_rod_jcr_tie_upright]))
        Q_rbr_tie_rod_jcr_tie_steering = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbr_tie_rod,self.ubar_rbr_tie_rod_jcr_tie_steering).T,multi_dot([B(self.P_rbr_tie_rod,self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]).T,A(self.P_vbr_steer),self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]])]]),self.L_jcr_tie_steering])
        self.F_rbr_tie_rod_jcr_tie_steering = Q_rbr_tie_rod_jcr_tie_steering[0:3]
        Te_rbr_tie_rod_jcr_tie_steering = Q_rbr_tie_rod_jcr_tie_steering[3:7]
        self.T_rbr_tie_rod_jcr_tie_steering = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_tie_rod),self.ubar_rbr_tie_rod_jcr_tie_steering])),self.F_rbr_tie_rod_jcr_tie_steering]) + (0.5) * multi_dot([E(self.P_rbr_tie_rod),Te_rbr_tie_rod_jcr_tie_steering]))
        Q_rbl_tie_rod_jcl_tie_upright = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbl_tie_rod,self.ubar_rbl_tie_rod_jcl_tie_upright).T]]),self.L_jcl_tie_upright])
        self.F_rbl_tie_rod_jcl_tie_upright = Q_rbl_tie_rod_jcl_tie_upright[0:3]
        Te_rbl_tie_rod_jcl_tie_upright = Q_rbl_tie_rod_jcl_tie_upright[3:7]
        self.T_rbl_tie_rod_jcl_tie_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_tie_rod),self.ubar_rbl_tie_rod_jcl_tie_upright])),self.F_rbl_tie_rod_jcl_tie_upright]) + (0.5) * multi_dot([E(self.P_rbl_tie_rod),Te_rbl_tie_rod_jcl_tie_upright]))
        Q_rbl_tie_rod_jcl_tie_steering = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbl_tie_rod,self.ubar_rbl_tie_rod_jcl_tie_steering).T,multi_dot([B(self.P_rbl_tie_rod,self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]).T,A(self.P_vbl_steer),self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]])]]),self.L_jcl_tie_steering])
        self.F_rbl_tie_rod_jcl_tie_steering = Q_rbl_tie_rod_jcl_tie_steering[0:3]
        Te_rbl_tie_rod_jcl_tie_steering = Q_rbl_tie_rod_jcl_tie_steering[3:7]
        self.T_rbl_tie_rod_jcl_tie_steering = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_tie_rod),self.ubar_rbl_tie_rod_jcl_tie_steering])),self.F_rbl_tie_rod_jcl_tie_steering]) + (0.5) * multi_dot([E(self.P_rbl_tie_rod),Te_rbl_tie_rod_jcl_tie_steering]))

        self.reactions = {'F_rbr_uca_jcr_uca_upright' : self.F_rbr_uca_jcr_uca_upright,
                        'T_rbr_uca_jcr_uca_upright' : self.T_rbr_uca_jcr_uca_upright,
                        'F_rbr_uca_jcr_uca_chassis' : self.F_rbr_uca_jcr_uca_chassis,
                        'T_rbr_uca_jcr_uca_chassis' : self.T_rbr_uca_jcr_uca_chassis,
                        'F_rbl_uca_jcl_uca_upright' : self.F_rbl_uca_jcl_uca_upright,
                        'T_rbl_uca_jcl_uca_upright' : self.T_rbl_uca_jcl_uca_upright,
                        'F_rbl_uca_jcl_uca_chassis' : self.F_rbl_uca_jcl_uca_chassis,
                        'T_rbl_uca_jcl_uca_chassis' : self.T_rbl_uca_jcl_uca_chassis,
                        'F_rbr_lca_jcr_lca_upright' : self.F_rbr_lca_jcr_lca_upright,
                        'T_rbr_lca_jcr_lca_upright' : self.T_rbr_lca_jcr_lca_upright,
                        'F_rbr_lca_jcr_lca_chassis' : self.F_rbr_lca_jcr_lca_chassis,
                        'T_rbr_lca_jcr_lca_chassis' : self.T_rbr_lca_jcr_lca_chassis,
                        'F_rbl_lca_jcl_lca_upright' : self.F_rbl_lca_jcl_lca_upright,
                        'T_rbl_lca_jcl_lca_upright' : self.T_rbl_lca_jcl_lca_upright,
                        'F_rbl_lca_jcl_lca_chassis' : self.F_rbl_lca_jcl_lca_chassis,
                        'T_rbl_lca_jcl_lca_chassis' : self.T_rbl_lca_jcl_lca_chassis,
                        'F_rbr_upright_jcr_hub_bearing' : self.F_rbr_upright_jcr_hub_bearing,
                        'T_rbr_upright_jcr_hub_bearing' : self.T_rbr_upright_jcr_hub_bearing,
                        'F_rbl_upright_jcl_hub_bearing' : self.F_rbl_upright_jcl_hub_bearing,
                        'T_rbl_upright_jcl_hub_bearing' : self.T_rbl_upright_jcl_hub_bearing,
                        'F_rbr_upper_strut_jcr_strut_chassis' : self.F_rbr_upper_strut_jcr_strut_chassis,
                        'T_rbr_upper_strut_jcr_strut_chassis' : self.T_rbr_upper_strut_jcr_strut_chassis,
                        'F_rbr_upper_strut_jcr_strut' : self.F_rbr_upper_strut_jcr_strut,
                        'T_rbr_upper_strut_jcr_strut' : self.T_rbr_upper_strut_jcr_strut,
                        'F_rbr_upper_strut_far_strut' : self.F_rbr_upper_strut_far_strut,
                        'T_rbr_upper_strut_far_strut' : self.T_rbr_upper_strut_far_strut,
                        'F_rbl_upper_strut_jcl_strut_chassis' : self.F_rbl_upper_strut_jcl_strut_chassis,
                        'T_rbl_upper_strut_jcl_strut_chassis' : self.T_rbl_upper_strut_jcl_strut_chassis,
                        'F_rbl_upper_strut_jcl_strut' : self.F_rbl_upper_strut_jcl_strut,
                        'T_rbl_upper_strut_jcl_strut' : self.T_rbl_upper_strut_jcl_strut,
                        'F_rbl_upper_strut_fal_strut' : self.F_rbl_upper_strut_fal_strut,
                        'T_rbl_upper_strut_fal_strut' : self.T_rbl_upper_strut_fal_strut,
                        'F_rbr_lower_strut_jcr_strut_lca' : self.F_rbr_lower_strut_jcr_strut_lca,
                        'T_rbr_lower_strut_jcr_strut_lca' : self.T_rbr_lower_strut_jcr_strut_lca,
                        'F_rbl_lower_strut_jcl_strut_lca' : self.F_rbl_lower_strut_jcl_strut_lca,
                        'T_rbl_lower_strut_jcl_strut_lca' : self.T_rbl_lower_strut_jcl_strut_lca,
                        'F_rbr_tie_rod_jcr_tie_upright' : self.F_rbr_tie_rod_jcr_tie_upright,
                        'T_rbr_tie_rod_jcr_tie_upright' : self.T_rbr_tie_rod_jcr_tie_upright,
                        'F_rbr_tie_rod_jcr_tie_steering' : self.F_rbr_tie_rod_jcr_tie_steering,
                        'T_rbr_tie_rod_jcr_tie_steering' : self.T_rbr_tie_rod_jcr_tie_steering,
                        'F_rbl_tie_rod_jcl_tie_upright' : self.F_rbl_tie_rod_jcl_tie_upright,
                        'T_rbl_tie_rod_jcl_tie_upright' : self.T_rbl_tie_rod_jcl_tie_upright,
                        'F_rbl_tie_rod_jcl_tie_steering' : self.F_rbl_tie_rod_jcl_tie_steering,
                        'T_rbl_tie_rod_jcl_tie_steering' : self.T_rbl_tie_rod_jcl_tie_steering}

