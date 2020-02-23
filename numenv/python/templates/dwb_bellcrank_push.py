
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

        self.indicies_map = {'vbs_ground': 0, 'rbr_uca': 1, 'rbl_uca': 2, 'rbr_lca': 3, 'rbl_lca': 4, 'rbr_upright': 5, 'rbl_upright': 6, 'rbr_pushrod': 7, 'rbl_pushrod': 8, 'rbr_rocker': 9, 'rbl_rocker': 10, 'rbr_upper_strut': 11, 'rbl_upper_strut': 12, 'rbr_lower_strut': 13, 'rbl_lower_strut': 14, 'rbr_tie_rod': 15, 'rbl_tie_rod': 16, 'rbr_hub': 17, 'rbl_hub': 18, 'vbr_steer': 19, 'vbl_steer': 20, 'vbs_chassis': 21}

        self.n  = 126
        self.nc = 122
        self.nrows = 74
        self.ncols = 2*18
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_rbr_uca_jcr_uca_upright', 'T_rbr_uca_jcr_uca_upright', 'F_rbr_uca_jcr_uca_chassis', 'T_rbr_uca_jcr_uca_chassis', 'F_rbr_uca_jcr_prod_uca', 'T_rbr_uca_jcr_prod_uca', 'F_rbl_uca_jcl_uca_upright', 'T_rbl_uca_jcl_uca_upright', 'F_rbl_uca_jcl_uca_chassis', 'T_rbl_uca_jcl_uca_chassis', 'F_rbl_uca_jcl_prod_uca', 'T_rbl_uca_jcl_prod_uca', 'F_rbr_lca_jcr_lca_upright', 'T_rbr_lca_jcr_lca_upright', 'F_rbr_lca_jcr_lca_chassis', 'T_rbr_lca_jcr_lca_chassis', 'F_rbl_lca_jcl_lca_upright', 'T_rbl_lca_jcl_lca_upright', 'F_rbl_lca_jcl_lca_chassis', 'T_rbl_lca_jcl_lca_chassis', 'F_rbr_upright_jcr_hub_bearing', 'T_rbr_upright_jcr_hub_bearing', 'F_rbl_upright_jcl_hub_bearing', 'T_rbl_upright_jcl_hub_bearing', 'F_rbr_rocker_jcr_prod_rocker', 'T_rbr_rocker_jcr_prod_rocker', 'F_rbr_rocker_jcr_rocker_chassis', 'T_rbr_rocker_jcr_rocker_chassis', 'F_rbl_rocker_jcl_prod_rocker', 'T_rbl_rocker_jcl_prod_rocker', 'F_rbl_rocker_jcl_rocker_chassis', 'T_rbl_rocker_jcl_rocker_chassis', 'F_rbr_upper_strut_jcr_strut_chassis', 'T_rbr_upper_strut_jcr_strut_chassis', 'F_rbr_upper_strut_jcr_strut', 'T_rbr_upper_strut_jcr_strut', 'F_rbl_upper_strut_jcl_strut_chassis', 'T_rbl_upper_strut_jcl_strut_chassis', 'F_rbl_upper_strut_jcl_strut', 'T_rbl_upper_strut_jcl_strut', 'F_rbr_lower_strut_jcr_strut_rocker', 'T_rbr_lower_strut_jcr_strut_rocker', 'F_rbl_lower_strut_jcl_strut_rocker', 'T_rbl_lower_strut_jcl_strut_rocker', 'F_rbr_tie_rod_jcr_tie_upright', 'T_rbr_tie_rod_jcr_tie_upright', 'F_rbr_tie_rod_jcr_tie_steering', 'T_rbr_tie_rod_jcr_tie_steering', 'F_rbl_tie_rod_jcl_tie_upright', 'T_rbl_tie_rod_jcl_tie_upright', 'F_rbl_tie_rod_jcl_tie_steering', 'T_rbl_tie_rod_jcl_tie_steering']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 46, 47, 47, 47, 47, 48, 48, 48, 48, 49, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 51, 52, 52, 52, 52, 53, 53, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71, 72, 72, 73, 73])
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.rbr_uca*2, self.rbr_uca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.rbr_pushrod*2, self.rbr_pushrod*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.rbr_pushrod*2, self.rbr_pushrod*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbl_pushrod*2, self.rbl_pushrod*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbl_pushrod*2, self.rbl_pushrod*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbr_pushrod*2, self.rbr_pushrod*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_pushrod*2, self.rbl_pushrod*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.vbr_steer*2, self.vbr_steer*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.vbr_steer*2, self.vbr_steer*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.vbl_steer*2, self.vbl_steer*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.vbl_steer*2, self.vbl_steer*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbr_pushrod*2, self.rbr_pushrod*2+1, self.rbl_pushrod*2, self.rbl_pushrod*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbl_hub*2, self.rbl_hub*2+1])

    def set_initial_states(self):
        self.q0  = np.concatenate([self.config.R_rbr_uca,
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
        self.config.R_rbr_pushrod,
        self.config.P_rbr_pushrod,
        self.config.R_rbl_pushrod,
        self.config.P_rbl_pushrod,
        self.config.R_rbr_rocker,
        self.config.P_rbr_rocker,
        self.config.R_rbl_rocker,
        self.config.P_rbl_rocker,
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
        self.config.P_rbl_hub])
        self.qd0 = np.concatenate([self.config.Rd_rbr_uca,
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
        self.config.Rd_rbr_pushrod,
        self.config.Pd_rbr_pushrod,
        self.config.Rd_rbl_pushrod,
        self.config.Pd_rbl_pushrod,
        self.config.Rd_rbr_rocker,
        self.config.Pd_rbr_rocker,
        self.config.Rd_rbl_rocker,
        self.config.Pd_rbl_rocker,
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
        self.config.Pd_rbl_hub])

        self.set_gen_coordinates(self.q0)
        self.set_gen_velocities(self.qd0)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbr_uca = indicies_map[p + 'rbr_uca']
        self.rbl_uca = indicies_map[p + 'rbl_uca']
        self.rbr_lca = indicies_map[p + 'rbr_lca']
        self.rbl_lca = indicies_map[p + 'rbl_lca']
        self.rbr_upright = indicies_map[p + 'rbr_upright']
        self.rbl_upright = indicies_map[p + 'rbl_upright']
        self.rbr_pushrod = indicies_map[p + 'rbr_pushrod']
        self.rbl_pushrod = indicies_map[p + 'rbl_pushrod']
        self.rbr_rocker = indicies_map[p + 'rbr_rocker']
        self.rbl_rocker = indicies_map[p + 'rbl_rocker']
        self.rbr_upper_strut = indicies_map[p + 'rbr_upper_strut']
        self.rbl_upper_strut = indicies_map[p + 'rbl_upper_strut']
        self.rbr_lower_strut = indicies_map[p + 'rbr_lower_strut']
        self.rbl_lower_strut = indicies_map[p + 'rbl_lower_strut']
        self.rbr_tie_rod = indicies_map[p + 'rbr_tie_rod']
        self.rbl_tie_rod = indicies_map[p + 'rbl_tie_rod']
        self.rbr_hub = indicies_map[p + 'rbr_hub']
        self.rbl_hub = indicies_map[p + 'rbl_hub']
        self.vbs_ground = indicies_map[interface_map[p + 'vbs_ground']]
        self.vbr_steer = indicies_map[interface_map[p + 'vbr_steer']]
        self.vbl_steer = indicies_map[interface_map[p + 'vbl_steer']]
        self.vbs_chassis = indicies_map[interface_map[p + 'vbs_chassis']]

    
    def eval_constants(self):
        config = self.config

        self.F_rbr_uca_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_uca]], dtype=np.float64)
        self.F_rbl_uca_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_uca]], dtype=np.float64)
        self.F_rbr_lca_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_lca]], dtype=np.float64)
        self.F_rbl_lca_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_lca]], dtype=np.float64)
        self.F_rbr_upright_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_upright]], dtype=np.float64)
        self.F_rbl_upright_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_upright]], dtype=np.float64)
        self.F_rbr_pushrod_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_pushrod]], dtype=np.float64)
        self.F_rbl_pushrod_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_pushrod]], dtype=np.float64)
        self.F_rbr_rocker_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_rocker]], dtype=np.float64)
        self.F_rbl_rocker_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_rocker]], dtype=np.float64)
        self.F_rbr_upper_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_upper_strut]], dtype=np.float64)
        self.T_rbr_upper_strut_far_strut = np.zeros((3,1),dtype=np.float64)
        self.T_rbr_lower_strut_far_strut = np.zeros((3,1),dtype=np.float64)
        self.F_rbl_upper_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_upper_strut]], dtype=np.float64)
        self.T_rbl_upper_strut_fal_strut = np.zeros((3,1),dtype=np.float64)
        self.T_rbl_lower_strut_fal_strut = np.zeros((3,1),dtype=np.float64)
        self.F_rbr_lower_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_lower_strut]], dtype=np.float64)
        self.F_rbl_lower_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_lower_strut]], dtype=np.float64)
        self.F_rbr_tie_rod_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_tie_rod]], dtype=np.float64)
        self.F_rbl_tie_rod_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_tie_rod]], dtype=np.float64)
        self.F_rbr_hub_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_hub]], dtype=np.float64)
        self.F_rbl_hub_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_hub]], dtype=np.float64)

        self.Mbar_rbr_uca_jcr_uca_upright = multi_dot([A(config.P_rbr_uca).T,triad(config.ax1_jcr_uca_upright)])
        self.Mbar_rbr_upright_jcr_uca_upright = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_uca_upright)])
        self.ubar_rbr_uca_jcr_uca_upright = (multi_dot([A(config.P_rbr_uca).T,config.pt1_jcr_uca_upright]) + -1*multi_dot([A(config.P_rbr_uca).T,config.R_rbr_uca]))
        self.ubar_rbr_upright_jcr_uca_upright = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_uca_upright]) + -1*multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.Mbar_rbr_uca_jcr_uca_chassis = multi_dot([A(config.P_rbr_uca).T,triad(config.ax1_jcr_uca_chassis)])
        self.Mbar_vbs_chassis_jcr_uca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcr_uca_chassis)])
        self.ubar_rbr_uca_jcr_uca_chassis = (multi_dot([A(config.P_rbr_uca).T,config.pt1_jcr_uca_chassis]) + -1*multi_dot([A(config.P_rbr_uca).T,config.R_rbr_uca]))
        self.ubar_vbs_chassis_jcr_uca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_uca_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbr_uca_jcr_prod_uca = multi_dot([A(config.P_rbr_uca).T,triad(config.ax1_jcr_prod_uca)])
        self.Mbar_rbr_pushrod_jcr_prod_uca = multi_dot([A(config.P_rbr_pushrod).T,triad(config.ax2_jcr_prod_uca,triad(config.ax1_jcr_prod_uca)[0:3,1:2])])
        self.ubar_rbr_uca_jcr_prod_uca = (multi_dot([A(config.P_rbr_uca).T,config.pt1_jcr_prod_uca]) + -1*multi_dot([A(config.P_rbr_uca).T,config.R_rbr_uca]))
        self.ubar_rbr_pushrod_jcr_prod_uca = (multi_dot([A(config.P_rbr_pushrod).T,config.pt1_jcr_prod_uca]) + -1*multi_dot([A(config.P_rbr_pushrod).T,config.R_rbr_pushrod]))
        self.Mbar_rbl_uca_jcl_uca_upright = multi_dot([A(config.P_rbl_uca).T,triad(config.ax1_jcl_uca_upright)])
        self.Mbar_rbl_upright_jcl_uca_upright = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_uca_upright)])
        self.ubar_rbl_uca_jcl_uca_upright = (multi_dot([A(config.P_rbl_uca).T,config.pt1_jcl_uca_upright]) + -1*multi_dot([A(config.P_rbl_uca).T,config.R_rbl_uca]))
        self.ubar_rbl_upright_jcl_uca_upright = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_uca_upright]) + -1*multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.Mbar_rbl_uca_jcl_uca_chassis = multi_dot([A(config.P_rbl_uca).T,triad(config.ax1_jcl_uca_chassis)])
        self.Mbar_vbs_chassis_jcl_uca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcl_uca_chassis)])
        self.ubar_rbl_uca_jcl_uca_chassis = (multi_dot([A(config.P_rbl_uca).T,config.pt1_jcl_uca_chassis]) + -1*multi_dot([A(config.P_rbl_uca).T,config.R_rbl_uca]))
        self.ubar_vbs_chassis_jcl_uca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_uca_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_uca_jcl_prod_uca = multi_dot([A(config.P_rbl_uca).T,triad(config.ax1_jcl_prod_uca)])
        self.Mbar_rbl_pushrod_jcl_prod_uca = multi_dot([A(config.P_rbl_pushrod).T,triad(config.ax2_jcl_prod_uca,triad(config.ax1_jcl_prod_uca)[0:3,1:2])])
        self.ubar_rbl_uca_jcl_prod_uca = (multi_dot([A(config.P_rbl_uca).T,config.pt1_jcl_prod_uca]) + -1*multi_dot([A(config.P_rbl_uca).T,config.R_rbl_uca]))
        self.ubar_rbl_pushrod_jcl_prod_uca = (multi_dot([A(config.P_rbl_pushrod).T,config.pt1_jcl_prod_uca]) + -1*multi_dot([A(config.P_rbl_pushrod).T,config.R_rbl_pushrod]))
        self.Mbar_rbr_lca_jcr_lca_upright = multi_dot([A(config.P_rbr_lca).T,triad(config.ax1_jcr_lca_upright)])
        self.Mbar_rbr_upright_jcr_lca_upright = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_lca_upright)])
        self.ubar_rbr_lca_jcr_lca_upright = (multi_dot([A(config.P_rbr_lca).T,config.pt1_jcr_lca_upright]) + -1*multi_dot([A(config.P_rbr_lca).T,config.R_rbr_lca]))
        self.ubar_rbr_upright_jcr_lca_upright = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_lca_upright]) + -1*multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.Mbar_rbr_lca_jcr_lca_chassis = multi_dot([A(config.P_rbr_lca).T,triad(config.ax1_jcr_lca_chassis)])
        self.Mbar_vbs_chassis_jcr_lca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcr_lca_chassis)])
        self.ubar_rbr_lca_jcr_lca_chassis = (multi_dot([A(config.P_rbr_lca).T,config.pt1_jcr_lca_chassis]) + -1*multi_dot([A(config.P_rbr_lca).T,config.R_rbr_lca]))
        self.ubar_vbs_chassis_jcr_lca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_lca_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_lca_jcl_lca_upright = multi_dot([A(config.P_rbl_lca).T,triad(config.ax1_jcl_lca_upright)])
        self.Mbar_rbl_upright_jcl_lca_upright = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_lca_upright)])
        self.ubar_rbl_lca_jcl_lca_upright = (multi_dot([A(config.P_rbl_lca).T,config.pt1_jcl_lca_upright]) + -1*multi_dot([A(config.P_rbl_lca).T,config.R_rbl_lca]))
        self.ubar_rbl_upright_jcl_lca_upright = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_lca_upright]) + -1*multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.Mbar_rbl_lca_jcl_lca_chassis = multi_dot([A(config.P_rbl_lca).T,triad(config.ax1_jcl_lca_chassis)])
        self.Mbar_vbs_chassis_jcl_lca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcl_lca_chassis)])
        self.ubar_rbl_lca_jcl_lca_chassis = (multi_dot([A(config.P_rbl_lca).T,config.pt1_jcl_lca_chassis]) + -1*multi_dot([A(config.P_rbl_lca).T,config.R_rbl_lca]))
        self.ubar_vbs_chassis_jcl_lca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_lca_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbr_upright_jcr_hub_bearing = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_hub_bearing)])
        self.Mbar_rbr_hub_jcr_hub_bearing = multi_dot([A(config.P_rbr_hub).T,triad(config.ax1_jcr_hub_bearing)])
        self.ubar_rbr_upright_jcr_hub_bearing = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_hub_bearing]) + -1*multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.ubar_rbr_hub_jcr_hub_bearing = (multi_dot([A(config.P_rbr_hub).T,config.pt1_jcr_hub_bearing]) + -1*multi_dot([A(config.P_rbr_hub).T,config.R_rbr_hub]))
        self.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_hub_bearing)])
        self.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([A(config.P_rbl_hub).T,triad(config.ax1_jcl_hub_bearing)])
        self.ubar_rbl_upright_jcl_hub_bearing = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_hub_bearing]) + -1*multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.ubar_rbl_hub_jcl_hub_bearing = (multi_dot([A(config.P_rbl_hub).T,config.pt1_jcl_hub_bearing]) + -1*multi_dot([A(config.P_rbl_hub).T,config.R_rbl_hub]))
        self.Mbar_rbr_rocker_jcr_prod_rocker = multi_dot([A(config.P_rbr_rocker).T,triad(config.ax1_jcr_prod_rocker)])
        self.Mbar_rbr_pushrod_jcr_prod_rocker = multi_dot([A(config.P_rbr_pushrod).T,triad(config.ax1_jcr_prod_rocker)])
        self.ubar_rbr_rocker_jcr_prod_rocker = (multi_dot([A(config.P_rbr_rocker).T,config.pt1_jcr_prod_rocker]) + -1*multi_dot([A(config.P_rbr_rocker).T,config.R_rbr_rocker]))
        self.ubar_rbr_pushrod_jcr_prod_rocker = (multi_dot([A(config.P_rbr_pushrod).T,config.pt1_jcr_prod_rocker]) + -1*multi_dot([A(config.P_rbr_pushrod).T,config.R_rbr_pushrod]))
        self.Mbar_rbr_rocker_jcr_rocker_chassis = multi_dot([A(config.P_rbr_rocker).T,triad(config.ax1_jcr_rocker_chassis)])
        self.Mbar_vbs_chassis_jcr_rocker_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcr_rocker_chassis)])
        self.ubar_rbr_rocker_jcr_rocker_chassis = (multi_dot([A(config.P_rbr_rocker).T,config.pt1_jcr_rocker_chassis]) + -1*multi_dot([A(config.P_rbr_rocker).T,config.R_rbr_rocker]))
        self.ubar_vbs_chassis_jcr_rocker_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_rocker_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_rocker_jcl_prod_rocker = multi_dot([A(config.P_rbl_rocker).T,triad(config.ax1_jcl_prod_rocker)])
        self.Mbar_rbl_pushrod_jcl_prod_rocker = multi_dot([A(config.P_rbl_pushrod).T,triad(config.ax1_jcl_prod_rocker)])
        self.ubar_rbl_rocker_jcl_prod_rocker = (multi_dot([A(config.P_rbl_rocker).T,config.pt1_jcl_prod_rocker]) + -1*multi_dot([A(config.P_rbl_rocker).T,config.R_rbl_rocker]))
        self.ubar_rbl_pushrod_jcl_prod_rocker = (multi_dot([A(config.P_rbl_pushrod).T,config.pt1_jcl_prod_rocker]) + -1*multi_dot([A(config.P_rbl_pushrod).T,config.R_rbl_pushrod]))
        self.Mbar_rbl_rocker_jcl_rocker_chassis = multi_dot([A(config.P_rbl_rocker).T,triad(config.ax1_jcl_rocker_chassis)])
        self.Mbar_vbs_chassis_jcl_rocker_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcl_rocker_chassis)])
        self.ubar_rbl_rocker_jcl_rocker_chassis = (multi_dot([A(config.P_rbl_rocker).T,config.pt1_jcl_rocker_chassis]) + -1*multi_dot([A(config.P_rbl_rocker).T,config.R_rbl_rocker]))
        self.ubar_vbs_chassis_jcl_rocker_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_rocker_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbr_upper_strut_jcr_strut_chassis = multi_dot([A(config.P_rbr_upper_strut).T,triad(config.ax1_jcr_strut_chassis)])
        self.Mbar_vbs_chassis_jcr_strut_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax2_jcr_strut_chassis,triad(config.ax1_jcr_strut_chassis)[0:3,1:2])])
        self.ubar_rbr_upper_strut_jcr_strut_chassis = (multi_dot([A(config.P_rbr_upper_strut).T,config.pt1_jcr_strut_chassis]) + -1*multi_dot([A(config.P_rbr_upper_strut).T,config.R_rbr_upper_strut]))
        self.ubar_vbs_chassis_jcr_strut_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_strut_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbr_upper_strut_jcr_strut = multi_dot([A(config.P_rbr_upper_strut).T,triad(config.ax1_jcr_strut)])
        self.Mbar_rbr_lower_strut_jcr_strut = multi_dot([A(config.P_rbr_lower_strut).T,triad(config.ax1_jcr_strut)])
        self.ubar_rbr_upper_strut_jcr_strut = (multi_dot([A(config.P_rbr_upper_strut).T,config.pt1_jcr_strut]) + -1*multi_dot([A(config.P_rbr_upper_strut).T,config.R_rbr_upper_strut]))
        self.ubar_rbr_lower_strut_jcr_strut = (multi_dot([A(config.P_rbr_lower_strut).T,config.pt1_jcr_strut]) + -1*multi_dot([A(config.P_rbr_lower_strut).T,config.R_rbr_lower_strut]))
        self.ubar_rbr_upper_strut_far_strut = (multi_dot([A(config.P_rbr_upper_strut).T,config.pt1_far_strut]) + -1*multi_dot([A(config.P_rbr_upper_strut).T,config.R_rbr_upper_strut]))
        self.ubar_rbr_lower_strut_far_strut = (multi_dot([A(config.P_rbr_lower_strut).T,config.pt2_far_strut]) + -1*multi_dot([A(config.P_rbr_lower_strut).T,config.R_rbr_lower_strut]))
        self.Mbar_rbl_upper_strut_jcl_strut_chassis = multi_dot([A(config.P_rbl_upper_strut).T,triad(config.ax1_jcl_strut_chassis)])
        self.Mbar_vbs_chassis_jcl_strut_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax2_jcl_strut_chassis,triad(config.ax1_jcl_strut_chassis)[0:3,1:2])])
        self.ubar_rbl_upper_strut_jcl_strut_chassis = (multi_dot([A(config.P_rbl_upper_strut).T,config.pt1_jcl_strut_chassis]) + -1*multi_dot([A(config.P_rbl_upper_strut).T,config.R_rbl_upper_strut]))
        self.ubar_vbs_chassis_jcl_strut_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_strut_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_upper_strut_jcl_strut = multi_dot([A(config.P_rbl_upper_strut).T,triad(config.ax1_jcl_strut)])
        self.Mbar_rbl_lower_strut_jcl_strut = multi_dot([A(config.P_rbl_lower_strut).T,triad(config.ax1_jcl_strut)])
        self.ubar_rbl_upper_strut_jcl_strut = (multi_dot([A(config.P_rbl_upper_strut).T,config.pt1_jcl_strut]) + -1*multi_dot([A(config.P_rbl_upper_strut).T,config.R_rbl_upper_strut]))
        self.ubar_rbl_lower_strut_jcl_strut = (multi_dot([A(config.P_rbl_lower_strut).T,config.pt1_jcl_strut]) + -1*multi_dot([A(config.P_rbl_lower_strut).T,config.R_rbl_lower_strut]))
        self.ubar_rbl_upper_strut_fal_strut = (multi_dot([A(config.P_rbl_upper_strut).T,config.pt1_fal_strut]) + -1*multi_dot([A(config.P_rbl_upper_strut).T,config.R_rbl_upper_strut]))
        self.ubar_rbl_lower_strut_fal_strut = (multi_dot([A(config.P_rbl_lower_strut).T,config.pt2_fal_strut]) + -1*multi_dot([A(config.P_rbl_lower_strut).T,config.R_rbl_lower_strut]))
        self.Mbar_rbr_lower_strut_jcr_strut_rocker = multi_dot([A(config.P_rbr_lower_strut).T,triad(config.ax1_jcr_strut_rocker)])
        self.Mbar_rbr_rocker_jcr_strut_rocker = multi_dot([A(config.P_rbr_rocker).T,triad(config.ax2_jcr_strut_rocker,triad(config.ax1_jcr_strut_rocker)[0:3,1:2])])
        self.ubar_rbr_lower_strut_jcr_strut_rocker = (multi_dot([A(config.P_rbr_lower_strut).T,config.pt1_jcr_strut_rocker]) + -1*multi_dot([A(config.P_rbr_lower_strut).T,config.R_rbr_lower_strut]))
        self.ubar_rbr_rocker_jcr_strut_rocker = (multi_dot([A(config.P_rbr_rocker).T,config.pt1_jcr_strut_rocker]) + -1*multi_dot([A(config.P_rbr_rocker).T,config.R_rbr_rocker]))
        self.Mbar_rbl_lower_strut_jcl_strut_rocker = multi_dot([A(config.P_rbl_lower_strut).T,triad(config.ax1_jcl_strut_rocker)])
        self.Mbar_rbl_rocker_jcl_strut_rocker = multi_dot([A(config.P_rbl_rocker).T,triad(config.ax2_jcl_strut_rocker,triad(config.ax1_jcl_strut_rocker)[0:3,1:2])])
        self.ubar_rbl_lower_strut_jcl_strut_rocker = (multi_dot([A(config.P_rbl_lower_strut).T,config.pt1_jcl_strut_rocker]) + -1*multi_dot([A(config.P_rbl_lower_strut).T,config.R_rbl_lower_strut]))
        self.ubar_rbl_rocker_jcl_strut_rocker = (multi_dot([A(config.P_rbl_rocker).T,config.pt1_jcl_strut_rocker]) + -1*multi_dot([A(config.P_rbl_rocker).T,config.R_rbl_rocker]))
        self.Mbar_rbr_tie_rod_jcr_tie_upright = multi_dot([A(config.P_rbr_tie_rod).T,triad(config.ax1_jcr_tie_upright)])
        self.Mbar_rbr_upright_jcr_tie_upright = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_tie_upright)])
        self.ubar_rbr_tie_rod_jcr_tie_upright = (multi_dot([A(config.P_rbr_tie_rod).T,config.pt1_jcr_tie_upright]) + -1*multi_dot([A(config.P_rbr_tie_rod).T,config.R_rbr_tie_rod]))
        self.ubar_rbr_upright_jcr_tie_upright = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_tie_upright]) + -1*multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.Mbar_rbr_tie_rod_jcr_tie_steering = multi_dot([A(config.P_rbr_tie_rod).T,triad(config.ax1_jcr_tie_steering)])
        self.Mbar_vbr_steer_jcr_tie_steering = multi_dot([A(config.P_vbr_steer).T,triad(config.ax2_jcr_tie_steering,triad(config.ax1_jcr_tie_steering)[0:3,1:2])])
        self.ubar_rbr_tie_rod_jcr_tie_steering = (multi_dot([A(config.P_rbr_tie_rod).T,config.pt1_jcr_tie_steering]) + -1*multi_dot([A(config.P_rbr_tie_rod).T,config.R_rbr_tie_rod]))
        self.ubar_vbr_steer_jcr_tie_steering = (multi_dot([A(config.P_vbr_steer).T,config.pt1_jcr_tie_steering]) + -1*multi_dot([A(config.P_vbr_steer).T,config.R_vbr_steer]))
        self.Mbar_rbl_tie_rod_jcl_tie_upright = multi_dot([A(config.P_rbl_tie_rod).T,triad(config.ax1_jcl_tie_upright)])
        self.Mbar_rbl_upright_jcl_tie_upright = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_tie_upright)])
        self.ubar_rbl_tie_rod_jcl_tie_upright = (multi_dot([A(config.P_rbl_tie_rod).T,config.pt1_jcl_tie_upright]) + -1*multi_dot([A(config.P_rbl_tie_rod).T,config.R_rbl_tie_rod]))
        self.ubar_rbl_upright_jcl_tie_upright = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_tie_upright]) + -1*multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.Mbar_rbl_tie_rod_jcl_tie_steering = multi_dot([A(config.P_rbl_tie_rod).T,triad(config.ax1_jcl_tie_steering)])
        self.Mbar_vbl_steer_jcl_tie_steering = multi_dot([A(config.P_vbl_steer).T,triad(config.ax2_jcl_tie_steering,triad(config.ax1_jcl_tie_steering)[0:3,1:2])])
        self.ubar_rbl_tie_rod_jcl_tie_steering = (multi_dot([A(config.P_rbl_tie_rod).T,config.pt1_jcl_tie_steering]) + -1*multi_dot([A(config.P_rbl_tie_rod).T,config.R_rbl_tie_rod]))
        self.ubar_vbl_steer_jcl_tie_steering = (multi_dot([A(config.P_vbl_steer).T,config.pt1_jcl_tie_steering]) + -1*multi_dot([A(config.P_vbl_steer).T,config.R_vbl_steer]))
        self.ubar_rbr_hub_far_tire = (multi_dot([A(config.P_rbr_hub).T,config.pt1_far_tire]) + -1*multi_dot([A(config.P_rbr_hub).T,config.R_rbr_hub]))
        self.ubar_vbs_ground_far_tire = (multi_dot([A(config.P_vbs_ground).T,config.pt1_far_tire]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))
        self.ubar_rbr_hub_far_drive = (multi_dot([A(config.P_rbr_hub).T,config.pt1_far_drive]) + -1*multi_dot([A(config.P_rbr_hub).T,config.R_rbr_hub]))
        self.ubar_vbs_ground_far_drive = (multi_dot([A(config.P_vbs_ground).T,config.pt1_far_drive]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))
        self.ubar_rbl_hub_fal_tire = (multi_dot([A(config.P_rbl_hub).T,config.pt1_fal_tire]) + -1*multi_dot([A(config.P_rbl_hub).T,config.R_rbl_hub]))
        self.ubar_vbs_ground_fal_tire = (multi_dot([A(config.P_vbs_ground).T,config.pt1_fal_tire]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))
        self.ubar_rbl_hub_fal_drive = (multi_dot([A(config.P_rbl_hub).T,config.pt1_fal_drive]) + -1*multi_dot([A(config.P_rbl_hub).T,config.R_rbl_hub]))
        self.ubar_vbs_ground_fal_drive = (multi_dot([A(config.P_vbs_ground).T,config.pt1_fal_drive]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))

    
    def set_gen_coordinates(self,q):
        self.R_rbr_uca = q[0:3,0:1]
        self.P_rbr_uca = q[3:7,0:1]
        self.R_rbl_uca = q[7:10,0:1]
        self.P_rbl_uca = q[10:14,0:1]
        self.R_rbr_lca = q[14:17,0:1]
        self.P_rbr_lca = q[17:21,0:1]
        self.R_rbl_lca = q[21:24,0:1]
        self.P_rbl_lca = q[24:28,0:1]
        self.R_rbr_upright = q[28:31,0:1]
        self.P_rbr_upright = q[31:35,0:1]
        self.R_rbl_upright = q[35:38,0:1]
        self.P_rbl_upright = q[38:42,0:1]
        self.R_rbr_pushrod = q[42:45,0:1]
        self.P_rbr_pushrod = q[45:49,0:1]
        self.R_rbl_pushrod = q[49:52,0:1]
        self.P_rbl_pushrod = q[52:56,0:1]
        self.R_rbr_rocker = q[56:59,0:1]
        self.P_rbr_rocker = q[59:63,0:1]
        self.R_rbl_rocker = q[63:66,0:1]
        self.P_rbl_rocker = q[66:70,0:1]
        self.R_rbr_upper_strut = q[70:73,0:1]
        self.P_rbr_upper_strut = q[73:77,0:1]
        self.R_rbl_upper_strut = q[77:80,0:1]
        self.P_rbl_upper_strut = q[80:84,0:1]
        self.R_rbr_lower_strut = q[84:87,0:1]
        self.P_rbr_lower_strut = q[87:91,0:1]
        self.R_rbl_lower_strut = q[91:94,0:1]
        self.P_rbl_lower_strut = q[94:98,0:1]
        self.R_rbr_tie_rod = q[98:101,0:1]
        self.P_rbr_tie_rod = q[101:105,0:1]
        self.R_rbl_tie_rod = q[105:108,0:1]
        self.P_rbl_tie_rod = q[108:112,0:1]
        self.R_rbr_hub = q[112:115,0:1]
        self.P_rbr_hub = q[115:119,0:1]
        self.R_rbl_hub = q[119:122,0:1]
        self.P_rbl_hub = q[122:126,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbr_uca = qd[0:3,0:1]
        self.Pd_rbr_uca = qd[3:7,0:1]
        self.Rd_rbl_uca = qd[7:10,0:1]
        self.Pd_rbl_uca = qd[10:14,0:1]
        self.Rd_rbr_lca = qd[14:17,0:1]
        self.Pd_rbr_lca = qd[17:21,0:1]
        self.Rd_rbl_lca = qd[21:24,0:1]
        self.Pd_rbl_lca = qd[24:28,0:1]
        self.Rd_rbr_upright = qd[28:31,0:1]
        self.Pd_rbr_upright = qd[31:35,0:1]
        self.Rd_rbl_upright = qd[35:38,0:1]
        self.Pd_rbl_upright = qd[38:42,0:1]
        self.Rd_rbr_pushrod = qd[42:45,0:1]
        self.Pd_rbr_pushrod = qd[45:49,0:1]
        self.Rd_rbl_pushrod = qd[49:52,0:1]
        self.Pd_rbl_pushrod = qd[52:56,0:1]
        self.Rd_rbr_rocker = qd[56:59,0:1]
        self.Pd_rbr_rocker = qd[59:63,0:1]
        self.Rd_rbl_rocker = qd[63:66,0:1]
        self.Pd_rbl_rocker = qd[66:70,0:1]
        self.Rd_rbr_upper_strut = qd[70:73,0:1]
        self.Pd_rbr_upper_strut = qd[73:77,0:1]
        self.Rd_rbl_upper_strut = qd[77:80,0:1]
        self.Pd_rbl_upper_strut = qd[80:84,0:1]
        self.Rd_rbr_lower_strut = qd[84:87,0:1]
        self.Pd_rbr_lower_strut = qd[87:91,0:1]
        self.Rd_rbl_lower_strut = qd[91:94,0:1]
        self.Pd_rbl_lower_strut = qd[94:98,0:1]
        self.Rd_rbr_tie_rod = qd[98:101,0:1]
        self.Pd_rbr_tie_rod = qd[101:105,0:1]
        self.Rd_rbl_tie_rod = qd[105:108,0:1]
        self.Pd_rbl_tie_rod = qd[108:112,0:1]
        self.Rd_rbr_hub = qd[112:115,0:1]
        self.Pd_rbr_hub = qd[115:119,0:1]
        self.Rd_rbl_hub = qd[119:122,0:1]
        self.Pd_rbl_hub = qd[122:126,0:1]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_rbr_uca = qdd[0:3,0:1]
        self.Pdd_rbr_uca = qdd[3:7,0:1]
        self.Rdd_rbl_uca = qdd[7:10,0:1]
        self.Pdd_rbl_uca = qdd[10:14,0:1]
        self.Rdd_rbr_lca = qdd[14:17,0:1]
        self.Pdd_rbr_lca = qdd[17:21,0:1]
        self.Rdd_rbl_lca = qdd[21:24,0:1]
        self.Pdd_rbl_lca = qdd[24:28,0:1]
        self.Rdd_rbr_upright = qdd[28:31,0:1]
        self.Pdd_rbr_upright = qdd[31:35,0:1]
        self.Rdd_rbl_upright = qdd[35:38,0:1]
        self.Pdd_rbl_upright = qdd[38:42,0:1]
        self.Rdd_rbr_pushrod = qdd[42:45,0:1]
        self.Pdd_rbr_pushrod = qdd[45:49,0:1]
        self.Rdd_rbl_pushrod = qdd[49:52,0:1]
        self.Pdd_rbl_pushrod = qdd[52:56,0:1]
        self.Rdd_rbr_rocker = qdd[56:59,0:1]
        self.Pdd_rbr_rocker = qdd[59:63,0:1]
        self.Rdd_rbl_rocker = qdd[63:66,0:1]
        self.Pdd_rbl_rocker = qdd[66:70,0:1]
        self.Rdd_rbr_upper_strut = qdd[70:73,0:1]
        self.Pdd_rbr_upper_strut = qdd[73:77,0:1]
        self.Rdd_rbl_upper_strut = qdd[77:80,0:1]
        self.Pdd_rbl_upper_strut = qdd[80:84,0:1]
        self.Rdd_rbr_lower_strut = qdd[84:87,0:1]
        self.Pdd_rbr_lower_strut = qdd[87:91,0:1]
        self.Rdd_rbl_lower_strut = qdd[91:94,0:1]
        self.Pdd_rbl_lower_strut = qdd[94:98,0:1]
        self.Rdd_rbr_tie_rod = qdd[98:101,0:1]
        self.Pdd_rbr_tie_rod = qdd[101:105,0:1]
        self.Rdd_rbl_tie_rod = qdd[105:108,0:1]
        self.Pdd_rbl_tie_rod = qdd[108:112,0:1]
        self.Rdd_rbr_hub = qdd[112:115,0:1]
        self.Pdd_rbr_hub = qdd[115:119,0:1]
        self.Rdd_rbl_hub = qdd[119:122,0:1]
        self.Pdd_rbl_hub = qdd[122:126,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcr_uca_upright = Lambda[0:3,0:1]
        self.L_jcr_uca_chassis = Lambda[3:8,0:1]
        self.L_jcr_prod_uca = Lambda[8:12,0:1]
        self.L_jcl_uca_upright = Lambda[12:15,0:1]
        self.L_jcl_uca_chassis = Lambda[15:20,0:1]
        self.L_jcl_prod_uca = Lambda[20:24,0:1]
        self.L_jcr_lca_upright = Lambda[24:27,0:1]
        self.L_jcr_lca_chassis = Lambda[27:32,0:1]
        self.L_jcl_lca_upright = Lambda[32:35,0:1]
        self.L_jcl_lca_chassis = Lambda[35:40,0:1]
        self.L_jcr_hub_bearing = Lambda[40:45,0:1]
        self.L_jcl_hub_bearing = Lambda[45:50,0:1]
        self.L_jcr_prod_rocker = Lambda[50:53,0:1]
        self.L_jcr_rocker_chassis = Lambda[53:58,0:1]
        self.L_jcl_prod_rocker = Lambda[58:61,0:1]
        self.L_jcl_rocker_chassis = Lambda[61:66,0:1]
        self.L_jcr_strut_chassis = Lambda[66:70,0:1]
        self.L_jcr_strut = Lambda[70:74,0:1]
        self.L_jcl_strut_chassis = Lambda[74:78,0:1]
        self.L_jcl_strut = Lambda[78:82,0:1]
        self.L_jcr_strut_rocker = Lambda[82:86,0:1]
        self.L_jcl_strut_rocker = Lambda[86:90,0:1]
        self.L_jcr_tie_upright = Lambda[90:93,0:1]
        self.L_jcr_tie_steering = Lambda[93:97,0:1]
        self.L_jcl_tie_upright = Lambda[97:100,0:1]
        self.L_jcl_tie_steering = Lambda[100:104,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_rbr_uca
        x1 = self.R_rbr_upright
        x2 = -1*x1
        x3 = self.P_rbr_uca
        x4 = A(x3)
        x5 = self.P_rbr_upright
        x6 = A(x5)
        x7 = -1*self.R_vbs_chassis
        x8 = A(self.P_vbs_chassis)
        x9 = x4.T
        x10 = self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        x11 = -1*self.R_rbr_pushrod
        x12 = self.P_rbr_pushrod
        x13 = A(x12)
        x14 = self.R_rbl_uca
        x15 = self.R_rbl_upright
        x16 = -1*x15
        x17 = self.P_rbl_uca
        x18 = A(x17)
        x19 = self.P_rbl_upright
        x20 = A(x19)
        x21 = x18.T
        x22 = self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        x23 = -1*self.R_rbl_pushrod
        x24 = self.P_rbl_pushrod
        x25 = A(x24)
        x26 = self.R_rbr_lca
        x27 = self.P_rbr_lca
        x28 = A(x27)
        x29 = x28.T
        x30 = self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        x31 = self.R_rbl_lca
        x32 = self.P_rbl_lca
        x33 = A(x32)
        x34 = x33.T
        x35 = self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        x36 = self.P_rbr_hub
        x37 = A(x36)
        x38 = x6.T
        x39 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        x40 = self.P_rbl_hub
        x41 = A(x40)
        x42 = x20.T
        x43 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        x44 = self.R_rbr_rocker
        x45 = self.P_rbr_rocker
        x46 = A(x45)
        x47 = x46.T
        x48 = self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]
        x49 = self.R_rbl_rocker
        x50 = self.P_rbl_rocker
        x51 = A(x50)
        x52 = x51.T
        x53 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]
        x54 = self.R_rbr_upper_strut
        x55 = self.P_rbr_upper_strut
        x56 = A(x55)
        x57 = x56.T
        x58 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1].T
        x59 = self.P_rbr_lower_strut
        x60 = A(x59)
        x61 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        x62 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2].T
        x63 = self.R_rbr_lower_strut
        x64 = (x54 + -1*x63 + multi_dot([x56,self.ubar_rbr_upper_strut_jcr_strut]) + -1*multi_dot([x60,self.ubar_rbr_lower_strut_jcr_strut]))
        x65 = self.R_rbl_upper_strut
        x66 = self.P_rbl_upper_strut
        x67 = A(x66)
        x68 = x67.T
        x69 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1].T
        x70 = self.P_rbl_lower_strut
        x71 = A(x70)
        x72 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        x73 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2].T
        x74 = self.R_rbl_lower_strut
        x75 = (x65 + -1*x74 + multi_dot([x67,self.ubar_rbl_upper_strut_jcl_strut]) + -1*multi_dot([x71,self.ubar_rbl_lower_strut_jcl_strut]))
        x76 = self.R_rbr_tie_rod
        x77 = self.P_rbr_tie_rod
        x78 = A(x77)
        x79 = A(self.P_vbr_steer)
        x80 = self.R_rbl_tie_rod
        x81 = self.P_rbl_tie_rod
        x82 = A(x81)
        x83 = A(self.P_vbl_steer)
        x84 = -1*np.eye(1, dtype=np.float64)

        self.pos_eq_blocks = ((x0 + x2 + multi_dot([x4,self.ubar_rbr_uca_jcr_uca_upright]) + -1*multi_dot([x6,self.ubar_rbr_upright_jcr_uca_upright])),
        (x0 + x7 + multi_dot([x4,self.ubar_rbr_uca_jcr_uca_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcr_uca_chassis])),
        multi_dot([self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1].T,x9,x8,x10]),
        multi_dot([self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2].T,x9,x8,x10]),
        (x0 + x11 + multi_dot([x4,self.ubar_rbr_uca_jcr_prod_uca]) + -1*multi_dot([x13,self.ubar_rbr_pushrod_jcr_prod_uca])),
        multi_dot([self.Mbar_rbr_uca_jcr_prod_uca[:,0:1].T,x9,x13,self.Mbar_rbr_pushrod_jcr_prod_uca[:,0:1]]),
        (x14 + x16 + multi_dot([x18,self.ubar_rbl_uca_jcl_uca_upright]) + -1*multi_dot([x20,self.ubar_rbl_upright_jcl_uca_upright])),
        (x14 + x7 + multi_dot([x18,self.ubar_rbl_uca_jcl_uca_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcl_uca_chassis])),
        multi_dot([self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1].T,x21,x8,x22]),
        multi_dot([self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2].T,x21,x8,x22]),
        (x14 + x23 + multi_dot([x18,self.ubar_rbl_uca_jcl_prod_uca]) + -1*multi_dot([x25,self.ubar_rbl_pushrod_jcl_prod_uca])),
        multi_dot([self.Mbar_rbl_uca_jcl_prod_uca[:,0:1].T,x21,x25,self.Mbar_rbl_pushrod_jcl_prod_uca[:,0:1]]),
        (x26 + x2 + multi_dot([x28,self.ubar_rbr_lca_jcr_lca_upright]) + -1*multi_dot([x6,self.ubar_rbr_upright_jcr_lca_upright])),
        (x26 + x7 + multi_dot([x28,self.ubar_rbr_lca_jcr_lca_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcr_lca_chassis])),
        multi_dot([self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1].T,x29,x8,x30]),
        multi_dot([self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2].T,x29,x8,x30]),
        (x31 + x16 + multi_dot([x33,self.ubar_rbl_lca_jcl_lca_upright]) + -1*multi_dot([x20,self.ubar_rbl_upright_jcl_lca_upright])),
        (x31 + x7 + multi_dot([x33,self.ubar_rbl_lca_jcl_lca_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcl_lca_chassis])),
        multi_dot([self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1].T,x34,x8,x35]),
        multi_dot([self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2].T,x34,x8,x35]),
        (x1 + -1*self.R_rbr_hub + multi_dot([x6,self.ubar_rbr_upright_jcr_hub_bearing]) + -1*multi_dot([x37,self.ubar_rbr_hub_jcr_hub_bearing])),
        multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x38,x37,x39]),
        multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x38,x37,x39]),
        (x15 + -1*self.R_rbl_hub + multi_dot([x20,self.ubar_rbl_upright_jcl_hub_bearing]) + -1*multi_dot([x41,self.ubar_rbl_hub_jcl_hub_bearing])),
        multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x42,x41,x43]),
        multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x42,x41,x43]),
        (x44 + x11 + multi_dot([x46,self.ubar_rbr_rocker_jcr_prod_rocker]) + -1*multi_dot([x13,self.ubar_rbr_pushrod_jcr_prod_rocker])),
        (x44 + x7 + multi_dot([x46,self.ubar_rbr_rocker_jcr_rocker_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcr_rocker_chassis])),
        multi_dot([self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1].T,x47,x8,x48]),
        multi_dot([self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2].T,x47,x8,x48]),
        (x49 + x23 + multi_dot([x51,self.ubar_rbl_rocker_jcl_prod_rocker]) + -1*multi_dot([x25,self.ubar_rbl_pushrod_jcl_prod_rocker])),
        (x49 + x7 + multi_dot([x51,self.ubar_rbl_rocker_jcl_rocker_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcl_rocker_chassis])),
        multi_dot([self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1].T,x52,x8,x53]),
        multi_dot([self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2].T,x52,x8,x53]),
        (x54 + x7 + multi_dot([x56,self.ubar_rbr_upper_strut_jcr_strut_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcr_strut_chassis])),
        multi_dot([self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1].T,x57,x8,self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]]),
        multi_dot([x58,x57,x60,x61]),
        multi_dot([x62,x57,x60,x61]),
        multi_dot([x58,x57,x64]),
        multi_dot([x62,x57,x64]),
        (x65 + x7 + multi_dot([x67,self.ubar_rbl_upper_strut_jcl_strut_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcl_strut_chassis])),
        multi_dot([self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1].T,x68,x8,self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]]),
        multi_dot([x69,x68,x71,x72]),
        multi_dot([x73,x68,x71,x72]),
        multi_dot([x69,x68,x75]),
        multi_dot([x73,x68,x75]),
        (x63 + -1*x44 + multi_dot([x60,self.ubar_rbr_lower_strut_jcr_strut_rocker]) + -1*multi_dot([x46,self.ubar_rbr_rocker_jcr_strut_rocker])),
        multi_dot([self.Mbar_rbr_lower_strut_jcr_strut_rocker[:,0:1].T,x60.T,x46,self.Mbar_rbr_rocker_jcr_strut_rocker[:,0:1]]),
        (x74 + -1*x49 + multi_dot([x71,self.ubar_rbl_lower_strut_jcl_strut_rocker]) + -1*multi_dot([x51,self.ubar_rbl_rocker_jcl_strut_rocker])),
        multi_dot([self.Mbar_rbl_lower_strut_jcl_strut_rocker[:,0:1].T,x71.T,x51,self.Mbar_rbl_rocker_jcl_strut_rocker[:,0:1]]),
        (x76 + x2 + multi_dot([x78,self.ubar_rbr_tie_rod_jcr_tie_upright]) + -1*multi_dot([x6,self.ubar_rbr_upright_jcr_tie_upright])),
        (x76 + -1*self.R_vbr_steer + multi_dot([x78,self.ubar_rbr_tie_rod_jcr_tie_steering]) + -1*multi_dot([x79,self.ubar_vbr_steer_jcr_tie_steering])),
        multi_dot([self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x78.T,x79,self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]]),
        (x80 + x16 + multi_dot([x82,self.ubar_rbl_tie_rod_jcl_tie_upright]) + -1*multi_dot([x20,self.ubar_rbl_upright_jcl_tie_upright])),
        (x80 + -1*self.R_vbl_steer + multi_dot([x82,self.ubar_rbl_tie_rod_jcl_tie_steering]) + -1*multi_dot([x83,self.ubar_vbl_steer_jcl_tie_steering])),
        multi_dot([self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1].T,x82.T,x83,self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]]),
        (x84 + multi_dot([x3.T,x3])),
        (x84 + multi_dot([x17.T,x17])),
        (x84 + multi_dot([x27.T,x27])),
        (x84 + multi_dot([x32.T,x32])),
        (x84 + multi_dot([x5.T,x5])),
        (x84 + multi_dot([x19.T,x19])),
        (x84 + multi_dot([x12.T,x12])),
        (x84 + multi_dot([x24.T,x24])),
        (x84 + multi_dot([x45.T,x45])),
        (x84 + multi_dot([x50.T,x50])),
        (x84 + multi_dot([x55.T,x55])),
        (x84 + multi_dot([x66.T,x66])),
        (x84 + multi_dot([x59.T,x59])),
        (x84 + multi_dot([x70.T,x70])),
        (x84 + multi_dot([x77.T,x77])),
        (x84 + multi_dot([x81.T,x81])),
        (x84 + multi_dot([x36.T,x36])),
        (x84 + multi_dot([x40.T,x40])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = (v0,
        v0,
        v1,
        v1,
        v0,
        v1,
        v0,
        v0,
        v1,
        v1,
        v0,
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
        a14 = self.Pd_rbr_pushrod
        a15 = self.Mbar_rbr_uca_jcr_prod_uca[:,0:1]
        a16 = self.Mbar_rbr_pushrod_jcr_prod_uca[:,0:1]
        a17 = self.P_rbr_pushrod
        a18 = self.Pd_rbl_uca
        a19 = self.Pd_rbl_upright
        a20 = self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        a21 = self.P_rbl_uca
        a22 = A(a21).T
        a23 = self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        a24 = B(a2,a23)
        a25 = a23.T
        a26 = a18.T
        a27 = B(a9,a23)
        a28 = self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        a29 = self.Pd_rbl_pushrod
        a30 = self.Mbar_rbl_uca_jcl_prod_uca[:,0:1]
        a31 = self.Mbar_rbl_pushrod_jcl_prod_uca[:,0:1]
        a32 = self.P_rbl_pushrod
        a33 = self.Pd_rbr_lca
        a34 = self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        a35 = self.P_rbr_lca
        a36 = A(a35).T
        a37 = self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        a38 = B(a2,a37)
        a39 = a37.T
        a40 = a33.T
        a41 = B(a9,a37)
        a42 = self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        a43 = self.Pd_rbl_lca
        a44 = self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        a45 = self.P_rbl_lca
        a46 = A(a45).T
        a47 = self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        a48 = B(a2,a47)
        a49 = a47.T
        a50 = a43.T
        a51 = B(a9,a47)
        a52 = self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        a53 = self.Pd_rbr_hub
        a54 = self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a55 = self.P_rbr_upright
        a56 = A(a55).T
        a57 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        a58 = B(a53,a57)
        a59 = a57.T
        a60 = self.P_rbr_hub
        a61 = A(a60).T
        a62 = a1.T
        a63 = B(a60,a57)
        a64 = self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a65 = self.Pd_rbl_hub
        a66 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a67 = self.P_rbl_upright
        a68 = A(a67).T
        a69 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        a70 = B(a65,a69)
        a71 = a69.T
        a72 = self.P_rbl_hub
        a73 = A(a72).T
        a74 = a19.T
        a75 = B(a72,a69)
        a76 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a77 = self.Pd_rbr_rocker
        a78 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1]
        a79 = self.P_rbr_rocker
        a80 = A(a79).T
        a81 = self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]
        a82 = B(a2,a81)
        a83 = a81.T
        a84 = a77.T
        a85 = B(a9,a81)
        a86 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2]
        a87 = self.Pd_rbl_rocker
        a88 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]
        a89 = self.P_rbl_rocker
        a90 = A(a89).T
        a91 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]
        a92 = B(a2,a91)
        a93 = a91.T
        a94 = a87.T
        a95 = B(a9,a91)
        a96 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]
        a97 = self.Pd_rbr_upper_strut
        a98 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        a99 = self.P_rbr_upper_strut
        a100 = A(a99).T
        a101 = self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        a102 = a97.T
        a103 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        a104 = a103.T
        a105 = self.Pd_rbr_lower_strut
        a106 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        a107 = B(a105,a106)
        a108 = a106.T
        a109 = self.P_rbr_lower_strut
        a110 = A(a109).T
        a111 = B(a97,a103)
        a112 = B(a99,a103).T
        a113 = B(a109,a106)
        a114 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        a115 = a114.T
        a116 = B(a97,a114)
        a117 = B(a99,a114).T
        a118 = self.ubar_rbr_upper_strut_jcr_strut
        a119 = self.ubar_rbr_lower_strut_jcr_strut
        a120 = (multi_dot([B(a97,a118),a97]) + -1*multi_dot([B(a105,a119),a105]))
        a121 = (self.Rd_rbr_upper_strut + -1*self.Rd_rbr_lower_strut + multi_dot([B(a99,a118),a97]) + -1*multi_dot([B(a109,a119),a105]))
        a122 = (self.R_rbr_upper_strut.T + -1*self.R_rbr_lower_strut.T + multi_dot([a118.T,a100]) + -1*multi_dot([a119.T,a110]))
        a123 = self.Pd_rbl_upper_strut
        a124 = self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        a125 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        a126 = self.P_rbl_upper_strut
        a127 = A(a126).T
        a128 = a123.T
        a129 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        a130 = a129.T
        a131 = self.P_rbl_lower_strut
        a132 = A(a131).T
        a133 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        a134 = B(a123,a133)
        a135 = a133.T
        a136 = self.Pd_rbl_lower_strut
        a137 = B(a136,a129)
        a138 = B(a126,a133).T
        a139 = B(a131,a129)
        a140 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        a141 = B(a123,a140)
        a142 = a140.T
        a143 = B(a126,a140).T
        a144 = self.ubar_rbl_upper_strut_jcl_strut
        a145 = self.ubar_rbl_lower_strut_jcl_strut
        a146 = (multi_dot([B(a123,a144),a123]) + -1*multi_dot([B(a136,a145),a136]))
        a147 = (self.Rd_rbl_upper_strut + -1*self.Rd_rbl_lower_strut + multi_dot([B(a126,a144),a123]) + -1*multi_dot([B(a131,a145),a136]))
        a148 = (self.R_rbl_upper_strut.T + -1*self.R_rbl_lower_strut.T + multi_dot([a144.T,a127]) + -1*multi_dot([a145.T,a132]))
        a149 = self.Mbar_rbr_lower_strut_jcr_strut_rocker[:,0:1]
        a150 = self.Mbar_rbr_rocker_jcr_strut_rocker[:,0:1]
        a151 = a105.T
        a152 = self.Mbar_rbl_lower_strut_jcl_strut_rocker[:,0:1]
        a153 = self.Mbar_rbl_rocker_jcl_strut_rocker[:,0:1]
        a154 = a136.T
        a155 = self.Pd_rbr_tie_rod
        a156 = self.Pd_vbr_steer
        a157 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        a158 = self.P_rbr_tie_rod
        a159 = self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        a160 = self.P_vbr_steer
        a161 = a155.T
        a162 = self.Pd_rbl_tie_rod
        a163 = self.Pd_vbl_steer
        a164 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        a165 = self.P_rbl_tie_rod
        a166 = self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        a167 = self.P_vbl_steer
        a168 = a162.T

        self.acc_eq_blocks = ((multi_dot([B(a0,self.ubar_rbr_uca_jcr_uca_upright),a0]) + -1*multi_dot([B(a1,self.ubar_rbr_upright_jcr_uca_upright),a1])),
        (multi_dot([B(a0,self.ubar_rbr_uca_jcr_uca_chassis),a0]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcr_uca_chassis),a2])),
        (multi_dot([a3.T,a5,a7,a2]) + multi_dot([a8,a10,B(a0,a3),a0]) + 2*multi_dot([a11,B(a4,a3).T,a12,a2])),
        (multi_dot([a13.T,a5,a7,a2]) + multi_dot([a8,a10,B(a0,a13),a0]) + 2*multi_dot([a11,B(a4,a13).T,a12,a2])),
        (multi_dot([B(a0,self.ubar_rbr_uca_jcr_prod_uca),a0]) + -1*multi_dot([B(a14,self.ubar_rbr_pushrod_jcr_prod_uca),a14])),
        (multi_dot([a15.T,a5,B(a14,a16),a14]) + multi_dot([a16.T,A(a17).T,B(a0,a15),a0]) + 2*multi_dot([a11,B(a4,a15).T,B(a17,a16),a14])),
        (multi_dot([B(a18,self.ubar_rbl_uca_jcl_uca_upright),a18]) + -1*multi_dot([B(a19,self.ubar_rbl_upright_jcl_uca_upright),a19])),
        (multi_dot([B(a18,self.ubar_rbl_uca_jcl_uca_chassis),a18]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcl_uca_chassis),a2])),
        (multi_dot([a20.T,a22,a24,a2]) + multi_dot([a25,a10,B(a18,a20),a18]) + 2*multi_dot([a26,B(a21,a20).T,a27,a2])),
        (multi_dot([a28.T,a22,a24,a2]) + multi_dot([a25,a10,B(a18,a28),a18]) + 2*multi_dot([a26,B(a21,a28).T,a27,a2])),
        (multi_dot([B(a18,self.ubar_rbl_uca_jcl_prod_uca),a18]) + -1*multi_dot([B(a29,self.ubar_rbl_pushrod_jcl_prod_uca),a29])),
        (multi_dot([a30.T,a22,B(a29,a31),a29]) + multi_dot([a31.T,A(a32).T,B(a18,a30),a18]) + 2*multi_dot([a26,B(a21,a30).T,B(a32,a31),a29])),
        (multi_dot([B(a33,self.ubar_rbr_lca_jcr_lca_upright),a33]) + -1*multi_dot([B(a1,self.ubar_rbr_upright_jcr_lca_upright),a1])),
        (multi_dot([B(a33,self.ubar_rbr_lca_jcr_lca_chassis),a33]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcr_lca_chassis),a2])),
        (multi_dot([a34.T,a36,a38,a2]) + multi_dot([a39,a10,B(a33,a34),a33]) + 2*multi_dot([a40,B(a35,a34).T,a41,a2])),
        (multi_dot([a42.T,a36,a38,a2]) + multi_dot([a39,a10,B(a33,a42),a33]) + 2*multi_dot([a40,B(a35,a42).T,a41,a2])),
        (multi_dot([B(a43,self.ubar_rbl_lca_jcl_lca_upright),a43]) + -1*multi_dot([B(a19,self.ubar_rbl_upright_jcl_lca_upright),a19])),
        (multi_dot([B(a43,self.ubar_rbl_lca_jcl_lca_chassis),a43]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcl_lca_chassis),a2])),
        (multi_dot([a44.T,a46,a48,a2]) + multi_dot([a49,a10,B(a43,a44),a43]) + 2*multi_dot([a50,B(a45,a44).T,a51,a2])),
        (multi_dot([a52.T,a46,a48,a2]) + multi_dot([a49,a10,B(a43,a52),a43]) + 2*multi_dot([a50,B(a45,a52).T,a51,a2])),
        (multi_dot([B(a1,self.ubar_rbr_upright_jcr_hub_bearing),a1]) + -1*multi_dot([B(a53,self.ubar_rbr_hub_jcr_hub_bearing),a53])),
        (multi_dot([a54.T,a56,a58,a53]) + multi_dot([a59,a61,B(a1,a54),a1]) + 2*multi_dot([a62,B(a55,a54).T,a63,a53])),
        (multi_dot([a64.T,a56,a58,a53]) + multi_dot([a59,a61,B(a1,a64),a1]) + 2*multi_dot([a62,B(a55,a64).T,a63,a53])),
        (multi_dot([B(a19,self.ubar_rbl_upright_jcl_hub_bearing),a19]) + -1*multi_dot([B(a65,self.ubar_rbl_hub_jcl_hub_bearing),a65])),
        (multi_dot([a66.T,a68,a70,a65]) + multi_dot([a71,a73,B(a19,a66),a19]) + 2*multi_dot([a74,B(a67,a66).T,a75,a65])),
        (multi_dot([a76.T,a68,a70,a65]) + multi_dot([a71,a73,B(a19,a76),a19]) + 2*multi_dot([a74,B(a67,a76).T,a75,a65])),
        (multi_dot([B(a77,self.ubar_rbr_rocker_jcr_prod_rocker),a77]) + -1*multi_dot([B(a14,self.ubar_rbr_pushrod_jcr_prod_rocker),a14])),
        (multi_dot([B(a77,self.ubar_rbr_rocker_jcr_rocker_chassis),a77]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcr_rocker_chassis),a2])),
        (multi_dot([a78.T,a80,a82,a2]) + multi_dot([a83,a10,B(a77,a78),a77]) + 2*multi_dot([a84,B(a79,a78).T,a85,a2])),
        (multi_dot([a86.T,a80,a82,a2]) + multi_dot([a83,a10,B(a77,a86),a77]) + 2*multi_dot([a84,B(a79,a86).T,a85,a2])),
        (multi_dot([B(a87,self.ubar_rbl_rocker_jcl_prod_rocker),a87]) + -1*multi_dot([B(a29,self.ubar_rbl_pushrod_jcl_prod_rocker),a29])),
        (multi_dot([B(a87,self.ubar_rbl_rocker_jcl_rocker_chassis),a87]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcl_rocker_chassis),a2])),
        (multi_dot([a88.T,a90,a92,a2]) + multi_dot([a93,a10,B(a87,a88),a87]) + 2*multi_dot([a94,B(a89,a88).T,a95,a2])),
        (multi_dot([a96.T,a90,a92,a2]) + multi_dot([a93,a10,B(a87,a96),a87]) + 2*multi_dot([a94,B(a89,a96).T,a95,a2])),
        (multi_dot([B(a97,self.ubar_rbr_upper_strut_jcr_strut_chassis),a97]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcr_strut_chassis),a2])),
        (multi_dot([a98.T,a100,B(a2,a101),a2]) + multi_dot([a101.T,a10,B(a97,a98),a97]) + 2*multi_dot([a102,B(a99,a98).T,B(a9,a101),a2])),
        (multi_dot([a104,a100,a107,a105]) + multi_dot([a108,a110,a111,a97]) + 2*multi_dot([a102,a112,a113,a105])),
        (multi_dot([a115,a100,a107,a105]) + multi_dot([a108,a110,a116,a97]) + 2*multi_dot([a102,a117,a113,a105])),
        (multi_dot([a104,a100,a120]) + 2*multi_dot([a102,a112,a121]) + multi_dot([a122,a111,a97])),
        (multi_dot([a115,a100,a120]) + 2*multi_dot([a102,a117,a121]) + multi_dot([a122,a116,a97])),
        (multi_dot([B(a123,self.ubar_rbl_upper_strut_jcl_strut_chassis),a123]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcl_strut_chassis),a2])),
        (multi_dot([a124.T,a10,B(a123,a125),a123]) + multi_dot([a125.T,a127,B(a2,a124),a2]) + 2*multi_dot([a128,B(a126,a125).T,B(a9,a124),a2])),
        (multi_dot([a130,a132,a134,a123]) + multi_dot([a135,a127,a137,a136]) + 2*multi_dot([a128,a138,a139,a136])),
        (multi_dot([a130,a132,a141,a123]) + multi_dot([a142,a127,a137,a136]) + 2*multi_dot([a128,a143,a139,a136])),
        (multi_dot([a135,a127,a146]) + 2*multi_dot([a128,a138,a147]) + multi_dot([a148,a134,a123])),
        (multi_dot([a142,a127,a146]) + 2*multi_dot([a128,a143,a147]) + multi_dot([a148,a141,a123])),
        (multi_dot([B(a105,self.ubar_rbr_lower_strut_jcr_strut_rocker),a105]) + -1*multi_dot([B(a77,self.ubar_rbr_rocker_jcr_strut_rocker),a77])),
        (multi_dot([a149.T,a110,B(a77,a150),a77]) + multi_dot([a150.T,a80,B(a105,a149),a105]) + 2*multi_dot([a151,B(a109,a149).T,B(a79,a150),a77])),
        (multi_dot([B(a136,self.ubar_rbl_lower_strut_jcl_strut_rocker),a136]) + -1*multi_dot([B(a87,self.ubar_rbl_rocker_jcl_strut_rocker),a87])),
        (multi_dot([a152.T,a132,B(a87,a153),a87]) + multi_dot([a153.T,a90,B(a136,a152),a136]) + 2*multi_dot([a154,B(a131,a152).T,B(a89,a153),a87])),
        (multi_dot([B(a155,self.ubar_rbr_tie_rod_jcr_tie_upright),a155]) + -1*multi_dot([B(a1,self.ubar_rbr_upright_jcr_tie_upright),a1])),
        (multi_dot([B(a155,self.ubar_rbr_tie_rod_jcr_tie_steering),a155]) + -1*multi_dot([B(a156,self.ubar_vbr_steer_jcr_tie_steering),a156])),
        (multi_dot([a157.T,A(a158).T,B(a156,a159),a156]) + multi_dot([a159.T,A(a160).T,B(a155,a157),a155]) + 2*multi_dot([a161,B(a158,a157).T,B(a160,a159),a156])),
        (multi_dot([B(a162,self.ubar_rbl_tie_rod_jcl_tie_upright),a162]) + -1*multi_dot([B(a19,self.ubar_rbl_upright_jcl_tie_upright),a19])),
        (multi_dot([B(a162,self.ubar_rbl_tie_rod_jcl_tie_steering),a162]) + -1*multi_dot([B(a163,self.ubar_vbl_steer_jcl_tie_steering),a163])),
        (multi_dot([a164.T,A(a165).T,B(a163,a166),a163]) + multi_dot([a166.T,A(a167).T,B(a162,a164),a162]) + 2*multi_dot([a168,B(a165,a164).T,B(a167,a166),a163])),
        2*multi_dot([a11,a0]),
        2*multi_dot([a26,a18]),
        2*multi_dot([a40,a33]),
        2*multi_dot([a50,a43]),
        2*multi_dot([a62,a1]),
        2*multi_dot([a74,a19]),
        2*multi_dot([a14.T,a14]),
        2*multi_dot([a29.T,a29]),
        2*multi_dot([a84,a77]),
        2*multi_dot([a94,a87]),
        2*multi_dot([a102,a97]),
        2*multi_dot([a128,a123]),
        2*multi_dot([a151,a105]),
        2*multi_dot([a154,a136]),
        2*multi_dot([a161,a155]),
        2*multi_dot([a168,a162]),
        2*multi_dot([a53.T,a53]),
        2*multi_dot([a65.T,a65]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3, dtype=np.float64)
        j1 = self.P_rbr_uca
        j2 = -1*j0
        j3 = self.P_rbr_upright
        j4 = np.zeros((1,3),dtype=np.float64)
        j5 = self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        j6 = j5.T
        j7 = self.P_vbs_chassis
        j8 = A(j7).T
        j9 = self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        j10 = self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        j11 = A(j1).T
        j12 = B(j7,j5)
        j13 = self.Mbar_rbr_pushrod_jcr_prod_uca[:,0:1]
        j14 = self.P_rbr_pushrod
        j15 = self.Mbar_rbr_uca_jcr_prod_uca[:,0:1]
        j16 = self.P_rbl_uca
        j17 = self.P_rbl_upright
        j18 = self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        j19 = j18.T
        j20 = self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        j21 = self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        j22 = A(j16).T
        j23 = B(j7,j18)
        j24 = self.Mbar_rbl_pushrod_jcl_prod_uca[:,0:1]
        j25 = self.P_rbl_pushrod
        j26 = self.Mbar_rbl_uca_jcl_prod_uca[:,0:1]
        j27 = self.P_rbr_lca
        j28 = self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        j29 = j28.T
        j30 = self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        j31 = self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        j32 = A(j27).T
        j33 = B(j7,j28)
        j34 = self.P_rbl_lca
        j35 = self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        j36 = j35.T
        j37 = self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        j38 = self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        j39 = A(j34).T
        j40 = B(j7,j35)
        j41 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        j42 = j41.T
        j43 = self.P_rbr_hub
        j44 = A(j43).T
        j45 = self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j46 = self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j47 = A(j3).T
        j48 = B(j43,j41)
        j49 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        j50 = j49.T
        j51 = self.P_rbl_hub
        j52 = A(j51).T
        j53 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j54 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j55 = A(j17).T
        j56 = B(j51,j49)
        j57 = self.P_rbr_rocker
        j58 = self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]
        j59 = j58.T
        j60 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1]
        j61 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2]
        j62 = A(j57).T
        j63 = B(j7,j58)
        j64 = self.P_rbl_rocker
        j65 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]
        j66 = j65.T
        j67 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]
        j68 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]
        j69 = A(j64).T
        j70 = B(j7,j65)
        j71 = self.P_rbr_upper_strut
        j72 = self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        j73 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        j74 = A(j71).T
        j75 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        j76 = j75.T
        j77 = self.P_rbr_lower_strut
        j78 = A(j77).T
        j79 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        j80 = B(j71,j79)
        j81 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        j82 = B(j71,j81)
        j83 = j79.T
        j84 = multi_dot([j83,j74])
        j85 = self.ubar_rbr_upper_strut_jcr_strut
        j86 = B(j71,j85)
        j87 = self.ubar_rbr_lower_strut_jcr_strut
        j88 = (self.R_rbr_upper_strut.T + -1*self.R_rbr_lower_strut.T + multi_dot([j85.T,j74]) + -1*multi_dot([j87.T,j78]))
        j89 = j81.T
        j90 = multi_dot([j89,j74])
        j91 = B(j77,j75)
        j92 = B(j77,j87)
        j93 = self.P_rbl_upper_strut
        j94 = self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        j95 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        j96 = A(j93).T
        j97 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        j98 = j97.T
        j99 = self.P_rbl_lower_strut
        j100 = A(j99).T
        j101 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        j102 = B(j93,j101)
        j103 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        j104 = B(j93,j103)
        j105 = j101.T
        j106 = multi_dot([j105,j96])
        j107 = self.ubar_rbl_upper_strut_jcl_strut
        j108 = B(j93,j107)
        j109 = self.ubar_rbl_lower_strut_jcl_strut
        j110 = (self.R_rbl_upper_strut.T + -1*self.R_rbl_lower_strut.T + multi_dot([j107.T,j96]) + -1*multi_dot([j109.T,j100]))
        j111 = j103.T
        j112 = multi_dot([j111,j96])
        j113 = B(j99,j97)
        j114 = B(j99,j109)
        j115 = self.Mbar_rbr_rocker_jcr_strut_rocker[:,0:1]
        j116 = self.Mbar_rbr_lower_strut_jcr_strut_rocker[:,0:1]
        j117 = self.Mbar_rbl_rocker_jcl_strut_rocker[:,0:1]
        j118 = self.Mbar_rbl_lower_strut_jcl_strut_rocker[:,0:1]
        j119 = self.P_rbr_tie_rod
        j120 = self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        j121 = self.P_vbr_steer
        j122 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        j123 = self.P_rbl_tie_rod
        j124 = self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        j125 = self.P_vbl_steer
        j126 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]

        self.jac_eq_blocks = (j0,
        B(j1,self.ubar_rbr_uca_jcr_uca_upright),
        j2,
        -1*B(j3,self.ubar_rbr_upright_jcr_uca_upright),
        j0,
        B(j1,self.ubar_rbr_uca_jcr_uca_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcr_uca_chassis),
        j4,
        multi_dot([j6,j8,B(j1,j9)]),
        j4,
        multi_dot([j9.T,j11,j12]),
        j4,
        multi_dot([j6,j8,B(j1,j10)]),
        j4,
        multi_dot([j10.T,j11,j12]),
        j0,
        B(j1,self.ubar_rbr_uca_jcr_prod_uca),
        j2,
        -1*B(j14,self.ubar_rbr_pushrod_jcr_prod_uca),
        j4,
        multi_dot([j13.T,A(j14).T,B(j1,j15)]),
        j4,
        multi_dot([j15.T,j11,B(j14,j13)]),
        j0,
        B(j16,self.ubar_rbl_uca_jcl_uca_upright),
        j2,
        -1*B(j17,self.ubar_rbl_upright_jcl_uca_upright),
        j0,
        B(j16,self.ubar_rbl_uca_jcl_uca_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcl_uca_chassis),
        j4,
        multi_dot([j19,j8,B(j16,j20)]),
        j4,
        multi_dot([j20.T,j22,j23]),
        j4,
        multi_dot([j19,j8,B(j16,j21)]),
        j4,
        multi_dot([j21.T,j22,j23]),
        j0,
        B(j16,self.ubar_rbl_uca_jcl_prod_uca),
        j2,
        -1*B(j25,self.ubar_rbl_pushrod_jcl_prod_uca),
        j4,
        multi_dot([j24.T,A(j25).T,B(j16,j26)]),
        j4,
        multi_dot([j26.T,j22,B(j25,j24)]),
        j0,
        B(j27,self.ubar_rbr_lca_jcr_lca_upright),
        j2,
        -1*B(j3,self.ubar_rbr_upright_jcr_lca_upright),
        j0,
        B(j27,self.ubar_rbr_lca_jcr_lca_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcr_lca_chassis),
        j4,
        multi_dot([j29,j8,B(j27,j30)]),
        j4,
        multi_dot([j30.T,j32,j33]),
        j4,
        multi_dot([j29,j8,B(j27,j31)]),
        j4,
        multi_dot([j31.T,j32,j33]),
        j0,
        B(j34,self.ubar_rbl_lca_jcl_lca_upright),
        j2,
        -1*B(j17,self.ubar_rbl_upright_jcl_lca_upright),
        j0,
        B(j34,self.ubar_rbl_lca_jcl_lca_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcl_lca_chassis),
        j4,
        multi_dot([j36,j8,B(j34,j37)]),
        j4,
        multi_dot([j37.T,j39,j40]),
        j4,
        multi_dot([j36,j8,B(j34,j38)]),
        j4,
        multi_dot([j38.T,j39,j40]),
        j0,
        B(j3,self.ubar_rbr_upright_jcr_hub_bearing),
        j2,
        -1*B(j43,self.ubar_rbr_hub_jcr_hub_bearing),
        j4,
        multi_dot([j42,j44,B(j3,j45)]),
        j4,
        multi_dot([j45.T,j47,j48]),
        j4,
        multi_dot([j42,j44,B(j3,j46)]),
        j4,
        multi_dot([j46.T,j47,j48]),
        j0,
        B(j17,self.ubar_rbl_upright_jcl_hub_bearing),
        j2,
        -1*B(j51,self.ubar_rbl_hub_jcl_hub_bearing),
        j4,
        multi_dot([j50,j52,B(j17,j53)]),
        j4,
        multi_dot([j53.T,j55,j56]),
        j4,
        multi_dot([j50,j52,B(j17,j54)]),
        j4,
        multi_dot([j54.T,j55,j56]),
        j2,
        -1*B(j14,self.ubar_rbr_pushrod_jcr_prod_rocker),
        j0,
        B(j57,self.ubar_rbr_rocker_jcr_prod_rocker),
        j0,
        B(j57,self.ubar_rbr_rocker_jcr_rocker_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcr_rocker_chassis),
        j4,
        multi_dot([j59,j8,B(j57,j60)]),
        j4,
        multi_dot([j60.T,j62,j63]),
        j4,
        multi_dot([j59,j8,B(j57,j61)]),
        j4,
        multi_dot([j61.T,j62,j63]),
        j2,
        -1*B(j25,self.ubar_rbl_pushrod_jcl_prod_rocker),
        j0,
        B(j64,self.ubar_rbl_rocker_jcl_prod_rocker),
        j0,
        B(j64,self.ubar_rbl_rocker_jcl_rocker_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcl_rocker_chassis),
        j4,
        multi_dot([j66,j8,B(j64,j67)]),
        j4,
        multi_dot([j67.T,j69,j70]),
        j4,
        multi_dot([j66,j8,B(j64,j68)]),
        j4,
        multi_dot([j68.T,j69,j70]),
        j0,
        B(j71,self.ubar_rbr_upper_strut_jcr_strut_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcr_strut_chassis),
        j4,
        multi_dot([j72.T,j8,B(j71,j73)]),
        j4,
        multi_dot([j73.T,j74,B(j7,j72)]),
        j4,
        multi_dot([j76,j78,j80]),
        j4,
        multi_dot([j83,j74,j91]),
        j4,
        multi_dot([j76,j78,j82]),
        j4,
        multi_dot([j89,j74,j91]),
        j84,
        (multi_dot([j83,j74,j86]) + multi_dot([j88,j80])),
        -1*j84,
        -1*multi_dot([j83,j74,j92]),
        j90,
        (multi_dot([j89,j74,j86]) + multi_dot([j88,j82])),
        -1*j90,
        -1*multi_dot([j89,j74,j92]),
        j0,
        B(j93,self.ubar_rbl_upper_strut_jcl_strut_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcl_strut_chassis),
        j4,
        multi_dot([j94.T,j8,B(j93,j95)]),
        j4,
        multi_dot([j95.T,j96,B(j7,j94)]),
        j4,
        multi_dot([j98,j100,j102]),
        j4,
        multi_dot([j105,j96,j113]),
        j4,
        multi_dot([j98,j100,j104]),
        j4,
        multi_dot([j111,j96,j113]),
        j106,
        (multi_dot([j105,j96,j108]) + multi_dot([j110,j102])),
        -1*j106,
        -1*multi_dot([j105,j96,j114]),
        j112,
        (multi_dot([j111,j96,j108]) + multi_dot([j110,j104])),
        -1*j112,
        -1*multi_dot([j111,j96,j114]),
        j2,
        -1*B(j57,self.ubar_rbr_rocker_jcr_strut_rocker),
        j0,
        B(j77,self.ubar_rbr_lower_strut_jcr_strut_rocker),
        j4,
        multi_dot([j116.T,j78,B(j57,j115)]),
        j4,
        multi_dot([j115.T,j62,B(j77,j116)]),
        j2,
        -1*B(j64,self.ubar_rbl_rocker_jcl_strut_rocker),
        j0,
        B(j99,self.ubar_rbl_lower_strut_jcl_strut_rocker),
        j4,
        multi_dot([j118.T,j100,B(j64,j117)]),
        j4,
        multi_dot([j117.T,j69,B(j99,j118)]),
        j2,
        -1*B(j3,self.ubar_rbr_upright_jcr_tie_upright),
        j0,
        B(j119,self.ubar_rbr_tie_rod_jcr_tie_upright),
        j0,
        B(j119,self.ubar_rbr_tie_rod_jcr_tie_steering),
        j2,
        -1*B(j121,self.ubar_vbr_steer_jcr_tie_steering),
        j4,
        multi_dot([j120.T,A(j121).T,B(j119,j122)]),
        j4,
        multi_dot([j122.T,A(j119).T,B(j121,j120)]),
        j2,
        -1*B(j17,self.ubar_rbl_upright_jcl_tie_upright),
        j0,
        B(j123,self.ubar_rbl_tie_rod_jcl_tie_upright),
        j0,
        B(j123,self.ubar_rbl_tie_rod_jcl_tie_steering),
        j2,
        -1*B(j125,self.ubar_vbl_steer_jcl_tie_steering),
        j4,
        multi_dot([j124.T,A(j125).T,B(j123,j126)]),
        j4,
        multi_dot([j126.T,A(j123).T,B(j125,j124)]),
        j4,
        2*j1.T,
        j4,
        2*j16.T,
        j4,
        2*j27.T,
        j4,
        2*j34.T,
        j4,
        2*j3.T,
        j4,
        2*j17.T,
        j4,
        2*j14.T,
        j4,
        2*j25.T,
        j4,
        2*j57.T,
        j4,
        2*j64.T,
        j4,
        2*j71.T,
        j4,
        2*j93.T,
        j4,
        2*j77.T,
        j4,
        2*j99.T,
        j4,
        2*j119.T,
        j4,
        2*j123.T,
        j4,
        2*j43.T,
        j4,
        2*j51.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = np.eye(3, dtype=np.float64)
        m1 = G(self.P_rbr_uca)
        m2 = G(self.P_rbl_uca)
        m3 = G(self.P_rbr_lca)
        m4 = G(self.P_rbl_lca)
        m5 = G(self.P_rbr_upright)
        m6 = G(self.P_rbl_upright)
        m7 = G(self.P_rbr_pushrod)
        m8 = G(self.P_rbl_pushrod)
        m9 = G(self.P_rbr_rocker)
        m10 = G(self.P_rbl_rocker)
        m11 = G(self.P_rbr_upper_strut)
        m12 = G(self.P_rbl_upper_strut)
        m13 = G(self.P_rbr_lower_strut)
        m14 = G(self.P_rbl_lower_strut)
        m15 = G(self.P_rbr_tie_rod)
        m16 = G(self.P_rbl_tie_rod)
        m17 = G(self.P_rbr_hub)
        m18 = G(self.P_rbl_hub)

        self.mass_eq_blocks = (config.m_rbr_uca*m0,
        4*multi_dot([m1.T,config.Jbar_rbr_uca,m1]),
        config.m_rbl_uca*m0,
        4*multi_dot([m2.T,config.Jbar_rbl_uca,m2]),
        config.m_rbr_lca*m0,
        4*multi_dot([m3.T,config.Jbar_rbr_lca,m3]),
        config.m_rbl_lca*m0,
        4*multi_dot([m4.T,config.Jbar_rbl_lca,m4]),
        config.m_rbr_upright*m0,
        4*multi_dot([m5.T,config.Jbar_rbr_upright,m5]),
        config.m_rbl_upright*m0,
        4*multi_dot([m6.T,config.Jbar_rbl_upright,m6]),
        config.m_rbr_pushrod*m0,
        4*multi_dot([m7.T,config.Jbar_rbr_pushrod,m7]),
        config.m_rbl_pushrod*m0,
        4*multi_dot([m8.T,config.Jbar_rbl_pushrod,m8]),
        config.m_rbr_rocker*m0,
        4*multi_dot([m9.T,config.Jbar_rbr_rocker,m9]),
        config.m_rbl_rocker*m0,
        4*multi_dot([m10.T,config.Jbar_rbl_rocker,m10]),
        config.m_rbr_upper_strut*m0,
        4*multi_dot([m11.T,config.Jbar_rbr_upper_strut,m11]),
        config.m_rbl_upper_strut*m0,
        4*multi_dot([m12.T,config.Jbar_rbl_upper_strut,m12]),
        config.m_rbr_lower_strut*m0,
        4*multi_dot([m13.T,config.Jbar_rbr_lower_strut,m13]),
        config.m_rbl_lower_strut*m0,
        4*multi_dot([m14.T,config.Jbar_rbl_lower_strut,m14]),
        config.m_rbr_tie_rod*m0,
        4*multi_dot([m15.T,config.Jbar_rbr_tie_rod,m15]),
        config.m_rbl_tie_rod*m0,
        4*multi_dot([m16.T,config.Jbar_rbl_tie_rod,m16]),
        config.m_rbr_hub*m0,
        4*multi_dot([m17.T,config.Jbar_rbr_hub,m17]),
        config.m_rbl_hub*m0,
        4*multi_dot([m18.T,config.Jbar_rbl_hub,m18]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbr_uca)
        f1 = G(self.Pd_rbl_uca)
        f2 = G(self.Pd_rbr_lca)
        f3 = G(self.Pd_rbl_lca)
        f4 = G(self.Pd_rbr_upright)
        f5 = G(self.Pd_rbl_upright)
        f6 = G(self.Pd_rbr_pushrod)
        f7 = G(self.Pd_rbl_pushrod)
        f8 = G(self.Pd_rbr_rocker)
        f9 = G(self.Pd_rbl_rocker)
        f10 = self.R_rbr_upper_strut
        f11 = self.R_rbr_lower_strut
        f12 = self.ubar_rbr_upper_strut_far_strut
        f13 = self.P_rbr_upper_strut
        f14 = A(f13)
        f15 = self.ubar_rbr_lower_strut_far_strut
        f16 = self.P_rbr_lower_strut
        f17 = A(f16)
        f18 = (f10.T + -1*f11.T + multi_dot([f12.T,f14.T]) + -1*multi_dot([f15.T,f17.T]))
        f19 = multi_dot([f14,f12])
        f20 = multi_dot([f17,f15])
        f21 = (f10 + -1*f11 + f19 + -1*f20)
        f22 = (multi_dot([f18,f21]))**(1.0/2.0)
        f23 = config.UF_far_strut_Fs(config.far_strut_FL - 1*f22[0])
        f24 = f22**(-1)
        f25 = self.Pd_rbr_upper_strut
        f26 = self.Pd_rbr_lower_strut
        f27 = config.UF_far_strut_Fd(multi_dot([f24,f18,(self.Rd_rbr_upper_strut + -1*self.Rd_rbr_lower_strut + multi_dot([B(f13,f12),f25]) + -1*multi_dot([B(f16,f15),f26]))]))
        f28 = f23 - 1*f27
        f29 = multi_dot([f21,f24])
        f30 = G(f25)
        f31 = self.R_rbl_upper_strut
        f32 = self.R_rbl_lower_strut
        f33 = self.ubar_rbl_upper_strut_fal_strut
        f34 = self.P_rbl_upper_strut
        f35 = A(f34)
        f36 = self.ubar_rbl_lower_strut_fal_strut
        f37 = self.P_rbl_lower_strut
        f38 = A(f37)
        f39 = (f31.T + -1*f32.T + multi_dot([f33.T,f35.T]) + -1*multi_dot([f36.T,f38.T]))
        f40 = multi_dot([f35,f33])
        f41 = multi_dot([f38,f36])
        f42 = (f31 + -1*f32 + f40 + -1*f41)
        f43 = (multi_dot([f39,f42]))**(1.0/2.0)
        f44 = config.UF_fal_strut_Fs(config.fal_strut_FL - 1*f43[0])
        f45 = f43**(-1)
        f46 = self.Pd_rbl_upper_strut
        f47 = self.Pd_rbl_lower_strut
        f48 = config.UF_fal_strut_Fd(multi_dot([f45,f39,(self.Rd_rbl_upper_strut + -1*self.Rd_rbl_lower_strut + multi_dot([B(f34,f33),f46]) + -1*multi_dot([B(f37,f36),f47]))]))
        f49 = f44 - 1*f48
        f50 = multi_dot([f42,f45])
        f51 = G(f46)
        f52 = np.zeros((3,1),dtype=np.float64)
        f53 = -1*f23 + f27
        f54 = np.zeros((4,1),dtype=np.float64)
        f55 = G(f26)
        f56 = -1*f44 + f48
        f57 = G(f47)
        f58 = G(self.Pd_rbr_tie_rod)
        f59 = G(self.Pd_rbl_tie_rod)
        f60 = t
        f61 = config.UF_far_drive_F(f60)
        f62 = config.UF_far_tire_F(f60)
        f63 = G(self.Pd_rbr_hub)
        f64 = self.P_rbr_hub
        f65 = G(f64).T
        f66 = A(f64)
        f67 = config.UF_fal_drive_F(f60)
        f68 = config.UF_fal_tire_F(f60)
        f69 = G(self.Pd_rbl_hub)
        f70 = self.P_rbl_hub
        f71 = G(f70).T
        f72 = A(f70)

        self.frc_eq_blocks = (self.F_rbr_uca_gravity,
        8*multi_dot([f0.T,config.Jbar_rbr_uca,f0,self.P_rbr_uca]),
        self.F_rbl_uca_gravity,
        8*multi_dot([f1.T,config.Jbar_rbl_uca,f1,self.P_rbl_uca]),
        self.F_rbr_lca_gravity,
        8*multi_dot([f2.T,config.Jbar_rbr_lca,f2,self.P_rbr_lca]),
        self.F_rbl_lca_gravity,
        8*multi_dot([f3.T,config.Jbar_rbl_lca,f3,self.P_rbl_lca]),
        self.F_rbr_upright_gravity,
        8*multi_dot([f4.T,config.Jbar_rbr_upright,f4,self.P_rbr_upright]),
        self.F_rbl_upright_gravity,
        8*multi_dot([f5.T,config.Jbar_rbl_upright,f5,self.P_rbl_upright]),
        self.F_rbr_pushrod_gravity,
        8*multi_dot([f6.T,config.Jbar_rbr_pushrod,f6,self.P_rbr_pushrod]),
        self.F_rbl_pushrod_gravity,
        8*multi_dot([f7.T,config.Jbar_rbl_pushrod,f7,self.P_rbl_pushrod]),
        self.F_rbr_rocker_gravity,
        8*multi_dot([f8.T,config.Jbar_rbr_rocker,f8,self.P_rbr_rocker]),
        self.F_rbl_rocker_gravity,
        8*multi_dot([f9.T,config.Jbar_rbl_rocker,f9,self.P_rbl_rocker]),
        (self.F_rbr_upper_strut_gravity + f28*f29),
        (8*multi_dot([f30.T,config.Jbar_rbr_upper_strut,f30,f13]) + 2*multi_dot([G(f13).T,(self.T_rbr_upper_strut_far_strut + f28*multi_dot([skew(f19).T,f21,f24]))])),
        (self.F_rbl_upper_strut_gravity + f49*f50),
        (8*multi_dot([f51.T,config.Jbar_rbl_upper_strut,f51,f34]) + 2*multi_dot([G(f34).T,(self.T_rbl_upper_strut_fal_strut + f49*multi_dot([skew(f40).T,f42,f45]))])),
        (self.F_rbr_lower_strut_gravity + f52 + f53*f29),
        (f54 + 8*multi_dot([f55.T,config.Jbar_rbr_lower_strut,f55,f16]) + 2*multi_dot([G(f16).T,(self.T_rbr_lower_strut_far_strut + f53*multi_dot([skew(f20).T,f21,f24]))])),
        (self.F_rbl_lower_strut_gravity + f52 + f56*f50),
        (f54 + 8*multi_dot([f57.T,config.Jbar_rbl_lower_strut,f57,f37]) + 2*multi_dot([G(f37).T,(self.T_rbl_lower_strut_fal_strut + f56*multi_dot([skew(f41).T,f42,f45]))])),
        self.F_rbr_tie_rod_gravity,
        8*multi_dot([f58.T,config.Jbar_rbr_tie_rod,f58,self.P_rbr_tie_rod]),
        self.F_rbl_tie_rod_gravity,
        8*multi_dot([f59.T,config.Jbar_rbl_tie_rod,f59,self.P_rbl_tie_rod]),
        (self.F_rbr_hub_gravity + f61 + f62),
        (8*multi_dot([f63.T,config.Jbar_rbr_hub,f63,f64]) + 2*multi_dot([f65,(config.UF_far_drive_T(f60) + multi_dot([skew(multi_dot([f66,self.ubar_rbr_hub_far_drive])).T,f61]))]) + 2*multi_dot([f65,(config.UF_far_tire_T(f60) + multi_dot([skew(multi_dot([f66,self.ubar_rbr_hub_far_tire])).T,f62]))])),
        (self.F_rbl_hub_gravity + f67 + f68),
        (8*multi_dot([f69.T,config.Jbar_rbl_hub,f69,f70]) + 2*multi_dot([f71,(config.UF_fal_drive_T(f60) + multi_dot([skew(multi_dot([f72,self.ubar_rbl_hub_fal_drive])).T,f67]))]) + 2*multi_dot([f71,(config.UF_fal_tire_T(f60) + multi_dot([skew(multi_dot([f72,self.ubar_rbl_hub_fal_tire])).T,f68]))])),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbr_uca_jcr_uca_upright = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbr_uca,self.ubar_rbr_uca_jcr_uca_upright).T]]),self.L_jcr_uca_upright])
        self.F_rbr_uca_jcr_uca_upright = Q_rbr_uca_jcr_uca_upright[0:3,0:1]
        Te_rbr_uca_jcr_uca_upright = Q_rbr_uca_jcr_uca_upright[3:7,0:1]
        self.T_rbr_uca_jcr_uca_upright = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_uca),self.ubar_rbr_uca_jcr_uca_upright])),self.F_rbr_uca_jcr_uca_upright]) + 0.5*multi_dot([E(self.P_rbr_uca),Te_rbr_uca_jcr_uca_upright]))
        Q_rbr_uca_jcr_uca_chassis = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_uca,self.ubar_rbr_uca_jcr_uca_chassis).T,multi_dot([B(self.P_rbr_uca,self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]]),multi_dot([B(self.P_rbr_uca,self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]])]]),self.L_jcr_uca_chassis])
        self.F_rbr_uca_jcr_uca_chassis = Q_rbr_uca_jcr_uca_chassis[0:3,0:1]
        Te_rbr_uca_jcr_uca_chassis = Q_rbr_uca_jcr_uca_chassis[3:7,0:1]
        self.T_rbr_uca_jcr_uca_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_uca),self.ubar_rbr_uca_jcr_uca_chassis])),self.F_rbr_uca_jcr_uca_chassis]) + 0.5*multi_dot([E(self.P_rbr_uca),Te_rbr_uca_jcr_uca_chassis]))
        Q_rbr_uca_jcr_prod_uca = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_uca,self.ubar_rbr_uca_jcr_prod_uca).T,multi_dot([B(self.P_rbr_uca,self.Mbar_rbr_uca_jcr_prod_uca[:,0:1]).T,A(self.P_rbr_pushrod),self.Mbar_rbr_pushrod_jcr_prod_uca[:,0:1]])]]),self.L_jcr_prod_uca])
        self.F_rbr_uca_jcr_prod_uca = Q_rbr_uca_jcr_prod_uca[0:3,0:1]
        Te_rbr_uca_jcr_prod_uca = Q_rbr_uca_jcr_prod_uca[3:7,0:1]
        self.T_rbr_uca_jcr_prod_uca = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_uca),self.ubar_rbr_uca_jcr_prod_uca])),self.F_rbr_uca_jcr_prod_uca]) + 0.5*multi_dot([E(self.P_rbr_uca),Te_rbr_uca_jcr_prod_uca]))
        Q_rbl_uca_jcl_uca_upright = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbl_uca,self.ubar_rbl_uca_jcl_uca_upright).T]]),self.L_jcl_uca_upright])
        self.F_rbl_uca_jcl_uca_upright = Q_rbl_uca_jcl_uca_upright[0:3,0:1]
        Te_rbl_uca_jcl_uca_upright = Q_rbl_uca_jcl_uca_upright[3:7,0:1]
        self.T_rbl_uca_jcl_uca_upright = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_uca),self.ubar_rbl_uca_jcl_uca_upright])),self.F_rbl_uca_jcl_uca_upright]) + 0.5*multi_dot([E(self.P_rbl_uca),Te_rbl_uca_jcl_uca_upright]))
        Q_rbl_uca_jcl_uca_chassis = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_uca,self.ubar_rbl_uca_jcl_uca_chassis).T,multi_dot([B(self.P_rbl_uca,self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]]),multi_dot([B(self.P_rbl_uca,self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]])]]),self.L_jcl_uca_chassis])
        self.F_rbl_uca_jcl_uca_chassis = Q_rbl_uca_jcl_uca_chassis[0:3,0:1]
        Te_rbl_uca_jcl_uca_chassis = Q_rbl_uca_jcl_uca_chassis[3:7,0:1]
        self.T_rbl_uca_jcl_uca_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_uca),self.ubar_rbl_uca_jcl_uca_chassis])),self.F_rbl_uca_jcl_uca_chassis]) + 0.5*multi_dot([E(self.P_rbl_uca),Te_rbl_uca_jcl_uca_chassis]))
        Q_rbl_uca_jcl_prod_uca = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_uca,self.ubar_rbl_uca_jcl_prod_uca).T,multi_dot([B(self.P_rbl_uca,self.Mbar_rbl_uca_jcl_prod_uca[:,0:1]).T,A(self.P_rbl_pushrod),self.Mbar_rbl_pushrod_jcl_prod_uca[:,0:1]])]]),self.L_jcl_prod_uca])
        self.F_rbl_uca_jcl_prod_uca = Q_rbl_uca_jcl_prod_uca[0:3,0:1]
        Te_rbl_uca_jcl_prod_uca = Q_rbl_uca_jcl_prod_uca[3:7,0:1]
        self.T_rbl_uca_jcl_prod_uca = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_uca),self.ubar_rbl_uca_jcl_prod_uca])),self.F_rbl_uca_jcl_prod_uca]) + 0.5*multi_dot([E(self.P_rbl_uca),Te_rbl_uca_jcl_prod_uca]))
        Q_rbr_lca_jcr_lca_upright = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbr_lca,self.ubar_rbr_lca_jcr_lca_upright).T]]),self.L_jcr_lca_upright])
        self.F_rbr_lca_jcr_lca_upright = Q_rbr_lca_jcr_lca_upright[0:3,0:1]
        Te_rbr_lca_jcr_lca_upright = Q_rbr_lca_jcr_lca_upright[3:7,0:1]
        self.T_rbr_lca_jcr_lca_upright = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_lca),self.ubar_rbr_lca_jcr_lca_upright])),self.F_rbr_lca_jcr_lca_upright]) + 0.5*multi_dot([E(self.P_rbr_lca),Te_rbr_lca_jcr_lca_upright]))
        Q_rbr_lca_jcr_lca_chassis = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_lca,self.ubar_rbr_lca_jcr_lca_chassis).T,multi_dot([B(self.P_rbr_lca,self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]]),multi_dot([B(self.P_rbr_lca,self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]])]]),self.L_jcr_lca_chassis])
        self.F_rbr_lca_jcr_lca_chassis = Q_rbr_lca_jcr_lca_chassis[0:3,0:1]
        Te_rbr_lca_jcr_lca_chassis = Q_rbr_lca_jcr_lca_chassis[3:7,0:1]
        self.T_rbr_lca_jcr_lca_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_lca),self.ubar_rbr_lca_jcr_lca_chassis])),self.F_rbr_lca_jcr_lca_chassis]) + 0.5*multi_dot([E(self.P_rbr_lca),Te_rbr_lca_jcr_lca_chassis]))
        Q_rbl_lca_jcl_lca_upright = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbl_lca,self.ubar_rbl_lca_jcl_lca_upright).T]]),self.L_jcl_lca_upright])
        self.F_rbl_lca_jcl_lca_upright = Q_rbl_lca_jcl_lca_upright[0:3,0:1]
        Te_rbl_lca_jcl_lca_upright = Q_rbl_lca_jcl_lca_upright[3:7,0:1]
        self.T_rbl_lca_jcl_lca_upright = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_lca),self.ubar_rbl_lca_jcl_lca_upright])),self.F_rbl_lca_jcl_lca_upright]) + 0.5*multi_dot([E(self.P_rbl_lca),Te_rbl_lca_jcl_lca_upright]))
        Q_rbl_lca_jcl_lca_chassis = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_lca,self.ubar_rbl_lca_jcl_lca_chassis).T,multi_dot([B(self.P_rbl_lca,self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]]),multi_dot([B(self.P_rbl_lca,self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]])]]),self.L_jcl_lca_chassis])
        self.F_rbl_lca_jcl_lca_chassis = Q_rbl_lca_jcl_lca_chassis[0:3,0:1]
        Te_rbl_lca_jcl_lca_chassis = Q_rbl_lca_jcl_lca_chassis[3:7,0:1]
        self.T_rbl_lca_jcl_lca_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_lca),self.ubar_rbl_lca_jcl_lca_chassis])),self.F_rbl_lca_jcl_lca_chassis]) + 0.5*multi_dot([E(self.P_rbl_lca),Te_rbl_lca_jcl_lca_chassis]))
        Q_rbr_upright_jcr_hub_bearing = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_upright,self.ubar_rbr_upright_jcr_hub_bearing).T,multi_dot([B(self.P_rbr_upright,self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]).T,A(self.P_rbr_hub),self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]]),multi_dot([B(self.P_rbr_upright,self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]).T,A(self.P_rbr_hub),self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]])]]),self.L_jcr_hub_bearing])
        self.F_rbr_upright_jcr_hub_bearing = Q_rbr_upright_jcr_hub_bearing[0:3,0:1]
        Te_rbr_upright_jcr_hub_bearing = Q_rbr_upright_jcr_hub_bearing[3:7,0:1]
        self.T_rbr_upright_jcr_hub_bearing = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_upright),self.ubar_rbr_upright_jcr_hub_bearing])),self.F_rbr_upright_jcr_hub_bearing]) + 0.5*multi_dot([E(self.P_rbr_upright),Te_rbr_upright_jcr_hub_bearing]))
        Q_rbl_upright_jcl_hub_bearing = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_upright,self.ubar_rbl_upright_jcl_hub_bearing).T,multi_dot([B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]).T,A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]]),multi_dot([B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]).T,A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]])]]),self.L_jcl_hub_bearing])
        self.F_rbl_upright_jcl_hub_bearing = Q_rbl_upright_jcl_hub_bearing[0:3,0:1]
        Te_rbl_upright_jcl_hub_bearing = Q_rbl_upright_jcl_hub_bearing[3:7,0:1]
        self.T_rbl_upright_jcl_hub_bearing = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_upright),self.ubar_rbl_upright_jcl_hub_bearing])),self.F_rbl_upright_jcl_hub_bearing]) + 0.5*multi_dot([E(self.P_rbl_upright),Te_rbl_upright_jcl_hub_bearing]))
        Q_rbr_rocker_jcr_prod_rocker = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbr_rocker,self.ubar_rbr_rocker_jcr_prod_rocker).T]]),self.L_jcr_prod_rocker])
        self.F_rbr_rocker_jcr_prod_rocker = Q_rbr_rocker_jcr_prod_rocker[0:3,0:1]
        Te_rbr_rocker_jcr_prod_rocker = Q_rbr_rocker_jcr_prod_rocker[3:7,0:1]
        self.T_rbr_rocker_jcr_prod_rocker = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_rocker),self.ubar_rbr_rocker_jcr_prod_rocker])),self.F_rbr_rocker_jcr_prod_rocker]) + 0.5*multi_dot([E(self.P_rbr_rocker),Te_rbr_rocker_jcr_prod_rocker]))
        Q_rbr_rocker_jcr_rocker_chassis = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_rocker,self.ubar_rbr_rocker_jcr_rocker_chassis).T,multi_dot([B(self.P_rbr_rocker,self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]]),multi_dot([B(self.P_rbr_rocker,self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]])]]),self.L_jcr_rocker_chassis])
        self.F_rbr_rocker_jcr_rocker_chassis = Q_rbr_rocker_jcr_rocker_chassis[0:3,0:1]
        Te_rbr_rocker_jcr_rocker_chassis = Q_rbr_rocker_jcr_rocker_chassis[3:7,0:1]
        self.T_rbr_rocker_jcr_rocker_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_rocker),self.ubar_rbr_rocker_jcr_rocker_chassis])),self.F_rbr_rocker_jcr_rocker_chassis]) + 0.5*multi_dot([E(self.P_rbr_rocker),Te_rbr_rocker_jcr_rocker_chassis]))
        Q_rbl_rocker_jcl_prod_rocker = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbl_rocker,self.ubar_rbl_rocker_jcl_prod_rocker).T]]),self.L_jcl_prod_rocker])
        self.F_rbl_rocker_jcl_prod_rocker = Q_rbl_rocker_jcl_prod_rocker[0:3,0:1]
        Te_rbl_rocker_jcl_prod_rocker = Q_rbl_rocker_jcl_prod_rocker[3:7,0:1]
        self.T_rbl_rocker_jcl_prod_rocker = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_rocker),self.ubar_rbl_rocker_jcl_prod_rocker])),self.F_rbl_rocker_jcl_prod_rocker]) + 0.5*multi_dot([E(self.P_rbl_rocker),Te_rbl_rocker_jcl_prod_rocker]))
        Q_rbl_rocker_jcl_rocker_chassis = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_rocker,self.ubar_rbl_rocker_jcl_rocker_chassis).T,multi_dot([B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]]),multi_dot([B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]])]]),self.L_jcl_rocker_chassis])
        self.F_rbl_rocker_jcl_rocker_chassis = Q_rbl_rocker_jcl_rocker_chassis[0:3,0:1]
        Te_rbl_rocker_jcl_rocker_chassis = Q_rbl_rocker_jcl_rocker_chassis[3:7,0:1]
        self.T_rbl_rocker_jcl_rocker_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_rocker),self.ubar_rbl_rocker_jcl_rocker_chassis])),self.F_rbl_rocker_jcl_rocker_chassis]) + 0.5*multi_dot([E(self.P_rbl_rocker),Te_rbl_rocker_jcl_rocker_chassis]))
        Q_rbr_upper_strut_jcr_strut_chassis = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut_chassis).T,multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]])]]),self.L_jcr_strut_chassis])
        self.F_rbr_upper_strut_jcr_strut_chassis = Q_rbr_upper_strut_jcr_strut_chassis[0:3,0:1]
        Te_rbr_upper_strut_jcr_strut_chassis = Q_rbr_upper_strut_jcr_strut_chassis[3:7,0:1]
        self.T_rbr_upper_strut_jcr_strut_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut_chassis])),self.F_rbr_upper_strut_jcr_strut_chassis]) + 0.5*multi_dot([E(self.P_rbr_upper_strut),Te_rbr_upper_strut_jcr_strut_chassis]))
        Q_rbr_upper_strut_jcr_strut = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]]),multi_dot([A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]])],[multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]).T,A(self.P_rbr_lower_strut),self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]]),multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]).T,A(self.P_rbr_lower_strut),self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]]),(multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]).T,(-1*self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut]) + -1*multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut]) + self.R_rbr_upper_strut)]) + multi_dot([B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut).T,A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]])),(multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]).T,(-1*self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut]) + -1*multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut]) + self.R_rbr_upper_strut)]) + multi_dot([B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut).T,A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]]))]]),self.L_jcr_strut])
        self.F_rbr_upper_strut_jcr_strut = Q_rbr_upper_strut_jcr_strut[0:3,0:1]
        Te_rbr_upper_strut_jcr_strut = Q_rbr_upper_strut_jcr_strut[3:7,0:1]
        self.T_rbr_upper_strut_jcr_strut = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut])),self.F_rbr_upper_strut_jcr_strut]) + 0.5*multi_dot([E(self.P_rbr_upper_strut),Te_rbr_upper_strut_jcr_strut]))
        Q_rbl_upper_strut_jcl_strut_chassis = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut_chassis).T,multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]])]]),self.L_jcl_strut_chassis])
        self.F_rbl_upper_strut_jcl_strut_chassis = Q_rbl_upper_strut_jcl_strut_chassis[0:3,0:1]
        Te_rbl_upper_strut_jcl_strut_chassis = Q_rbl_upper_strut_jcl_strut_chassis[3:7,0:1]
        self.T_rbl_upper_strut_jcl_strut_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut_chassis])),self.F_rbl_upper_strut_jcl_strut_chassis]) + 0.5*multi_dot([E(self.P_rbl_upper_strut),Te_rbl_upper_strut_jcl_strut_chassis]))
        Q_rbl_upper_strut_jcl_strut = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]]),multi_dot([A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]])],[multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]).T,A(self.P_rbl_lower_strut),self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]]),multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]).T,A(self.P_rbl_lower_strut),self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]]),(multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]).T,(-1*self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut]) + -1*multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut]) + self.R_rbl_upper_strut)]) + multi_dot([B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut).T,A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]])),(multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]).T,(-1*self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut]) + -1*multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut]) + self.R_rbl_upper_strut)]) + multi_dot([B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut).T,A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]]))]]),self.L_jcl_strut])
        self.F_rbl_upper_strut_jcl_strut = Q_rbl_upper_strut_jcl_strut[0:3,0:1]
        Te_rbl_upper_strut_jcl_strut = Q_rbl_upper_strut_jcl_strut[3:7,0:1]
        self.T_rbl_upper_strut_jcl_strut = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut])),self.F_rbl_upper_strut_jcl_strut]) + 0.5*multi_dot([E(self.P_rbl_upper_strut),Te_rbl_upper_strut_jcl_strut]))
        Q_rbr_lower_strut_jcr_strut_rocker = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_lower_strut,self.ubar_rbr_lower_strut_jcr_strut_rocker).T,multi_dot([B(self.P_rbr_lower_strut,self.Mbar_rbr_lower_strut_jcr_strut_rocker[:,0:1]).T,A(self.P_rbr_rocker),self.Mbar_rbr_rocker_jcr_strut_rocker[:,0:1]])]]),self.L_jcr_strut_rocker])
        self.F_rbr_lower_strut_jcr_strut_rocker = Q_rbr_lower_strut_jcr_strut_rocker[0:3,0:1]
        Te_rbr_lower_strut_jcr_strut_rocker = Q_rbr_lower_strut_jcr_strut_rocker[3:7,0:1]
        self.T_rbr_lower_strut_jcr_strut_rocker = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut_rocker])),self.F_rbr_lower_strut_jcr_strut_rocker]) + 0.5*multi_dot([E(self.P_rbr_lower_strut),Te_rbr_lower_strut_jcr_strut_rocker]))
        Q_rbl_lower_strut_jcl_strut_rocker = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_lower_strut,self.ubar_rbl_lower_strut_jcl_strut_rocker).T,multi_dot([B(self.P_rbl_lower_strut,self.Mbar_rbl_lower_strut_jcl_strut_rocker[:,0:1]).T,A(self.P_rbl_rocker),self.Mbar_rbl_rocker_jcl_strut_rocker[:,0:1]])]]),self.L_jcl_strut_rocker])
        self.F_rbl_lower_strut_jcl_strut_rocker = Q_rbl_lower_strut_jcl_strut_rocker[0:3,0:1]
        Te_rbl_lower_strut_jcl_strut_rocker = Q_rbl_lower_strut_jcl_strut_rocker[3:7,0:1]
        self.T_rbl_lower_strut_jcl_strut_rocker = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut_rocker])),self.F_rbl_lower_strut_jcl_strut_rocker]) + 0.5*multi_dot([E(self.P_rbl_lower_strut),Te_rbl_lower_strut_jcl_strut_rocker]))
        Q_rbr_tie_rod_jcr_tie_upright = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbr_tie_rod,self.ubar_rbr_tie_rod_jcr_tie_upright).T]]),self.L_jcr_tie_upright])
        self.F_rbr_tie_rod_jcr_tie_upright = Q_rbr_tie_rod_jcr_tie_upright[0:3,0:1]
        Te_rbr_tie_rod_jcr_tie_upright = Q_rbr_tie_rod_jcr_tie_upright[3:7,0:1]
        self.T_rbr_tie_rod_jcr_tie_upright = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_tie_rod),self.ubar_rbr_tie_rod_jcr_tie_upright])),self.F_rbr_tie_rod_jcr_tie_upright]) + 0.5*multi_dot([E(self.P_rbr_tie_rod),Te_rbr_tie_rod_jcr_tie_upright]))
        Q_rbr_tie_rod_jcr_tie_steering = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_tie_rod,self.ubar_rbr_tie_rod_jcr_tie_steering).T,multi_dot([B(self.P_rbr_tie_rod,self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]).T,A(self.P_vbr_steer),self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]])]]),self.L_jcr_tie_steering])
        self.F_rbr_tie_rod_jcr_tie_steering = Q_rbr_tie_rod_jcr_tie_steering[0:3,0:1]
        Te_rbr_tie_rod_jcr_tie_steering = Q_rbr_tie_rod_jcr_tie_steering[3:7,0:1]
        self.T_rbr_tie_rod_jcr_tie_steering = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_tie_rod),self.ubar_rbr_tie_rod_jcr_tie_steering])),self.F_rbr_tie_rod_jcr_tie_steering]) + 0.5*multi_dot([E(self.P_rbr_tie_rod),Te_rbr_tie_rod_jcr_tie_steering]))
        Q_rbl_tie_rod_jcl_tie_upright = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbl_tie_rod,self.ubar_rbl_tie_rod_jcl_tie_upright).T]]),self.L_jcl_tie_upright])
        self.F_rbl_tie_rod_jcl_tie_upright = Q_rbl_tie_rod_jcl_tie_upright[0:3,0:1]
        Te_rbl_tie_rod_jcl_tie_upright = Q_rbl_tie_rod_jcl_tie_upright[3:7,0:1]
        self.T_rbl_tie_rod_jcl_tie_upright = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_tie_rod),self.ubar_rbl_tie_rod_jcl_tie_upright])),self.F_rbl_tie_rod_jcl_tie_upright]) + 0.5*multi_dot([E(self.P_rbl_tie_rod),Te_rbl_tie_rod_jcl_tie_upright]))
        Q_rbl_tie_rod_jcl_tie_steering = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_tie_rod,self.ubar_rbl_tie_rod_jcl_tie_steering).T,multi_dot([B(self.P_rbl_tie_rod,self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]).T,A(self.P_vbl_steer),self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]])]]),self.L_jcl_tie_steering])
        self.F_rbl_tie_rod_jcl_tie_steering = Q_rbl_tie_rod_jcl_tie_steering[0:3,0:1]
        Te_rbl_tie_rod_jcl_tie_steering = Q_rbl_tie_rod_jcl_tie_steering[3:7,0:1]
        self.T_rbl_tie_rod_jcl_tie_steering = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_tie_rod),self.ubar_rbl_tie_rod_jcl_tie_steering])),self.F_rbl_tie_rod_jcl_tie_steering]) + 0.5*multi_dot([E(self.P_rbl_tie_rod),Te_rbl_tie_rod_jcl_tie_steering]))

        self.reactions = {'F_rbr_uca_jcr_uca_upright' : self.F_rbr_uca_jcr_uca_upright,
                        'T_rbr_uca_jcr_uca_upright' : self.T_rbr_uca_jcr_uca_upright,
                        'F_rbr_uca_jcr_uca_chassis' : self.F_rbr_uca_jcr_uca_chassis,
                        'T_rbr_uca_jcr_uca_chassis' : self.T_rbr_uca_jcr_uca_chassis,
                        'F_rbr_uca_jcr_prod_uca' : self.F_rbr_uca_jcr_prod_uca,
                        'T_rbr_uca_jcr_prod_uca' : self.T_rbr_uca_jcr_prod_uca,
                        'F_rbl_uca_jcl_uca_upright' : self.F_rbl_uca_jcl_uca_upright,
                        'T_rbl_uca_jcl_uca_upright' : self.T_rbl_uca_jcl_uca_upright,
                        'F_rbl_uca_jcl_uca_chassis' : self.F_rbl_uca_jcl_uca_chassis,
                        'T_rbl_uca_jcl_uca_chassis' : self.T_rbl_uca_jcl_uca_chassis,
                        'F_rbl_uca_jcl_prod_uca' : self.F_rbl_uca_jcl_prod_uca,
                        'T_rbl_uca_jcl_prod_uca' : self.T_rbl_uca_jcl_prod_uca,
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
                        'F_rbr_rocker_jcr_prod_rocker' : self.F_rbr_rocker_jcr_prod_rocker,
                        'T_rbr_rocker_jcr_prod_rocker' : self.T_rbr_rocker_jcr_prod_rocker,
                        'F_rbr_rocker_jcr_rocker_chassis' : self.F_rbr_rocker_jcr_rocker_chassis,
                        'T_rbr_rocker_jcr_rocker_chassis' : self.T_rbr_rocker_jcr_rocker_chassis,
                        'F_rbl_rocker_jcl_prod_rocker' : self.F_rbl_rocker_jcl_prod_rocker,
                        'T_rbl_rocker_jcl_prod_rocker' : self.T_rbl_rocker_jcl_prod_rocker,
                        'F_rbl_rocker_jcl_rocker_chassis' : self.F_rbl_rocker_jcl_rocker_chassis,
                        'T_rbl_rocker_jcl_rocker_chassis' : self.T_rbl_rocker_jcl_rocker_chassis,
                        'F_rbr_upper_strut_jcr_strut_chassis' : self.F_rbr_upper_strut_jcr_strut_chassis,
                        'T_rbr_upper_strut_jcr_strut_chassis' : self.T_rbr_upper_strut_jcr_strut_chassis,
                        'F_rbr_upper_strut_jcr_strut' : self.F_rbr_upper_strut_jcr_strut,
                        'T_rbr_upper_strut_jcr_strut' : self.T_rbr_upper_strut_jcr_strut,
                        'F_rbl_upper_strut_jcl_strut_chassis' : self.F_rbl_upper_strut_jcl_strut_chassis,
                        'T_rbl_upper_strut_jcl_strut_chassis' : self.T_rbl_upper_strut_jcl_strut_chassis,
                        'F_rbl_upper_strut_jcl_strut' : self.F_rbl_upper_strut_jcl_strut,
                        'T_rbl_upper_strut_jcl_strut' : self.T_rbl_upper_strut_jcl_strut,
                        'F_rbr_lower_strut_jcr_strut_rocker' : self.F_rbr_lower_strut_jcr_strut_rocker,
                        'T_rbr_lower_strut_jcr_strut_rocker' : self.T_rbr_lower_strut_jcr_strut_rocker,
                        'F_rbl_lower_strut_jcl_strut_rocker' : self.F_rbl_lower_strut_jcl_strut_rocker,
                        'T_rbl_lower_strut_jcl_strut_rocker' : self.T_rbl_lower_strut_jcl_strut_rocker,
                        'F_rbr_tie_rod_jcr_tie_upright' : self.F_rbr_tie_rod_jcr_tie_upright,
                        'T_rbr_tie_rod_jcr_tie_upright' : self.T_rbr_tie_rod_jcr_tie_upright,
                        'F_rbr_tie_rod_jcr_tie_steering' : self.F_rbr_tie_rod_jcr_tie_steering,
                        'T_rbr_tie_rod_jcr_tie_steering' : self.T_rbr_tie_rod_jcr_tie_steering,
                        'F_rbl_tie_rod_jcl_tie_upright' : self.F_rbl_tie_rod_jcl_tie_upright,
                        'T_rbl_tie_rod_jcl_tie_upright' : self.T_rbl_tie_rod_jcl_tie_upright,
                        'F_rbl_tie_rod_jcl_tie_steering' : self.F_rbl_tie_rod_jcl_tie_steering,
                        'T_rbl_tie_rod_jcl_tie_steering' : self.T_rbl_tie_rod_jcl_tie_steering}

