
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

        self.indicies_map = {'vbs_ground': 0, 'rbr_uca': 1, 'rbl_uca': 2, 'rbr_lca': 3, 'rbl_lca': 4, 'rbr_upright': 5, 'rbl_upright': 6, 'rbr_tie_rod': 7, 'rbl_tie_rod': 8, 'rbr_hub': 9, 'rbl_hub': 10, 'rbr_pushrod': 11, 'rbl_pushrod': 12, 'rbr_rocker': 13, 'rbl_rocker': 14, 'rbr_upper_strut': 15, 'rbl_upper_strut': 16, 'rbr_lower_strut': 17, 'rbl_lower_strut': 18, 'vbr_steer': 19, 'vbl_steer': 20, 'vbs_chassis': 21}

        self.n  = 126
        self.nc = 122
        self.nrows = 74
        self.ncols = 2*18
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_rbr_uca_jcr_uca_upright', 'T_rbr_uca_jcr_uca_upright', 'F_rbr_uca_jcr_uca_chassis', 'T_rbr_uca_jcr_uca_chassis', 'F_rbr_uca_jcr_prod_uca', 'T_rbr_uca_jcr_prod_uca', 'F_rbl_uca_jcl_uca_upright', 'T_rbl_uca_jcl_uca_upright', 'F_rbl_uca_jcl_uca_chassis', 'T_rbl_uca_jcl_uca_chassis', 'F_rbl_uca_jcl_prod_uca', 'T_rbl_uca_jcl_prod_uca', 'F_rbr_lca_jcr_lca_upright', 'T_rbr_lca_jcr_lca_upright', 'F_rbr_lca_jcr_lca_chassis', 'T_rbr_lca_jcr_lca_chassis', 'F_rbl_lca_jcl_lca_upright', 'T_rbl_lca_jcl_lca_upright', 'F_rbl_lca_jcl_lca_chassis', 'T_rbl_lca_jcl_lca_chassis', 'F_rbr_upright_jcr_hub_bearing', 'T_rbr_upright_jcr_hub_bearing', 'F_rbl_upright_jcl_hub_bearing', 'T_rbl_upright_jcl_hub_bearing', 'F_rbr_tie_rod_jcr_tie_upright', 'T_rbr_tie_rod_jcr_tie_upright', 'F_rbr_tie_rod_jcr_tie_steering', 'T_rbr_tie_rod_jcr_tie_steering', 'F_rbl_tie_rod_jcl_tie_upright', 'T_rbl_tie_rod_jcl_tie_upright', 'F_rbl_tie_rod_jcl_tie_steering', 'T_rbl_tie_rod_jcl_tie_steering', 'F_rbr_rocker_jcr_prod_rocker', 'T_rbr_rocker_jcr_prod_rocker', 'F_rbr_rocker_jcr_rocker_chassis', 'T_rbr_rocker_jcr_rocker_chassis', 'F_rbl_rocker_jcl_prod_rocker', 'T_rbl_rocker_jcl_prod_rocker', 'F_rbl_rocker_jcl_rocker_chassis', 'T_rbl_rocker_jcl_rocker_chassis', 'F_rbr_upper_strut_jcr_strut_chassis', 'T_rbr_upper_strut_jcr_strut_chassis', 'F_rbr_upper_strut_jcr_strut', 'T_rbr_upper_strut_jcr_strut', 'F_rbr_upper_strut_far_strut', 'T_rbr_upper_strut_far_strut', 'F_rbl_upper_strut_jcl_strut_chassis', 'T_rbl_upper_strut_jcl_strut_chassis', 'F_rbl_upper_strut_jcl_strut', 'T_rbl_upper_strut_jcl_strut', 'F_rbl_upper_strut_fal_strut', 'T_rbl_upper_strut_fal_strut', 'F_rbr_lower_strut_jcr_strut_rocker', 'T_rbr_lower_strut_jcr_strut_rocker', 'F_rbl_lower_strut_jcl_strut_rocker', 'T_rbl_lower_strut_jcl_strut_rocker']
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
        self.jac_cols = np.array([self.rbr_uca*2, self.rbr_uca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.rbr_pushrod*2, self.rbr_pushrod*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.rbr_pushrod*2, self.rbr_pushrod*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbl_pushrod*2, self.rbl_pushrod*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbl_pushrod*2, self.rbl_pushrod*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.vbr_steer*2, self.vbr_steer*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.vbr_steer*2, self.vbr_steer*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.vbl_steer*2, self.vbl_steer*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.vbl_steer*2, self.vbl_steer*2+1, self.rbr_pushrod*2, self.rbr_pushrod*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_pushrod*2, self.rbl_pushrod*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbr_pushrod*2, self.rbr_pushrod*2+1, self.rbl_pushrod*2, self.rbl_pushrod*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1])

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
        self.config.R_rbr_tie_rod,
        self.config.P_rbr_tie_rod,
        self.config.R_rbl_tie_rod,
        self.config.P_rbl_tie_rod,
        self.config.R_rbr_hub,
        self.config.P_rbr_hub,
        self.config.R_rbl_hub,
        self.config.P_rbl_hub,
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
        self.config.P_rbl_lower_strut])
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
        self.config.Rd_rbr_tie_rod,
        self.config.Pd_rbr_tie_rod,
        self.config.Rd_rbl_tie_rod,
        self.config.Pd_rbl_tie_rod,
        self.config.Rd_rbr_hub,
        self.config.Pd_rbr_hub,
        self.config.Rd_rbl_hub,
        self.config.Pd_rbl_hub,
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
        self.config.Pd_rbl_lower_strut])

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
        self.rbr_tie_rod = indicies_map[p + 'rbr_tie_rod']
        self.rbl_tie_rod = indicies_map[p + 'rbl_tie_rod']
        self.rbr_hub = indicies_map[p + 'rbr_hub']
        self.rbl_hub = indicies_map[p + 'rbl_hub']
        self.rbr_pushrod = indicies_map[p + 'rbr_pushrod']
        self.rbl_pushrod = indicies_map[p + 'rbl_pushrod']
        self.rbr_rocker = indicies_map[p + 'rbr_rocker']
        self.rbl_rocker = indicies_map[p + 'rbl_rocker']
        self.rbr_upper_strut = indicies_map[p + 'rbr_upper_strut']
        self.rbl_upper_strut = indicies_map[p + 'rbl_upper_strut']
        self.rbr_lower_strut = indicies_map[p + 'rbr_lower_strut']
        self.rbl_lower_strut = indicies_map[p + 'rbl_lower_strut']
        self.vbs_ground = indicies_map[interface_map[p + 'vbs_ground']]
        self.vbr_steer = indicies_map[interface_map[p + 'vbr_steer']]
        self.vbs_chassis = indicies_map[interface_map[p + 'vbs_chassis']]
        self.vbl_steer = indicies_map[interface_map[p + 'vbl_steer']]

    
    def eval_constants(self):
        config = self.config

        self.F_rbr_uca_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_uca]], dtype=np.float64)
        self.F_rbl_uca_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_uca]], dtype=np.float64)
        self.F_rbr_lca_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_lca]], dtype=np.float64)
        self.F_rbl_lca_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_lca]], dtype=np.float64)
        self.F_rbr_upright_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_upright]], dtype=np.float64)
        self.F_rbl_upright_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_upright]], dtype=np.float64)
        self.F_rbr_tie_rod_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_tie_rod]], dtype=np.float64)
        self.F_rbl_tie_rod_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_tie_rod]], dtype=np.float64)
        self.F_rbr_hub_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_hub]], dtype=np.float64)
        self.F_rbr_hub_far_drive = np.zeros((3,1),dtype=np.float64)
        self.F_rbl_hub_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_hub]], dtype=np.float64)
        self.F_rbl_hub_fal_drive = np.zeros((3,1),dtype=np.float64)
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

        self.Mbar_rbr_uca_jcr_uca_upright = multi_dot([A(config.P_rbr_uca).T,triad(config.ax1_jcr_uca_upright)])
        self.Mbar_rbr_upright_jcr_uca_upright = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_uca_upright)])
        self.ubar_rbr_uca_jcr_uca_upright = (multi_dot([A(config.P_rbr_uca).T,config.pt1_jcr_uca_upright]) + (-1) * multi_dot([A(config.P_rbr_uca).T,config.R_rbr_uca]))
        self.ubar_rbr_upright_jcr_uca_upright = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_uca_upright]) + (-1) * multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.Mbar_rbr_uca_jcr_uca_chassis = multi_dot([A(config.P_rbr_uca).T,triad(config.ax1_jcr_uca_chassis)])
        self.Mbar_vbs_chassis_jcr_uca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcr_uca_chassis)])
        self.ubar_rbr_uca_jcr_uca_chassis = (multi_dot([A(config.P_rbr_uca).T,config.pt1_jcr_uca_chassis]) + (-1) * multi_dot([A(config.P_rbr_uca).T,config.R_rbr_uca]))
        self.ubar_vbs_chassis_jcr_uca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_uca_chassis]) + (-1) * multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbr_uca_jcr_prod_uca = multi_dot([A(config.P_rbr_uca).T,triad(config.ax1_jcr_prod_uca)])
        self.Mbar_rbr_pushrod_jcr_prod_uca = multi_dot([A(config.P_rbr_pushrod).T,triad(config.ax2_jcr_prod_uca,triad(config.ax1_jcr_prod_uca)[0:3,1:2])])
        self.ubar_rbr_uca_jcr_prod_uca = (multi_dot([A(config.P_rbr_uca).T,config.pt1_jcr_prod_uca]) + (-1) * multi_dot([A(config.P_rbr_uca).T,config.R_rbr_uca]))
        self.ubar_rbr_pushrod_jcr_prod_uca = (multi_dot([A(config.P_rbr_pushrod).T,config.pt1_jcr_prod_uca]) + (-1) * multi_dot([A(config.P_rbr_pushrod).T,config.R_rbr_pushrod]))
        self.Mbar_rbl_uca_jcl_uca_upright = multi_dot([A(config.P_rbl_uca).T,triad(config.ax1_jcl_uca_upright)])
        self.Mbar_rbl_upright_jcl_uca_upright = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_uca_upright)])
        self.ubar_rbl_uca_jcl_uca_upright = (multi_dot([A(config.P_rbl_uca).T,config.pt1_jcl_uca_upright]) + (-1) * multi_dot([A(config.P_rbl_uca).T,config.R_rbl_uca]))
        self.ubar_rbl_upright_jcl_uca_upright = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_uca_upright]) + (-1) * multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.Mbar_rbl_uca_jcl_uca_chassis = multi_dot([A(config.P_rbl_uca).T,triad(config.ax1_jcl_uca_chassis)])
        self.Mbar_vbs_chassis_jcl_uca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcl_uca_chassis)])
        self.ubar_rbl_uca_jcl_uca_chassis = (multi_dot([A(config.P_rbl_uca).T,config.pt1_jcl_uca_chassis]) + (-1) * multi_dot([A(config.P_rbl_uca).T,config.R_rbl_uca]))
        self.ubar_vbs_chassis_jcl_uca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_uca_chassis]) + (-1) * multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_uca_jcl_prod_uca = multi_dot([A(config.P_rbl_uca).T,triad(config.ax1_jcl_prod_uca)])
        self.Mbar_rbl_pushrod_jcl_prod_uca = multi_dot([A(config.P_rbl_pushrod).T,triad(config.ax2_jcl_prod_uca,triad(config.ax1_jcl_prod_uca)[0:3,1:2])])
        self.ubar_rbl_uca_jcl_prod_uca = (multi_dot([A(config.P_rbl_uca).T,config.pt1_jcl_prod_uca]) + (-1) * multi_dot([A(config.P_rbl_uca).T,config.R_rbl_uca]))
        self.ubar_rbl_pushrod_jcl_prod_uca = (multi_dot([A(config.P_rbl_pushrod).T,config.pt1_jcl_prod_uca]) + (-1) * multi_dot([A(config.P_rbl_pushrod).T,config.R_rbl_pushrod]))
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
        self.vbar_rbr_hub_far_drive = multi_dot([A(config.P_rbr_hub).T,config.ax1_far_drive,(multi_dot([config.ax1_far_drive.T,A(config.P_rbr_hub),A(config.P_rbr_hub).T,config.ax1_far_drive]))**(-1.0/2.0)])
        self.Mbar_rbr_hub_far_drive = multi_dot([A(config.P_rbr_hub).T,triad(config.ax1_far_drive)])
        self.Mbar_vbs_ground_far_drive = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_far_drive)])
        self.ubar_rbl_hub_fal_tire = (multi_dot([A(config.P_rbl_hub).T,config.pt1_fal_tire]) + (-1) * multi_dot([A(config.P_rbl_hub).T,config.R_rbl_hub]))
        self.ubar_vbs_ground_fal_tire = (multi_dot([A(config.P_vbs_ground).T,config.pt1_fal_tire]) + (-1) * multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))
        self.vbar_rbl_hub_fal_drive = multi_dot([A(config.P_rbl_hub).T,config.ax1_fal_drive,(multi_dot([config.ax1_fal_drive.T,A(config.P_rbl_hub),A(config.P_rbl_hub).T,config.ax1_fal_drive]))**(-1.0/2.0)])
        self.Mbar_rbl_hub_fal_drive = multi_dot([A(config.P_rbl_hub).T,triad(config.ax1_fal_drive)])
        self.Mbar_vbs_ground_fal_drive = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_fal_drive)])
        self.Mbar_rbr_rocker_jcr_prod_rocker = multi_dot([A(config.P_rbr_rocker).T,triad(config.ax1_jcr_prod_rocker)])
        self.Mbar_rbr_pushrod_jcr_prod_rocker = multi_dot([A(config.P_rbr_pushrod).T,triad(config.ax1_jcr_prod_rocker)])
        self.ubar_rbr_rocker_jcr_prod_rocker = (multi_dot([A(config.P_rbr_rocker).T,config.pt1_jcr_prod_rocker]) + (-1) * multi_dot([A(config.P_rbr_rocker).T,config.R_rbr_rocker]))
        self.ubar_rbr_pushrod_jcr_prod_rocker = (multi_dot([A(config.P_rbr_pushrod).T,config.pt1_jcr_prod_rocker]) + (-1) * multi_dot([A(config.P_rbr_pushrod).T,config.R_rbr_pushrod]))
        self.Mbar_rbr_rocker_jcr_rocker_chassis = multi_dot([A(config.P_rbr_rocker).T,triad(config.ax1_jcr_rocker_chassis)])
        self.Mbar_vbs_chassis_jcr_rocker_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcr_rocker_chassis)])
        self.ubar_rbr_rocker_jcr_rocker_chassis = (multi_dot([A(config.P_rbr_rocker).T,config.pt1_jcr_rocker_chassis]) + (-1) * multi_dot([A(config.P_rbr_rocker).T,config.R_rbr_rocker]))
        self.ubar_vbs_chassis_jcr_rocker_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_rocker_chassis]) + (-1) * multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_rocker_jcl_prod_rocker = multi_dot([A(config.P_rbl_rocker).T,triad(config.ax1_jcl_prod_rocker)])
        self.Mbar_rbl_pushrod_jcl_prod_rocker = multi_dot([A(config.P_rbl_pushrod).T,triad(config.ax1_jcl_prod_rocker)])
        self.ubar_rbl_rocker_jcl_prod_rocker = (multi_dot([A(config.P_rbl_rocker).T,config.pt1_jcl_prod_rocker]) + (-1) * multi_dot([A(config.P_rbl_rocker).T,config.R_rbl_rocker]))
        self.ubar_rbl_pushrod_jcl_prod_rocker = (multi_dot([A(config.P_rbl_pushrod).T,config.pt1_jcl_prod_rocker]) + (-1) * multi_dot([A(config.P_rbl_pushrod).T,config.R_rbl_pushrod]))
        self.Mbar_rbl_rocker_jcl_rocker_chassis = multi_dot([A(config.P_rbl_rocker).T,triad(config.ax1_jcl_rocker_chassis)])
        self.Mbar_vbs_chassis_jcl_rocker_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcl_rocker_chassis)])
        self.ubar_rbl_rocker_jcl_rocker_chassis = (multi_dot([A(config.P_rbl_rocker).T,config.pt1_jcl_rocker_chassis]) + (-1) * multi_dot([A(config.P_rbl_rocker).T,config.R_rbl_rocker]))
        self.ubar_vbs_chassis_jcl_rocker_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_rocker_chassis]) + (-1) * multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
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
        self.Mbar_rbr_lower_strut_jcr_strut_rocker = multi_dot([A(config.P_rbr_lower_strut).T,triad(config.ax1_jcr_strut_rocker)])
        self.Mbar_rbr_rocker_jcr_strut_rocker = multi_dot([A(config.P_rbr_rocker).T,triad(config.ax2_jcr_strut_rocker,triad(config.ax1_jcr_strut_rocker)[0:3,1:2])])
        self.ubar_rbr_lower_strut_jcr_strut_rocker = (multi_dot([A(config.P_rbr_lower_strut).T,config.pt1_jcr_strut_rocker]) + (-1) * multi_dot([A(config.P_rbr_lower_strut).T,config.R_rbr_lower_strut]))
        self.ubar_rbr_rocker_jcr_strut_rocker = (multi_dot([A(config.P_rbr_rocker).T,config.pt1_jcr_strut_rocker]) + (-1) * multi_dot([A(config.P_rbr_rocker).T,config.R_rbr_rocker]))
        self.Mbar_rbl_lower_strut_jcl_strut_rocker = multi_dot([A(config.P_rbl_lower_strut).T,triad(config.ax1_jcl_strut_rocker)])
        self.Mbar_rbl_rocker_jcl_strut_rocker = multi_dot([A(config.P_rbl_rocker).T,triad(config.ax2_jcl_strut_rocker,triad(config.ax1_jcl_strut_rocker)[0:3,1:2])])
        self.ubar_rbl_lower_strut_jcl_strut_rocker = (multi_dot([A(config.P_rbl_lower_strut).T,config.pt1_jcl_strut_rocker]) + (-1) * multi_dot([A(config.P_rbl_lower_strut).T,config.R_rbl_lower_strut]))
        self.ubar_rbl_rocker_jcl_strut_rocker = (multi_dot([A(config.P_rbl_rocker).T,config.pt1_jcl_strut_rocker]) + (-1) * multi_dot([A(config.P_rbl_rocker).T,config.R_rbl_rocker]))

    
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
        self.R_rbr_tie_rod = q[42:45,0:1]
        self.P_rbr_tie_rod = q[45:49,0:1]
        self.R_rbl_tie_rod = q[49:52,0:1]
        self.P_rbl_tie_rod = q[52:56,0:1]
        self.R_rbr_hub = q[56:59,0:1]
        self.P_rbr_hub = q[59:63,0:1]
        self.R_rbl_hub = q[63:66,0:1]
        self.P_rbl_hub = q[66:70,0:1]
        self.R_rbr_pushrod = q[70:73,0:1]
        self.P_rbr_pushrod = q[73:77,0:1]
        self.R_rbl_pushrod = q[77:80,0:1]
        self.P_rbl_pushrod = q[80:84,0:1]
        self.R_rbr_rocker = q[84:87,0:1]
        self.P_rbr_rocker = q[87:91,0:1]
        self.R_rbl_rocker = q[91:94,0:1]
        self.P_rbl_rocker = q[94:98,0:1]
        self.R_rbr_upper_strut = q[98:101,0:1]
        self.P_rbr_upper_strut = q[101:105,0:1]
        self.R_rbl_upper_strut = q[105:108,0:1]
        self.P_rbl_upper_strut = q[108:112,0:1]
        self.R_rbr_lower_strut = q[112:115,0:1]
        self.P_rbr_lower_strut = q[115:119,0:1]
        self.R_rbl_lower_strut = q[119:122,0:1]
        self.P_rbl_lower_strut = q[122:126,0:1]

    
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
        self.Rd_rbr_tie_rod = qd[42:45,0:1]
        self.Pd_rbr_tie_rod = qd[45:49,0:1]
        self.Rd_rbl_tie_rod = qd[49:52,0:1]
        self.Pd_rbl_tie_rod = qd[52:56,0:1]
        self.Rd_rbr_hub = qd[56:59,0:1]
        self.Pd_rbr_hub = qd[59:63,0:1]
        self.Rd_rbl_hub = qd[63:66,0:1]
        self.Pd_rbl_hub = qd[66:70,0:1]
        self.Rd_rbr_pushrod = qd[70:73,0:1]
        self.Pd_rbr_pushrod = qd[73:77,0:1]
        self.Rd_rbl_pushrod = qd[77:80,0:1]
        self.Pd_rbl_pushrod = qd[80:84,0:1]
        self.Rd_rbr_rocker = qd[84:87,0:1]
        self.Pd_rbr_rocker = qd[87:91,0:1]
        self.Rd_rbl_rocker = qd[91:94,0:1]
        self.Pd_rbl_rocker = qd[94:98,0:1]
        self.Rd_rbr_upper_strut = qd[98:101,0:1]
        self.Pd_rbr_upper_strut = qd[101:105,0:1]
        self.Rd_rbl_upper_strut = qd[105:108,0:1]
        self.Pd_rbl_upper_strut = qd[108:112,0:1]
        self.Rd_rbr_lower_strut = qd[112:115,0:1]
        self.Pd_rbr_lower_strut = qd[115:119,0:1]
        self.Rd_rbl_lower_strut = qd[119:122,0:1]
        self.Pd_rbl_lower_strut = qd[122:126,0:1]

    
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
        self.Rdd_rbr_tie_rod = qdd[42:45,0:1]
        self.Pdd_rbr_tie_rod = qdd[45:49,0:1]
        self.Rdd_rbl_tie_rod = qdd[49:52,0:1]
        self.Pdd_rbl_tie_rod = qdd[52:56,0:1]
        self.Rdd_rbr_hub = qdd[56:59,0:1]
        self.Pdd_rbr_hub = qdd[59:63,0:1]
        self.Rdd_rbl_hub = qdd[63:66,0:1]
        self.Pdd_rbl_hub = qdd[66:70,0:1]
        self.Rdd_rbr_pushrod = qdd[70:73,0:1]
        self.Pdd_rbr_pushrod = qdd[73:77,0:1]
        self.Rdd_rbl_pushrod = qdd[77:80,0:1]
        self.Pdd_rbl_pushrod = qdd[80:84,0:1]
        self.Rdd_rbr_rocker = qdd[84:87,0:1]
        self.Pdd_rbr_rocker = qdd[87:91,0:1]
        self.Rdd_rbl_rocker = qdd[91:94,0:1]
        self.Pdd_rbl_rocker = qdd[94:98,0:1]
        self.Rdd_rbr_upper_strut = qdd[98:101,0:1]
        self.Pdd_rbr_upper_strut = qdd[101:105,0:1]
        self.Rdd_rbl_upper_strut = qdd[105:108,0:1]
        self.Pdd_rbl_upper_strut = qdd[108:112,0:1]
        self.Rdd_rbr_lower_strut = qdd[112:115,0:1]
        self.Pdd_rbr_lower_strut = qdd[115:119,0:1]
        self.Rdd_rbl_lower_strut = qdd[119:122,0:1]
        self.Pdd_rbl_lower_strut = qdd[122:126,0:1]

    
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
        self.L_jcr_tie_upright = Lambda[50:53,0:1]
        self.L_jcr_tie_steering = Lambda[53:57,0:1]
        self.L_jcl_tie_upright = Lambda[57:60,0:1]
        self.L_jcl_tie_steering = Lambda[60:64,0:1]
        self.L_jcr_prod_rocker = Lambda[64:67,0:1]
        self.L_jcr_rocker_chassis = Lambda[67:72,0:1]
        self.L_jcl_prod_rocker = Lambda[72:75,0:1]
        self.L_jcl_rocker_chassis = Lambda[75:80,0:1]
        self.L_jcr_strut_chassis = Lambda[80:84,0:1]
        self.L_jcr_strut = Lambda[84:88,0:1]
        self.L_jcl_strut_chassis = Lambda[88:92,0:1]
        self.L_jcl_strut = Lambda[92:96,0:1]
        self.L_jcr_strut_rocker = Lambda[96:100,0:1]
        self.L_jcl_strut_rocker = Lambda[100:104,0:1]

    
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
        x11 = (-1) * self.R_rbr_pushrod
        x12 = self.P_rbr_pushrod
        x13 = A(x12)
        x14 = self.R_rbl_uca
        x15 = self.R_rbl_upright
        x16 = (-1) * x15
        x17 = self.P_rbl_uca
        x18 = A(x17)
        x19 = self.P_rbl_upright
        x20 = A(x19)
        x21 = x18.T
        x22 = self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        x23 = (-1) * self.R_rbl_pushrod
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
        x44 = self.R_rbr_tie_rod
        x45 = self.P_rbr_tie_rod
        x46 = A(x45)
        x47 = A(self.P_vbr_steer)
        x48 = self.R_rbl_tie_rod
        x49 = self.P_rbl_tie_rod
        x50 = A(x49)
        x51 = A(self.P_vbl_steer)
        x52 = self.R_rbr_rocker
        x53 = self.P_rbr_rocker
        x54 = A(x53)
        x55 = x54.T
        x56 = self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]
        x57 = self.R_rbl_rocker
        x58 = self.P_rbl_rocker
        x59 = A(x58)
        x60 = x59.T
        x61 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]
        x62 = self.R_rbr_upper_strut
        x63 = self.P_rbr_upper_strut
        x64 = A(x63)
        x65 = x64.T
        x66 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1].T
        x67 = self.P_rbr_lower_strut
        x68 = A(x67)
        x69 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        x70 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2].T
        x71 = self.R_rbr_lower_strut
        x72 = (x62 + (-1) * x71 + multi_dot([x64,self.ubar_rbr_upper_strut_jcr_strut]) + (-1) * multi_dot([x68,self.ubar_rbr_lower_strut_jcr_strut]))
        x73 = self.R_rbl_upper_strut
        x74 = self.P_rbl_upper_strut
        x75 = A(x74)
        x76 = x75.T
        x77 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1].T
        x78 = self.P_rbl_lower_strut
        x79 = A(x78)
        x80 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        x81 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2].T
        x82 = self.R_rbl_lower_strut
        x83 = (x73 + (-1) * x82 + multi_dot([x75,self.ubar_rbl_upper_strut_jcl_strut]) + (-1) * multi_dot([x79,self.ubar_rbl_lower_strut_jcl_strut]))
        x84 = (-1) * np.eye(1, dtype=np.float64)

        self.pos_eq_blocks = ((x0 + x2 + multi_dot([x4,self.ubar_rbr_uca_jcr_uca_upright]) + (-1) * multi_dot([x6,self.ubar_rbr_upright_jcr_uca_upright])),
        (x0 + x7 + multi_dot([x4,self.ubar_rbr_uca_jcr_uca_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcr_uca_chassis])),
        multi_dot([self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1].T,x9,x8,x10]),
        multi_dot([self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2].T,x9,x8,x10]),
        (x0 + x11 + multi_dot([x4,self.ubar_rbr_uca_jcr_prod_uca]) + (-1) * multi_dot([x13,self.ubar_rbr_pushrod_jcr_prod_uca])),
        multi_dot([self.Mbar_rbr_uca_jcr_prod_uca[:,0:1].T,x9,x13,self.Mbar_rbr_pushrod_jcr_prod_uca[:,0:1]]),
        (x14 + x16 + multi_dot([x18,self.ubar_rbl_uca_jcl_uca_upright]) + (-1) * multi_dot([x20,self.ubar_rbl_upright_jcl_uca_upright])),
        (x14 + x7 + multi_dot([x18,self.ubar_rbl_uca_jcl_uca_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcl_uca_chassis])),
        multi_dot([self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1].T,x21,x8,x22]),
        multi_dot([self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2].T,x21,x8,x22]),
        (x14 + x23 + multi_dot([x18,self.ubar_rbl_uca_jcl_prod_uca]) + (-1) * multi_dot([x25,self.ubar_rbl_pushrod_jcl_prod_uca])),
        multi_dot([self.Mbar_rbl_uca_jcl_prod_uca[:,0:1].T,x21,x25,self.Mbar_rbl_pushrod_jcl_prod_uca[:,0:1]]),
        (x26 + x2 + multi_dot([x28,self.ubar_rbr_lca_jcr_lca_upright]) + (-1) * multi_dot([x6,self.ubar_rbr_upright_jcr_lca_upright])),
        (x26 + x7 + multi_dot([x28,self.ubar_rbr_lca_jcr_lca_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcr_lca_chassis])),
        multi_dot([self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1].T,x29,x8,x30]),
        multi_dot([self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2].T,x29,x8,x30]),
        (x31 + x16 + multi_dot([x33,self.ubar_rbl_lca_jcl_lca_upright]) + (-1) * multi_dot([x20,self.ubar_rbl_upright_jcl_lca_upright])),
        (x31 + x7 + multi_dot([x33,self.ubar_rbl_lca_jcl_lca_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcl_lca_chassis])),
        multi_dot([self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1].T,x34,x8,x35]),
        multi_dot([self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2].T,x34,x8,x35]),
        (x1 + (-1) * self.R_rbr_hub + multi_dot([x6,self.ubar_rbr_upright_jcr_hub_bearing]) + (-1) * multi_dot([x37,self.ubar_rbr_hub_jcr_hub_bearing])),
        multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x38,x37,x39]),
        multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x38,x37,x39]),
        (x15 + (-1) * self.R_rbl_hub + multi_dot([x20,self.ubar_rbl_upright_jcl_hub_bearing]) + (-1) * multi_dot([x41,self.ubar_rbl_hub_jcl_hub_bearing])),
        multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x42,x41,x43]),
        multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x42,x41,x43]),
        (x44 + x2 + multi_dot([x46,self.ubar_rbr_tie_rod_jcr_tie_upright]) + (-1) * multi_dot([x6,self.ubar_rbr_upright_jcr_tie_upright])),
        (x44 + (-1) * self.R_vbr_steer + multi_dot([x46,self.ubar_rbr_tie_rod_jcr_tie_steering]) + (-1) * multi_dot([x47,self.ubar_vbr_steer_jcr_tie_steering])),
        multi_dot([self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x46.T,x47,self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]]),
        (x48 + x16 + multi_dot([x50,self.ubar_rbl_tie_rod_jcl_tie_upright]) + (-1) * multi_dot([x20,self.ubar_rbl_upright_jcl_tie_upright])),
        (x48 + (-1) * self.R_vbl_steer + multi_dot([x50,self.ubar_rbl_tie_rod_jcl_tie_steering]) + (-1) * multi_dot([x51,self.ubar_vbl_steer_jcl_tie_steering])),
        multi_dot([self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1].T,x50.T,x51,self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]]),
        (x52 + x11 + multi_dot([x54,self.ubar_rbr_rocker_jcr_prod_rocker]) + (-1) * multi_dot([x13,self.ubar_rbr_pushrod_jcr_prod_rocker])),
        (x52 + x7 + multi_dot([x54,self.ubar_rbr_rocker_jcr_rocker_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcr_rocker_chassis])),
        multi_dot([self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1].T,x55,x8,x56]),
        multi_dot([self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2].T,x55,x8,x56]),
        (x57 + x23 + multi_dot([x59,self.ubar_rbl_rocker_jcl_prod_rocker]) + (-1) * multi_dot([x25,self.ubar_rbl_pushrod_jcl_prod_rocker])),
        (x57 + x7 + multi_dot([x59,self.ubar_rbl_rocker_jcl_rocker_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcl_rocker_chassis])),
        multi_dot([self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1].T,x60,x8,x61]),
        multi_dot([self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2].T,x60,x8,x61]),
        (x62 + x7 + multi_dot([x64,self.ubar_rbr_upper_strut_jcr_strut_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcr_strut_chassis])),
        multi_dot([self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1].T,x65,x8,self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]]),
        multi_dot([x66,x65,x68,x69]),
        multi_dot([x70,x65,x68,x69]),
        multi_dot([x66,x65,x72]),
        multi_dot([x70,x65,x72]),
        (x73 + x7 + multi_dot([x75,self.ubar_rbl_upper_strut_jcl_strut_chassis]) + (-1) * multi_dot([x8,self.ubar_vbs_chassis_jcl_strut_chassis])),
        multi_dot([self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1].T,x76,x8,self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]]),
        multi_dot([x77,x76,x79,x80]),
        multi_dot([x81,x76,x79,x80]),
        multi_dot([x77,x76,x83]),
        multi_dot([x81,x76,x83]),
        (x71 + (-1) * x52 + multi_dot([x68,self.ubar_rbr_lower_strut_jcr_strut_rocker]) + (-1) * multi_dot([x54,self.ubar_rbr_rocker_jcr_strut_rocker])),
        multi_dot([self.Mbar_rbr_lower_strut_jcr_strut_rocker[:,0:1].T,x68.T,x54,self.Mbar_rbr_rocker_jcr_strut_rocker[:,0:1]]),
        (x82 + (-1) * x57 + multi_dot([x79,self.ubar_rbl_lower_strut_jcl_strut_rocker]) + (-1) * multi_dot([x59,self.ubar_rbl_rocker_jcl_strut_rocker])),
        multi_dot([self.Mbar_rbl_lower_strut_jcl_strut_rocker[:,0:1].T,x79.T,x59,self.Mbar_rbl_rocker_jcl_strut_rocker[:,0:1]]),
        (x84 + multi_dot([x3.T,x3])),
        (x84 + multi_dot([x17.T,x17])),
        (x84 + multi_dot([x27.T,x27])),
        (x84 + multi_dot([x32.T,x32])),
        (x84 + multi_dot([x5.T,x5])),
        (x84 + multi_dot([x19.T,x19])),
        (x84 + multi_dot([x45.T,x45])),
        (x84 + multi_dot([x49.T,x49])),
        (x84 + multi_dot([x36.T,x36])),
        (x84 + multi_dot([x40.T,x40])),
        (x84 + multi_dot([x12.T,x12])),
        (x84 + multi_dot([x24.T,x24])),
        (x84 + multi_dot([x53.T,x53])),
        (x84 + multi_dot([x58.T,x58])),
        (x84 + multi_dot([x63.T,x63])),
        (x84 + multi_dot([x74.T,x74])),
        (x84 + multi_dot([x67.T,x67])),
        (x84 + multi_dot([x78.T,x78])),)

    
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
        v0,
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
        a77 = self.Pd_rbr_tie_rod
        a78 = self.Pd_vbr_steer
        a79 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        a80 = self.P_rbr_tie_rod
        a81 = self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        a82 = self.P_vbr_steer
        a83 = a77.T
        a84 = self.Pd_rbl_tie_rod
        a85 = self.Pd_vbl_steer
        a86 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        a87 = self.P_rbl_tie_rod
        a88 = self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        a89 = self.P_vbl_steer
        a90 = a84.T
        a91 = self.Pd_rbr_rocker
        a92 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1]
        a93 = self.P_rbr_rocker
        a94 = A(a93).T
        a95 = self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]
        a96 = B(a2,a95)
        a97 = a95.T
        a98 = a91.T
        a99 = B(a9,a95)
        a100 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2]
        a101 = self.Pd_rbl_rocker
        a102 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]
        a103 = self.P_rbl_rocker
        a104 = A(a103).T
        a105 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]
        a106 = B(a2,a105)
        a107 = a105.T
        a108 = a101.T
        a109 = B(a9,a105)
        a110 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]
        a111 = self.Pd_rbr_upper_strut
        a112 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        a113 = self.P_rbr_upper_strut
        a114 = A(a113).T
        a115 = self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        a116 = a111.T
        a117 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        a118 = a117.T
        a119 = self.P_rbr_lower_strut
        a120 = A(a119).T
        a121 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        a122 = B(a111,a121)
        a123 = a121.T
        a124 = self.Pd_rbr_lower_strut
        a125 = B(a124,a117)
        a126 = B(a113,a121).T
        a127 = B(a119,a117)
        a128 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        a129 = B(a111,a128)
        a130 = a128.T
        a131 = B(a113,a128).T
        a132 = self.ubar_rbr_upper_strut_jcr_strut
        a133 = self.ubar_rbr_lower_strut_jcr_strut
        a134 = (multi_dot([B(a111,a132),a111]) + (-1) * multi_dot([B(a124,a133),a124]))
        a135 = (self.Rd_rbr_upper_strut + (-1) * self.Rd_rbr_lower_strut + multi_dot([B(a113,a132),a111]) + (-1) * multi_dot([B(a119,a133),a124]))
        a136 = (self.R_rbr_upper_strut.T + (-1) * self.R_rbr_lower_strut.T + multi_dot([a132.T,a114]) + (-1) * multi_dot([a133.T,a120]))
        a137 = self.Pd_rbl_upper_strut
        a138 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        a139 = self.P_rbl_upper_strut
        a140 = A(a139).T
        a141 = self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        a142 = a137.T
        a143 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        a144 = a143.T
        a145 = self.P_rbl_lower_strut
        a146 = A(a145).T
        a147 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        a148 = B(a137,a147)
        a149 = a147.T
        a150 = self.Pd_rbl_lower_strut
        a151 = B(a150,a143)
        a152 = B(a139,a147).T
        a153 = B(a145,a143)
        a154 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        a155 = B(a137,a154)
        a156 = a154.T
        a157 = B(a139,a154).T
        a158 = self.ubar_rbl_upper_strut_jcl_strut
        a159 = self.ubar_rbl_lower_strut_jcl_strut
        a160 = (multi_dot([B(a137,a158),a137]) + (-1) * multi_dot([B(a150,a159),a150]))
        a161 = (self.Rd_rbl_upper_strut + (-1) * self.Rd_rbl_lower_strut + multi_dot([B(a139,a158),a137]) + (-1) * multi_dot([B(a145,a159),a150]))
        a162 = (self.R_rbl_upper_strut.T + (-1) * self.R_rbl_lower_strut.T + multi_dot([a158.T,a140]) + (-1) * multi_dot([a159.T,a146]))
        a163 = self.Mbar_rbr_lower_strut_jcr_strut_rocker[:,0:1]
        a164 = self.Mbar_rbr_rocker_jcr_strut_rocker[:,0:1]
        a165 = a124.T
        a166 = self.Mbar_rbl_lower_strut_jcl_strut_rocker[:,0:1]
        a167 = self.Mbar_rbl_rocker_jcl_strut_rocker[:,0:1]
        a168 = a150.T

        self.acc_eq_blocks = ((multi_dot([B(a0,self.ubar_rbr_uca_jcr_uca_upright),a0]) + (-1) * multi_dot([B(a1,self.ubar_rbr_upright_jcr_uca_upright),a1])),
        (multi_dot([B(a0,self.ubar_rbr_uca_jcr_uca_chassis),a0]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcr_uca_chassis),a2])),
        (multi_dot([a3.T,a5,a7,a2]) + multi_dot([a8,a10,B(a0,a3),a0]) + (2) * multi_dot([a11,B(a4,a3).T,a12,a2])),
        (multi_dot([a13.T,a5,a7,a2]) + multi_dot([a8,a10,B(a0,a13),a0]) + (2) * multi_dot([a11,B(a4,a13).T,a12,a2])),
        (multi_dot([B(a0,self.ubar_rbr_uca_jcr_prod_uca),a0]) + (-1) * multi_dot([B(a14,self.ubar_rbr_pushrod_jcr_prod_uca),a14])),
        (multi_dot([a15.T,a5,B(a14,a16),a14]) + multi_dot([a16.T,A(a17).T,B(a0,a15),a0]) + (2) * multi_dot([a11,B(a4,a15).T,B(a17,a16),a14])),
        (multi_dot([B(a18,self.ubar_rbl_uca_jcl_uca_upright),a18]) + (-1) * multi_dot([B(a19,self.ubar_rbl_upright_jcl_uca_upright),a19])),
        (multi_dot([B(a18,self.ubar_rbl_uca_jcl_uca_chassis),a18]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcl_uca_chassis),a2])),
        (multi_dot([a20.T,a22,a24,a2]) + multi_dot([a25,a10,B(a18,a20),a18]) + (2) * multi_dot([a26,B(a21,a20).T,a27,a2])),
        (multi_dot([a28.T,a22,a24,a2]) + multi_dot([a25,a10,B(a18,a28),a18]) + (2) * multi_dot([a26,B(a21,a28).T,a27,a2])),
        (multi_dot([B(a18,self.ubar_rbl_uca_jcl_prod_uca),a18]) + (-1) * multi_dot([B(a29,self.ubar_rbl_pushrod_jcl_prod_uca),a29])),
        (multi_dot([a30.T,a22,B(a29,a31),a29]) + multi_dot([a31.T,A(a32).T,B(a18,a30),a18]) + (2) * multi_dot([a26,B(a21,a30).T,B(a32,a31),a29])),
        (multi_dot([B(a33,self.ubar_rbr_lca_jcr_lca_upright),a33]) + (-1) * multi_dot([B(a1,self.ubar_rbr_upright_jcr_lca_upright),a1])),
        (multi_dot([B(a33,self.ubar_rbr_lca_jcr_lca_chassis),a33]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcr_lca_chassis),a2])),
        (multi_dot([a34.T,a36,a38,a2]) + multi_dot([a39,a10,B(a33,a34),a33]) + (2) * multi_dot([a40,B(a35,a34).T,a41,a2])),
        (multi_dot([a42.T,a36,a38,a2]) + multi_dot([a39,a10,B(a33,a42),a33]) + (2) * multi_dot([a40,B(a35,a42).T,a41,a2])),
        (multi_dot([B(a43,self.ubar_rbl_lca_jcl_lca_upright),a43]) + (-1) * multi_dot([B(a19,self.ubar_rbl_upright_jcl_lca_upright),a19])),
        (multi_dot([B(a43,self.ubar_rbl_lca_jcl_lca_chassis),a43]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcl_lca_chassis),a2])),
        (multi_dot([a44.T,a46,a48,a2]) + multi_dot([a49,a10,B(a43,a44),a43]) + (2) * multi_dot([a50,B(a45,a44).T,a51,a2])),
        (multi_dot([a52.T,a46,a48,a2]) + multi_dot([a49,a10,B(a43,a52),a43]) + (2) * multi_dot([a50,B(a45,a52).T,a51,a2])),
        (multi_dot([B(a1,self.ubar_rbr_upright_jcr_hub_bearing),a1]) + (-1) * multi_dot([B(a53,self.ubar_rbr_hub_jcr_hub_bearing),a53])),
        (multi_dot([a54.T,a56,a58,a53]) + multi_dot([a59,a61,B(a1,a54),a1]) + (2) * multi_dot([a62,B(a55,a54).T,a63,a53])),
        (multi_dot([a64.T,a56,a58,a53]) + multi_dot([a59,a61,B(a1,a64),a1]) + (2) * multi_dot([a62,B(a55,a64).T,a63,a53])),
        (multi_dot([B(a19,self.ubar_rbl_upright_jcl_hub_bearing),a19]) + (-1) * multi_dot([B(a65,self.ubar_rbl_hub_jcl_hub_bearing),a65])),
        (multi_dot([a66.T,a68,a70,a65]) + multi_dot([a71,a73,B(a19,a66),a19]) + (2) * multi_dot([a74,B(a67,a66).T,a75,a65])),
        (multi_dot([a76.T,a68,a70,a65]) + multi_dot([a71,a73,B(a19,a76),a19]) + (2) * multi_dot([a74,B(a67,a76).T,a75,a65])),
        (multi_dot([B(a77,self.ubar_rbr_tie_rod_jcr_tie_upright),a77]) + (-1) * multi_dot([B(a1,self.ubar_rbr_upright_jcr_tie_upright),a1])),
        (multi_dot([B(a77,self.ubar_rbr_tie_rod_jcr_tie_steering),a77]) + (-1) * multi_dot([B(a78,self.ubar_vbr_steer_jcr_tie_steering),a78])),
        (multi_dot([a79.T,A(a80).T,B(a78,a81),a78]) + multi_dot([a81.T,A(a82).T,B(a77,a79),a77]) + (2) * multi_dot([a83,B(a80,a79).T,B(a82,a81),a78])),
        (multi_dot([B(a84,self.ubar_rbl_tie_rod_jcl_tie_upright),a84]) + (-1) * multi_dot([B(a19,self.ubar_rbl_upright_jcl_tie_upright),a19])),
        (multi_dot([B(a84,self.ubar_rbl_tie_rod_jcl_tie_steering),a84]) + (-1) * multi_dot([B(a85,self.ubar_vbl_steer_jcl_tie_steering),a85])),
        (multi_dot([a86.T,A(a87).T,B(a85,a88),a85]) + multi_dot([a88.T,A(a89).T,B(a84,a86),a84]) + (2) * multi_dot([a90,B(a87,a86).T,B(a89,a88),a85])),
        (multi_dot([B(a91,self.ubar_rbr_rocker_jcr_prod_rocker),a91]) + (-1) * multi_dot([B(a14,self.ubar_rbr_pushrod_jcr_prod_rocker),a14])),
        (multi_dot([B(a91,self.ubar_rbr_rocker_jcr_rocker_chassis),a91]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcr_rocker_chassis),a2])),
        (multi_dot([a92.T,a94,a96,a2]) + multi_dot([a97,a10,B(a91,a92),a91]) + (2) * multi_dot([a98,B(a93,a92).T,a99,a2])),
        (multi_dot([a100.T,a94,a96,a2]) + multi_dot([a97,a10,B(a91,a100),a91]) + (2) * multi_dot([a98,B(a93,a100).T,a99,a2])),
        (multi_dot([B(a101,self.ubar_rbl_rocker_jcl_prod_rocker),a101]) + (-1) * multi_dot([B(a29,self.ubar_rbl_pushrod_jcl_prod_rocker),a29])),
        (multi_dot([B(a101,self.ubar_rbl_rocker_jcl_rocker_chassis),a101]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcl_rocker_chassis),a2])),
        (multi_dot([a102.T,a104,a106,a2]) + multi_dot([a107,a10,B(a101,a102),a101]) + (2) * multi_dot([a108,B(a103,a102).T,a109,a2])),
        (multi_dot([a110.T,a104,a106,a2]) + multi_dot([a107,a10,B(a101,a110),a101]) + (2) * multi_dot([a108,B(a103,a110).T,a109,a2])),
        (multi_dot([B(a111,self.ubar_rbr_upper_strut_jcr_strut_chassis),a111]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcr_strut_chassis),a2])),
        (multi_dot([a112.T,a114,B(a2,a115),a2]) + multi_dot([a115.T,a10,B(a111,a112),a111]) + (2) * multi_dot([a116,B(a113,a112).T,B(a9,a115),a2])),
        (multi_dot([a118,a120,a122,a111]) + multi_dot([a123,a114,a125,a124]) + (2) * multi_dot([a116,a126,a127,a124])),
        (multi_dot([a118,a120,a129,a111]) + multi_dot([a130,a114,a125,a124]) + (2) * multi_dot([a116,a131,a127,a124])),
        (multi_dot([a123,a114,a134]) + (2) * multi_dot([a116,a126,a135]) + multi_dot([a136,a122,a111])),
        (multi_dot([a130,a114,a134]) + (2) * multi_dot([a116,a131,a135]) + multi_dot([a136,a129,a111])),
        (multi_dot([B(a137,self.ubar_rbl_upper_strut_jcl_strut_chassis),a137]) + (-1) * multi_dot([B(a2,self.ubar_vbs_chassis_jcl_strut_chassis),a2])),
        (multi_dot([a138.T,a140,B(a2,a141),a2]) + multi_dot([a141.T,a10,B(a137,a138),a137]) + (2) * multi_dot([a142,B(a139,a138).T,B(a9,a141),a2])),
        (multi_dot([a144,a146,a148,a137]) + multi_dot([a149,a140,a151,a150]) + (2) * multi_dot([a142,a152,a153,a150])),
        (multi_dot([a144,a146,a155,a137]) + multi_dot([a156,a140,a151,a150]) + (2) * multi_dot([a142,a157,a153,a150])),
        (multi_dot([a149,a140,a160]) + (2) * multi_dot([a142,a152,a161]) + multi_dot([a162,a148,a137])),
        (multi_dot([a156,a140,a160]) + (2) * multi_dot([a142,a157,a161]) + multi_dot([a162,a155,a137])),
        (multi_dot([B(a124,self.ubar_rbr_lower_strut_jcr_strut_rocker),a124]) + (-1) * multi_dot([B(a91,self.ubar_rbr_rocker_jcr_strut_rocker),a91])),
        (multi_dot([a163.T,a120,B(a91,a164),a91]) + multi_dot([a164.T,a94,B(a124,a163),a124]) + (2) * multi_dot([a165,B(a119,a163).T,B(a93,a164),a91])),
        (multi_dot([B(a150,self.ubar_rbl_lower_strut_jcl_strut_rocker),a150]) + (-1) * multi_dot([B(a101,self.ubar_rbl_rocker_jcl_strut_rocker),a101])),
        (multi_dot([a166.T,a146,B(a101,a167),a101]) + multi_dot([a167.T,a104,B(a150,a166),a150]) + (2) * multi_dot([a168,B(a145,a166).T,B(a103,a167),a101])),
        (2) * multi_dot([a11,a0]),
        (2) * multi_dot([a26,a18]),
        (2) * multi_dot([a40,a33]),
        (2) * multi_dot([a50,a43]),
        (2) * multi_dot([a62,a1]),
        (2) * multi_dot([a74,a19]),
        (2) * multi_dot([a83,a77]),
        (2) * multi_dot([a90,a84]),
        (2) * multi_dot([a53.T,a53]),
        (2) * multi_dot([a65.T,a65]),
        (2) * multi_dot([a14.T,a14]),
        (2) * multi_dot([a29.T,a29]),
        (2) * multi_dot([a98,a91]),
        (2) * multi_dot([a108,a101]),
        (2) * multi_dot([a116,a111]),
        (2) * multi_dot([a142,a137]),
        (2) * multi_dot([a165,a124]),
        (2) * multi_dot([a168,a150]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3, dtype=np.float64)
        j1 = self.P_rbr_uca
        j2 = (-1) * j0
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
        j57 = self.P_rbr_tie_rod
        j58 = self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        j59 = self.P_vbr_steer
        j60 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        j61 = self.P_rbl_tie_rod
        j62 = self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        j63 = self.P_vbl_steer
        j64 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        j65 = self.P_rbr_rocker
        j66 = self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]
        j67 = j66.T
        j68 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1]
        j69 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2]
        j70 = A(j65).T
        j71 = B(j7,j66)
        j72 = self.P_rbl_rocker
        j73 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]
        j74 = j73.T
        j75 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]
        j76 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]
        j77 = A(j72).T
        j78 = B(j7,j73)
        j79 = self.P_rbr_upper_strut
        j80 = self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        j81 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        j82 = A(j79).T
        j83 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        j84 = j83.T
        j85 = self.P_rbr_lower_strut
        j86 = A(j85).T
        j87 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        j88 = B(j79,j87)
        j89 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        j90 = B(j79,j89)
        j91 = j87.T
        j92 = multi_dot([j91,j82])
        j93 = self.ubar_rbr_upper_strut_jcr_strut
        j94 = B(j79,j93)
        j95 = self.ubar_rbr_lower_strut_jcr_strut
        j96 = (self.R_rbr_upper_strut.T + (-1) * self.R_rbr_lower_strut.T + multi_dot([j93.T,j82]) + (-1) * multi_dot([j95.T,j86]))
        j97 = j89.T
        j98 = multi_dot([j97,j82])
        j99 = B(j85,j83)
        j100 = B(j85,j95)
        j101 = self.P_rbl_upper_strut
        j102 = self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        j103 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        j104 = A(j101).T
        j105 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        j106 = j105.T
        j107 = self.P_rbl_lower_strut
        j108 = A(j107).T
        j109 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        j110 = B(j101,j109)
        j111 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        j112 = B(j101,j111)
        j113 = j109.T
        j114 = multi_dot([j113,j104])
        j115 = self.ubar_rbl_upper_strut_jcl_strut
        j116 = B(j101,j115)
        j117 = self.ubar_rbl_lower_strut_jcl_strut
        j118 = (self.R_rbl_upper_strut.T + (-1) * self.R_rbl_lower_strut.T + multi_dot([j115.T,j104]) + (-1) * multi_dot([j117.T,j108]))
        j119 = j111.T
        j120 = multi_dot([j119,j104])
        j121 = B(j107,j105)
        j122 = B(j107,j117)
        j123 = self.Mbar_rbr_rocker_jcr_strut_rocker[:,0:1]
        j124 = self.Mbar_rbr_lower_strut_jcr_strut_rocker[:,0:1]
        j125 = self.Mbar_rbl_rocker_jcl_strut_rocker[:,0:1]
        j126 = self.Mbar_rbl_lower_strut_jcl_strut_rocker[:,0:1]

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
        B(j1,self.ubar_rbr_uca_jcr_prod_uca),
        j2,
        (-1) * B(j14,self.ubar_rbr_pushrod_jcr_prod_uca),
        j4,
        multi_dot([j13.T,A(j14).T,B(j1,j15)]),
        j4,
        multi_dot([j15.T,j11,B(j14,j13)]),
        j0,
        B(j16,self.ubar_rbl_uca_jcl_uca_upright),
        j2,
        (-1) * B(j17,self.ubar_rbl_upright_jcl_uca_upright),
        j0,
        B(j16,self.ubar_rbl_uca_jcl_uca_chassis),
        j2,
        (-1) * B(j7,self.ubar_vbs_chassis_jcl_uca_chassis),
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
        (-1) * B(j25,self.ubar_rbl_pushrod_jcl_prod_uca),
        j4,
        multi_dot([j24.T,A(j25).T,B(j16,j26)]),
        j4,
        multi_dot([j26.T,j22,B(j25,j24)]),
        j0,
        B(j27,self.ubar_rbr_lca_jcr_lca_upright),
        j2,
        (-1) * B(j3,self.ubar_rbr_upright_jcr_lca_upright),
        j0,
        B(j27,self.ubar_rbr_lca_jcr_lca_chassis),
        j2,
        (-1) * B(j7,self.ubar_vbs_chassis_jcr_lca_chassis),
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
        (-1) * B(j17,self.ubar_rbl_upright_jcl_lca_upright),
        j0,
        B(j34,self.ubar_rbl_lca_jcl_lca_chassis),
        j2,
        (-1) * B(j7,self.ubar_vbs_chassis_jcl_lca_chassis),
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
        (-1) * B(j43,self.ubar_rbr_hub_jcr_hub_bearing),
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
        (-1) * B(j51,self.ubar_rbl_hub_jcl_hub_bearing),
        j4,
        multi_dot([j50,j52,B(j17,j53)]),
        j4,
        multi_dot([j53.T,j55,j56]),
        j4,
        multi_dot([j50,j52,B(j17,j54)]),
        j4,
        multi_dot([j54.T,j55,j56]),
        j2,
        (-1) * B(j3,self.ubar_rbr_upright_jcr_tie_upright),
        j0,
        B(j57,self.ubar_rbr_tie_rod_jcr_tie_upright),
        j0,
        B(j57,self.ubar_rbr_tie_rod_jcr_tie_steering),
        j2,
        (-1) * B(j59,self.ubar_vbr_steer_jcr_tie_steering),
        j4,
        multi_dot([j58.T,A(j59).T,B(j57,j60)]),
        j4,
        multi_dot([j60.T,A(j57).T,B(j59,j58)]),
        j2,
        (-1) * B(j17,self.ubar_rbl_upright_jcl_tie_upright),
        j0,
        B(j61,self.ubar_rbl_tie_rod_jcl_tie_upright),
        j0,
        B(j61,self.ubar_rbl_tie_rod_jcl_tie_steering),
        j2,
        (-1) * B(j63,self.ubar_vbl_steer_jcl_tie_steering),
        j4,
        multi_dot([j62.T,A(j63).T,B(j61,j64)]),
        j4,
        multi_dot([j64.T,A(j61).T,B(j63,j62)]),
        j2,
        (-1) * B(j14,self.ubar_rbr_pushrod_jcr_prod_rocker),
        j0,
        B(j65,self.ubar_rbr_rocker_jcr_prod_rocker),
        j0,
        B(j65,self.ubar_rbr_rocker_jcr_rocker_chassis),
        j2,
        (-1) * B(j7,self.ubar_vbs_chassis_jcr_rocker_chassis),
        j4,
        multi_dot([j67,j8,B(j65,j68)]),
        j4,
        multi_dot([j68.T,j70,j71]),
        j4,
        multi_dot([j67,j8,B(j65,j69)]),
        j4,
        multi_dot([j69.T,j70,j71]),
        j2,
        (-1) * B(j25,self.ubar_rbl_pushrod_jcl_prod_rocker),
        j0,
        B(j72,self.ubar_rbl_rocker_jcl_prod_rocker),
        j0,
        B(j72,self.ubar_rbl_rocker_jcl_rocker_chassis),
        j2,
        (-1) * B(j7,self.ubar_vbs_chassis_jcl_rocker_chassis),
        j4,
        multi_dot([j74,j8,B(j72,j75)]),
        j4,
        multi_dot([j75.T,j77,j78]),
        j4,
        multi_dot([j74,j8,B(j72,j76)]),
        j4,
        multi_dot([j76.T,j77,j78]),
        j0,
        B(j79,self.ubar_rbr_upper_strut_jcr_strut_chassis),
        j2,
        (-1) * B(j7,self.ubar_vbs_chassis_jcr_strut_chassis),
        j4,
        multi_dot([j80.T,j8,B(j79,j81)]),
        j4,
        multi_dot([j81.T,j82,B(j7,j80)]),
        j4,
        multi_dot([j84,j86,j88]),
        j4,
        multi_dot([j91,j82,j99]),
        j4,
        multi_dot([j84,j86,j90]),
        j4,
        multi_dot([j97,j82,j99]),
        j92,
        (multi_dot([j91,j82,j94]) + multi_dot([j96,j88])),
        (-1) * j92,
        (-1) * multi_dot([j91,j82,j100]),
        j98,
        (multi_dot([j97,j82,j94]) + multi_dot([j96,j90])),
        (-1) * j98,
        (-1) * multi_dot([j97,j82,j100]),
        j0,
        B(j101,self.ubar_rbl_upper_strut_jcl_strut_chassis),
        j2,
        (-1) * B(j7,self.ubar_vbs_chassis_jcl_strut_chassis),
        j4,
        multi_dot([j102.T,j8,B(j101,j103)]),
        j4,
        multi_dot([j103.T,j104,B(j7,j102)]),
        j4,
        multi_dot([j106,j108,j110]),
        j4,
        multi_dot([j113,j104,j121]),
        j4,
        multi_dot([j106,j108,j112]),
        j4,
        multi_dot([j119,j104,j121]),
        j114,
        (multi_dot([j113,j104,j116]) + multi_dot([j118,j110])),
        (-1) * j114,
        (-1) * multi_dot([j113,j104,j122]),
        j120,
        (multi_dot([j119,j104,j116]) + multi_dot([j118,j112])),
        (-1) * j120,
        (-1) * multi_dot([j119,j104,j122]),
        j2,
        (-1) * B(j65,self.ubar_rbr_rocker_jcr_strut_rocker),
        j0,
        B(j85,self.ubar_rbr_lower_strut_jcr_strut_rocker),
        j4,
        multi_dot([j124.T,j86,B(j65,j123)]),
        j4,
        multi_dot([j123.T,j70,B(j85,j124)]),
        j2,
        (-1) * B(j72,self.ubar_rbl_rocker_jcl_strut_rocker),
        j0,
        B(j107,self.ubar_rbl_lower_strut_jcl_strut_rocker),
        j4,
        multi_dot([j126.T,j108,B(j72,j125)]),
        j4,
        multi_dot([j125.T,j77,B(j107,j126)]),
        j4,
        (2) * j1.T,
        j4,
        (2) * j16.T,
        j4,
        (2) * j27.T,
        j4,
        (2) * j34.T,
        j4,
        (2) * j3.T,
        j4,
        (2) * j17.T,
        j4,
        (2) * j57.T,
        j4,
        (2) * j61.T,
        j4,
        (2) * j43.T,
        j4,
        (2) * j51.T,
        j4,
        (2) * j14.T,
        j4,
        (2) * j25.T,
        j4,
        (2) * j65.T,
        j4,
        (2) * j72.T,
        j4,
        (2) * j79.T,
        j4,
        (2) * j101.T,
        j4,
        (2) * j85.T,
        j4,
        (2) * j107.T,)

    
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
        m7 = G(self.P_rbr_tie_rod)
        m8 = G(self.P_rbl_tie_rod)
        m9 = G(self.P_rbr_hub)
        m10 = G(self.P_rbl_hub)
        m11 = G(self.P_rbr_pushrod)
        m12 = G(self.P_rbl_pushrod)
        m13 = G(self.P_rbr_rocker)
        m14 = G(self.P_rbl_rocker)
        m15 = G(self.P_rbr_upper_strut)
        m16 = G(self.P_rbl_upper_strut)
        m17 = G(self.P_rbr_lower_strut)
        m18 = G(self.P_rbl_lower_strut)

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
        config.m_rbr_tie_rod * m0,
        (4) * multi_dot([m7.T,config.Jbar_rbr_tie_rod,m7]),
        config.m_rbl_tie_rod * m0,
        (4) * multi_dot([m8.T,config.Jbar_rbl_tie_rod,m8]),
        config.m_rbr_hub * m0,
        (4) * multi_dot([m9.T,config.Jbar_rbr_hub,m9]),
        config.m_rbl_hub * m0,
        (4) * multi_dot([m10.T,config.Jbar_rbl_hub,m10]),
        config.m_rbr_pushrod * m0,
        (4) * multi_dot([m11.T,config.Jbar_rbr_pushrod,m11]),
        config.m_rbl_pushrod * m0,
        (4) * multi_dot([m12.T,config.Jbar_rbl_pushrod,m12]),
        config.m_rbr_rocker * m0,
        (4) * multi_dot([m13.T,config.Jbar_rbr_rocker,m13]),
        config.m_rbl_rocker * m0,
        (4) * multi_dot([m14.T,config.Jbar_rbl_rocker,m14]),
        config.m_rbr_upper_strut * m0,
        (4) * multi_dot([m15.T,config.Jbar_rbr_upper_strut,m15]),
        config.m_rbl_upper_strut * m0,
        (4) * multi_dot([m16.T,config.Jbar_rbl_upper_strut,m16]),
        config.m_rbr_lower_strut * m0,
        (4) * multi_dot([m17.T,config.Jbar_rbr_lower_strut,m17]),
        config.m_rbl_lower_strut * m0,
        (4) * multi_dot([m18.T,config.Jbar_rbl_lower_strut,m18]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbr_uca)
        f1 = G(self.Pd_rbl_uca)
        f2 = G(self.Pd_rbr_lca)
        f3 = G(self.Pd_rbl_lca)
        f4 = G(self.Pd_rbr_upright)
        f5 = G(self.Pd_rbl_upright)
        f6 = G(self.Pd_rbr_tie_rod)
        f7 = G(self.Pd_rbl_tie_rod)
        f8 = np.zeros((3,1),dtype=np.float64)
        f9 = t
        f10 = config.UF_far_tire_F(f9)
        f11 = self.P_rbr_hub
        f12 = G(self.Pd_rbr_hub)
        f13 = config.UF_fal_tire_F(f9)
        f14 = self.P_rbl_hub
        f15 = G(self.Pd_rbl_hub)
        f16 = G(self.Pd_rbr_pushrod)
        f17 = G(self.Pd_rbl_pushrod)
        f18 = G(self.Pd_rbr_rocker)
        f19 = G(self.Pd_rbl_rocker)
        f20 = self.R_rbr_upper_strut
        f21 = self.R_rbr_lower_strut
        f22 = self.ubar_rbr_upper_strut_far_strut
        f23 = self.P_rbr_upper_strut
        f24 = A(f23)
        f25 = self.ubar_rbr_lower_strut_far_strut
        f26 = self.P_rbr_lower_strut
        f27 = A(f26)
        f28 = (f20.T + (-1) * f21.T + multi_dot([f22.T,f24.T]) + (-1) * multi_dot([f25.T,f27.T]))
        f29 = multi_dot([f24,f22])
        f30 = multi_dot([f27,f25])
        f31 = (f20 + (-1) * f21 + f29 + (-1) * f30)
        f32 = ((multi_dot([f28,f31]))**(1.0/2.0))[0]
        f33 = 1.0/f32
        f34 = config.UF_far_strut_Fs((config.far_strut_FL + (-1 * f32)))
        f35 = self.Pd_rbr_upper_strut
        f36 = self.Pd_rbr_lower_strut
        f37 = config.UF_far_strut_Fd((-1 * 1.0/f32) * multi_dot([f28,(self.Rd_rbr_upper_strut + (-1) * self.Rd_rbr_lower_strut + multi_dot([B(f23,f22),f35]) + (-1) * multi_dot([B(f26,f25),f36]))]))
        f38 = (f33 * (f34 + f37)) * f31
        f39 = G(f35)
        f40 = (2 * f34)
        f41 = (2 * f37)
        f42 = self.R_rbl_upper_strut
        f43 = self.R_rbl_lower_strut
        f44 = self.ubar_rbl_upper_strut_fal_strut
        f45 = self.P_rbl_upper_strut
        f46 = A(f45)
        f47 = self.ubar_rbl_lower_strut_fal_strut
        f48 = self.P_rbl_lower_strut
        f49 = A(f48)
        f50 = (f42.T + (-1) * f43.T + multi_dot([f44.T,f46.T]) + (-1) * multi_dot([f47.T,f49.T]))
        f51 = multi_dot([f46,f44])
        f52 = multi_dot([f49,f47])
        f53 = (f42 + (-1) * f43 + f51 + (-1) * f52)
        f54 = ((multi_dot([f50,f53]))**(1.0/2.0))[0]
        f55 = 1.0/f54
        f56 = config.UF_fal_strut_Fs((config.fal_strut_FL + (-1 * f54)))
        f57 = self.Pd_rbl_upper_strut
        f58 = self.Pd_rbl_lower_strut
        f59 = config.UF_fal_strut_Fd((-1 * 1.0/f54) * multi_dot([f50,(self.Rd_rbl_upper_strut + (-1) * self.Rd_rbl_lower_strut + multi_dot([B(f45,f44),f57]) + (-1) * multi_dot([B(f48,f47),f58]))]))
        f60 = (f55 * (f56 + f59)) * f53
        f61 = G(f57)
        f62 = (2 * f56)
        f63 = (2 * f59)
        f64 = np.zeros((4,1),dtype=np.float64)
        f65 = G(f36)
        f66 = G(f58)

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
        self.F_rbr_tie_rod_gravity,
        (8) * multi_dot([f6.T,config.Jbar_rbr_tie_rod,f6,self.P_rbr_tie_rod]),
        self.F_rbl_tie_rod_gravity,
        (8) * multi_dot([f7.T,config.Jbar_rbl_tie_rod,f7,self.P_rbl_tie_rod]),
        (self.F_rbr_hub_gravity + f8 + f10),
        ((2 * config.UF_far_drive(t)) * multi_dot([G(f11).T,self.vbar_rbr_hub_far_drive]) + (8) * multi_dot([f12.T,config.Jbar_rbr_hub,f12,f11]) + (2) * multi_dot([E(f11).T,(config.UF_far_tire_T(f9) + multi_dot([skew(multi_dot([A(f11),self.ubar_rbr_hub_far_tire])),f10]))])),
        (self.F_rbl_hub_gravity + f8 + f13),
        ((2 * config.UF_fal_drive(t)) * multi_dot([G(f14).T,self.vbar_rbl_hub_fal_drive]) + (8) * multi_dot([f15.T,config.Jbar_rbl_hub,f15,f14]) + (2) * multi_dot([E(f14).T,(config.UF_fal_tire_T(f9) + multi_dot([skew(multi_dot([A(f14),self.ubar_rbl_hub_fal_tire])),f13]))])),
        self.F_rbr_pushrod_gravity,
        (8) * multi_dot([f16.T,config.Jbar_rbr_pushrod,f16,self.P_rbr_pushrod]),
        self.F_rbl_pushrod_gravity,
        (8) * multi_dot([f17.T,config.Jbar_rbl_pushrod,f17,self.P_rbl_pushrod]),
        self.F_rbr_rocker_gravity,
        (8) * multi_dot([f18.T,config.Jbar_rbr_rocker,f18,self.P_rbr_rocker]),
        self.F_rbl_rocker_gravity,
        (8) * multi_dot([f19.T,config.Jbar_rbl_rocker,f19,self.P_rbl_rocker]),
        (self.F_rbr_upper_strut_gravity + f38),
        ((8) * multi_dot([f39.T,config.Jbar_rbr_upper_strut,f39,f23]) + (f33 * ((-1 * f40) + (-1 * f41))) * multi_dot([E(f23).T,skew(f29).T,f31])),
        (self.F_rbl_upper_strut_gravity + f60),
        ((8) * multi_dot([f61.T,config.Jbar_rbl_upper_strut,f61,f45]) + (f55 * ((-1 * f62) + (-1 * f63))) * multi_dot([E(f45).T,skew(f51).T,f53])),
        (self.F_rbr_lower_strut_gravity + f8 + (-1) * f38),
        (f64 + (8) * multi_dot([f65.T,config.Jbar_rbr_lower_strut,f65,f26]) + (f33 * (f40 + f41)) * multi_dot([E(f26).T,skew(f30).T,f31])),
        (self.F_rbl_lower_strut_gravity + f8 + (-1) * f60),
        (f64 + (8) * multi_dot([f66.T,config.Jbar_rbl_lower_strut,f66,f48]) + (f55 * (f62 + f63)) * multi_dot([E(f48).T,skew(f52).T,f53])),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbr_uca_jcr_uca_upright = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbr_uca,self.ubar_rbr_uca_jcr_uca_upright).T]]),self.L_jcr_uca_upright])
        self.F_rbr_uca_jcr_uca_upright = Q_rbr_uca_jcr_uca_upright[0:3,0:1]
        Te_rbr_uca_jcr_uca_upright = Q_rbr_uca_jcr_uca_upright[3:7,0:1]
        self.T_rbr_uca_jcr_uca_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_uca),self.ubar_rbr_uca_jcr_uca_upright])),self.F_rbr_uca_jcr_uca_upright]) + (0.5) * multi_dot([E(self.P_rbr_uca),Te_rbr_uca_jcr_uca_upright]))
        Q_rbr_uca_jcr_uca_chassis = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_uca,self.ubar_rbr_uca_jcr_uca_chassis).T,multi_dot([B(self.P_rbr_uca,self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]]),multi_dot([B(self.P_rbr_uca,self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]])]]),self.L_jcr_uca_chassis])
        self.F_rbr_uca_jcr_uca_chassis = Q_rbr_uca_jcr_uca_chassis[0:3,0:1]
        Te_rbr_uca_jcr_uca_chassis = Q_rbr_uca_jcr_uca_chassis[3:7,0:1]
        self.T_rbr_uca_jcr_uca_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_uca),self.ubar_rbr_uca_jcr_uca_chassis])),self.F_rbr_uca_jcr_uca_chassis]) + (0.5) * multi_dot([E(self.P_rbr_uca),Te_rbr_uca_jcr_uca_chassis]))
        Q_rbr_uca_jcr_prod_uca = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_uca,self.ubar_rbr_uca_jcr_prod_uca).T,multi_dot([B(self.P_rbr_uca,self.Mbar_rbr_uca_jcr_prod_uca[:,0:1]).T,A(self.P_rbr_pushrod),self.Mbar_rbr_pushrod_jcr_prod_uca[:,0:1]])]]),self.L_jcr_prod_uca])
        self.F_rbr_uca_jcr_prod_uca = Q_rbr_uca_jcr_prod_uca[0:3,0:1]
        Te_rbr_uca_jcr_prod_uca = Q_rbr_uca_jcr_prod_uca[3:7,0:1]
        self.T_rbr_uca_jcr_prod_uca = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_uca),self.ubar_rbr_uca_jcr_prod_uca])),self.F_rbr_uca_jcr_prod_uca]) + (0.5) * multi_dot([E(self.P_rbr_uca),Te_rbr_uca_jcr_prod_uca]))
        Q_rbl_uca_jcl_uca_upright = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbl_uca,self.ubar_rbl_uca_jcl_uca_upright).T]]),self.L_jcl_uca_upright])
        self.F_rbl_uca_jcl_uca_upright = Q_rbl_uca_jcl_uca_upright[0:3,0:1]
        Te_rbl_uca_jcl_uca_upright = Q_rbl_uca_jcl_uca_upright[3:7,0:1]
        self.T_rbl_uca_jcl_uca_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_uca),self.ubar_rbl_uca_jcl_uca_upright])),self.F_rbl_uca_jcl_uca_upright]) + (0.5) * multi_dot([E(self.P_rbl_uca),Te_rbl_uca_jcl_uca_upright]))
        Q_rbl_uca_jcl_uca_chassis = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_uca,self.ubar_rbl_uca_jcl_uca_chassis).T,multi_dot([B(self.P_rbl_uca,self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]]),multi_dot([B(self.P_rbl_uca,self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]])]]),self.L_jcl_uca_chassis])
        self.F_rbl_uca_jcl_uca_chassis = Q_rbl_uca_jcl_uca_chassis[0:3,0:1]
        Te_rbl_uca_jcl_uca_chassis = Q_rbl_uca_jcl_uca_chassis[3:7,0:1]
        self.T_rbl_uca_jcl_uca_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_uca),self.ubar_rbl_uca_jcl_uca_chassis])),self.F_rbl_uca_jcl_uca_chassis]) + (0.5) * multi_dot([E(self.P_rbl_uca),Te_rbl_uca_jcl_uca_chassis]))
        Q_rbl_uca_jcl_prod_uca = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_uca,self.ubar_rbl_uca_jcl_prod_uca).T,multi_dot([B(self.P_rbl_uca,self.Mbar_rbl_uca_jcl_prod_uca[:,0:1]).T,A(self.P_rbl_pushrod),self.Mbar_rbl_pushrod_jcl_prod_uca[:,0:1]])]]),self.L_jcl_prod_uca])
        self.F_rbl_uca_jcl_prod_uca = Q_rbl_uca_jcl_prod_uca[0:3,0:1]
        Te_rbl_uca_jcl_prod_uca = Q_rbl_uca_jcl_prod_uca[3:7,0:1]
        self.T_rbl_uca_jcl_prod_uca = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_uca),self.ubar_rbl_uca_jcl_prod_uca])),self.F_rbl_uca_jcl_prod_uca]) + (0.5) * multi_dot([E(self.P_rbl_uca),Te_rbl_uca_jcl_prod_uca]))
        Q_rbr_lca_jcr_lca_upright = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbr_lca,self.ubar_rbr_lca_jcr_lca_upright).T]]),self.L_jcr_lca_upright])
        self.F_rbr_lca_jcr_lca_upright = Q_rbr_lca_jcr_lca_upright[0:3,0:1]
        Te_rbr_lca_jcr_lca_upright = Q_rbr_lca_jcr_lca_upright[3:7,0:1]
        self.T_rbr_lca_jcr_lca_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_lca),self.ubar_rbr_lca_jcr_lca_upright])),self.F_rbr_lca_jcr_lca_upright]) + (0.5) * multi_dot([E(self.P_rbr_lca),Te_rbr_lca_jcr_lca_upright]))
        Q_rbr_lca_jcr_lca_chassis = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_lca,self.ubar_rbr_lca_jcr_lca_chassis).T,multi_dot([B(self.P_rbr_lca,self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]]),multi_dot([B(self.P_rbr_lca,self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]])]]),self.L_jcr_lca_chassis])
        self.F_rbr_lca_jcr_lca_chassis = Q_rbr_lca_jcr_lca_chassis[0:3,0:1]
        Te_rbr_lca_jcr_lca_chassis = Q_rbr_lca_jcr_lca_chassis[3:7,0:1]
        self.T_rbr_lca_jcr_lca_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_lca),self.ubar_rbr_lca_jcr_lca_chassis])),self.F_rbr_lca_jcr_lca_chassis]) + (0.5) * multi_dot([E(self.P_rbr_lca),Te_rbr_lca_jcr_lca_chassis]))
        Q_rbl_lca_jcl_lca_upright = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbl_lca,self.ubar_rbl_lca_jcl_lca_upright).T]]),self.L_jcl_lca_upright])
        self.F_rbl_lca_jcl_lca_upright = Q_rbl_lca_jcl_lca_upright[0:3,0:1]
        Te_rbl_lca_jcl_lca_upright = Q_rbl_lca_jcl_lca_upright[3:7,0:1]
        self.T_rbl_lca_jcl_lca_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_lca),self.ubar_rbl_lca_jcl_lca_upright])),self.F_rbl_lca_jcl_lca_upright]) + (0.5) * multi_dot([E(self.P_rbl_lca),Te_rbl_lca_jcl_lca_upright]))
        Q_rbl_lca_jcl_lca_chassis = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_lca,self.ubar_rbl_lca_jcl_lca_chassis).T,multi_dot([B(self.P_rbl_lca,self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]]),multi_dot([B(self.P_rbl_lca,self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]])]]),self.L_jcl_lca_chassis])
        self.F_rbl_lca_jcl_lca_chassis = Q_rbl_lca_jcl_lca_chassis[0:3,0:1]
        Te_rbl_lca_jcl_lca_chassis = Q_rbl_lca_jcl_lca_chassis[3:7,0:1]
        self.T_rbl_lca_jcl_lca_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_lca),self.ubar_rbl_lca_jcl_lca_chassis])),self.F_rbl_lca_jcl_lca_chassis]) + (0.5) * multi_dot([E(self.P_rbl_lca),Te_rbl_lca_jcl_lca_chassis]))
        Q_rbr_upright_jcr_hub_bearing = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_upright,self.ubar_rbr_upright_jcr_hub_bearing).T,multi_dot([B(self.P_rbr_upright,self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]).T,A(self.P_rbr_hub),self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]]),multi_dot([B(self.P_rbr_upright,self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]).T,A(self.P_rbr_hub),self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]])]]),self.L_jcr_hub_bearing])
        self.F_rbr_upright_jcr_hub_bearing = Q_rbr_upright_jcr_hub_bearing[0:3,0:1]
        Te_rbr_upright_jcr_hub_bearing = Q_rbr_upright_jcr_hub_bearing[3:7,0:1]
        self.T_rbr_upright_jcr_hub_bearing = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_upright),self.ubar_rbr_upright_jcr_hub_bearing])),self.F_rbr_upright_jcr_hub_bearing]) + (0.5) * multi_dot([E(self.P_rbr_upright),Te_rbr_upright_jcr_hub_bearing]))
        Q_rbl_upright_jcl_hub_bearing = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_upright,self.ubar_rbl_upright_jcl_hub_bearing).T,multi_dot([B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]).T,A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]]),multi_dot([B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]).T,A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]])]]),self.L_jcl_hub_bearing])
        self.F_rbl_upright_jcl_hub_bearing = Q_rbl_upright_jcl_hub_bearing[0:3,0:1]
        Te_rbl_upright_jcl_hub_bearing = Q_rbl_upright_jcl_hub_bearing[3:7,0:1]
        self.T_rbl_upright_jcl_hub_bearing = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_upright),self.ubar_rbl_upright_jcl_hub_bearing])),self.F_rbl_upright_jcl_hub_bearing]) + (0.5) * multi_dot([E(self.P_rbl_upright),Te_rbl_upright_jcl_hub_bearing]))
        Q_rbr_tie_rod_jcr_tie_upright = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbr_tie_rod,self.ubar_rbr_tie_rod_jcr_tie_upright).T]]),self.L_jcr_tie_upright])
        self.F_rbr_tie_rod_jcr_tie_upright = Q_rbr_tie_rod_jcr_tie_upright[0:3,0:1]
        Te_rbr_tie_rod_jcr_tie_upright = Q_rbr_tie_rod_jcr_tie_upright[3:7,0:1]
        self.T_rbr_tie_rod_jcr_tie_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_tie_rod),self.ubar_rbr_tie_rod_jcr_tie_upright])),self.F_rbr_tie_rod_jcr_tie_upright]) + (0.5) * multi_dot([E(self.P_rbr_tie_rod),Te_rbr_tie_rod_jcr_tie_upright]))
        Q_rbr_tie_rod_jcr_tie_steering = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_tie_rod,self.ubar_rbr_tie_rod_jcr_tie_steering).T,multi_dot([B(self.P_rbr_tie_rod,self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]).T,A(self.P_vbr_steer),self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]])]]),self.L_jcr_tie_steering])
        self.F_rbr_tie_rod_jcr_tie_steering = Q_rbr_tie_rod_jcr_tie_steering[0:3,0:1]
        Te_rbr_tie_rod_jcr_tie_steering = Q_rbr_tie_rod_jcr_tie_steering[3:7,0:1]
        self.T_rbr_tie_rod_jcr_tie_steering = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_tie_rod),self.ubar_rbr_tie_rod_jcr_tie_steering])),self.F_rbr_tie_rod_jcr_tie_steering]) + (0.5) * multi_dot([E(self.P_rbr_tie_rod),Te_rbr_tie_rod_jcr_tie_steering]))
        Q_rbl_tie_rod_jcl_tie_upright = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbl_tie_rod,self.ubar_rbl_tie_rod_jcl_tie_upright).T]]),self.L_jcl_tie_upright])
        self.F_rbl_tie_rod_jcl_tie_upright = Q_rbl_tie_rod_jcl_tie_upright[0:3,0:1]
        Te_rbl_tie_rod_jcl_tie_upright = Q_rbl_tie_rod_jcl_tie_upright[3:7,0:1]
        self.T_rbl_tie_rod_jcl_tie_upright = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_tie_rod),self.ubar_rbl_tie_rod_jcl_tie_upright])),self.F_rbl_tie_rod_jcl_tie_upright]) + (0.5) * multi_dot([E(self.P_rbl_tie_rod),Te_rbl_tie_rod_jcl_tie_upright]))
        Q_rbl_tie_rod_jcl_tie_steering = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_tie_rod,self.ubar_rbl_tie_rod_jcl_tie_steering).T,multi_dot([B(self.P_rbl_tie_rod,self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]).T,A(self.P_vbl_steer),self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]])]]),self.L_jcl_tie_steering])
        self.F_rbl_tie_rod_jcl_tie_steering = Q_rbl_tie_rod_jcl_tie_steering[0:3,0:1]
        Te_rbl_tie_rod_jcl_tie_steering = Q_rbl_tie_rod_jcl_tie_steering[3:7,0:1]
        self.T_rbl_tie_rod_jcl_tie_steering = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_tie_rod),self.ubar_rbl_tie_rod_jcl_tie_steering])),self.F_rbl_tie_rod_jcl_tie_steering]) + (0.5) * multi_dot([E(self.P_rbl_tie_rod),Te_rbl_tie_rod_jcl_tie_steering]))
        Q_rbr_rocker_jcr_prod_rocker = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbr_rocker,self.ubar_rbr_rocker_jcr_prod_rocker).T]]),self.L_jcr_prod_rocker])
        self.F_rbr_rocker_jcr_prod_rocker = Q_rbr_rocker_jcr_prod_rocker[0:3,0:1]
        Te_rbr_rocker_jcr_prod_rocker = Q_rbr_rocker_jcr_prod_rocker[3:7,0:1]
        self.T_rbr_rocker_jcr_prod_rocker = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_rocker),self.ubar_rbr_rocker_jcr_prod_rocker])),self.F_rbr_rocker_jcr_prod_rocker]) + (0.5) * multi_dot([E(self.P_rbr_rocker),Te_rbr_rocker_jcr_prod_rocker]))
        Q_rbr_rocker_jcr_rocker_chassis = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_rocker,self.ubar_rbr_rocker_jcr_rocker_chassis).T,multi_dot([B(self.P_rbr_rocker,self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]]),multi_dot([B(self.P_rbr_rocker,self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]])]]),self.L_jcr_rocker_chassis])
        self.F_rbr_rocker_jcr_rocker_chassis = Q_rbr_rocker_jcr_rocker_chassis[0:3,0:1]
        Te_rbr_rocker_jcr_rocker_chassis = Q_rbr_rocker_jcr_rocker_chassis[3:7,0:1]
        self.T_rbr_rocker_jcr_rocker_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_rocker),self.ubar_rbr_rocker_jcr_rocker_chassis])),self.F_rbr_rocker_jcr_rocker_chassis]) + (0.5) * multi_dot([E(self.P_rbr_rocker),Te_rbr_rocker_jcr_rocker_chassis]))
        Q_rbl_rocker_jcl_prod_rocker = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbl_rocker,self.ubar_rbl_rocker_jcl_prod_rocker).T]]),self.L_jcl_prod_rocker])
        self.F_rbl_rocker_jcl_prod_rocker = Q_rbl_rocker_jcl_prod_rocker[0:3,0:1]
        Te_rbl_rocker_jcl_prod_rocker = Q_rbl_rocker_jcl_prod_rocker[3:7,0:1]
        self.T_rbl_rocker_jcl_prod_rocker = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_rocker),self.ubar_rbl_rocker_jcl_prod_rocker])),self.F_rbl_rocker_jcl_prod_rocker]) + (0.5) * multi_dot([E(self.P_rbl_rocker),Te_rbl_rocker_jcl_prod_rocker]))
        Q_rbl_rocker_jcl_rocker_chassis = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_rocker,self.ubar_rbl_rocker_jcl_rocker_chassis).T,multi_dot([B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]]),multi_dot([B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]])]]),self.L_jcl_rocker_chassis])
        self.F_rbl_rocker_jcl_rocker_chassis = Q_rbl_rocker_jcl_rocker_chassis[0:3,0:1]
        Te_rbl_rocker_jcl_rocker_chassis = Q_rbl_rocker_jcl_rocker_chassis[3:7,0:1]
        self.T_rbl_rocker_jcl_rocker_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_rocker),self.ubar_rbl_rocker_jcl_rocker_chassis])),self.F_rbl_rocker_jcl_rocker_chassis]) + (0.5) * multi_dot([E(self.P_rbl_rocker),Te_rbl_rocker_jcl_rocker_chassis]))
        Q_rbr_upper_strut_jcr_strut_chassis = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut_chassis).T,multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]])]]),self.L_jcr_strut_chassis])
        self.F_rbr_upper_strut_jcr_strut_chassis = Q_rbr_upper_strut_jcr_strut_chassis[0:3,0:1]
        Te_rbr_upper_strut_jcr_strut_chassis = Q_rbr_upper_strut_jcr_strut_chassis[3:7,0:1]
        self.T_rbr_upper_strut_jcr_strut_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut_chassis])),self.F_rbr_upper_strut_jcr_strut_chassis]) + (0.5) * multi_dot([E(self.P_rbr_upper_strut),Te_rbr_upper_strut_jcr_strut_chassis]))
        Q_rbr_upper_strut_jcr_strut = (-1) * multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]]),multi_dot([A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]])],[multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]).T,A(self.P_rbr_lower_strut),self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]]),multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]).T,A(self.P_rbr_lower_strut),self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]]),(multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]).T,((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut]) + self.R_rbr_upper_strut)]) + multi_dot([B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut).T,A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]])),(multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]).T,((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut]) + self.R_rbr_upper_strut)]) + multi_dot([B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut).T,A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]]))]]),self.L_jcr_strut])
        self.F_rbr_upper_strut_jcr_strut = Q_rbr_upper_strut_jcr_strut[0:3,0:1]
        Te_rbr_upper_strut_jcr_strut = Q_rbr_upper_strut_jcr_strut[3:7,0:1]
        self.T_rbr_upper_strut_jcr_strut = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut])),self.F_rbr_upper_strut_jcr_strut]) + (0.5) * multi_dot([E(self.P_rbr_upper_strut),Te_rbr_upper_strut_jcr_strut]))
        self.F_rbr_upper_strut_far_strut = (1.0/((multi_dot([((-1) * self.R_rbr_lower_strut.T + multi_dot([self.ubar_rbr_upper_strut_far_strut.T,A(self.P_rbr_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbr_lower_strut_far_strut.T,A(self.P_rbr_lower_strut).T]) + self.R_rbr_upper_strut.T),((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_far_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_far_strut]) + self.R_rbr_upper_strut)]))**(1.0/2.0))[0] * (config.UF_far_strut_Fd((-1 * 1.0/((multi_dot([((-1) * self.R_rbr_lower_strut.T + multi_dot([self.ubar_rbr_upper_strut_far_strut.T,A(self.P_rbr_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbr_lower_strut_far_strut.T,A(self.P_rbr_lower_strut).T]) + self.R_rbr_upper_strut.T),((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_far_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_far_strut]) + self.R_rbr_upper_strut)]))**(1.0/2.0))[0]) * multi_dot([((-1) * self.R_rbr_lower_strut.T + multi_dot([self.ubar_rbr_upper_strut_far_strut.T,A(self.P_rbr_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbr_lower_strut_far_strut.T,A(self.P_rbr_lower_strut).T]) + self.R_rbr_upper_strut.T),((-1) * self.Rd_rbr_lower_strut + multi_dot([B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_far_strut),self.Pd_rbr_upper_strut]) + (-1) * multi_dot([B(self.P_rbr_lower_strut,self.ubar_rbr_lower_strut_far_strut),self.Pd_rbr_lower_strut]) + self.Rd_rbr_upper_strut)])) + config.UF_far_strut_Fs((config.far_strut_FL + (-1 * ((multi_dot([((-1) * self.R_rbr_lower_strut.T + multi_dot([self.ubar_rbr_upper_strut_far_strut.T,A(self.P_rbr_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbr_lower_strut_far_strut.T,A(self.P_rbr_lower_strut).T]) + self.R_rbr_upper_strut.T),((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_far_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_far_strut]) + self.R_rbr_upper_strut)]))**(1.0/2.0))[0]))))) * ((-1) * self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_far_strut]) + (-1) * multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_far_strut]) + self.R_rbr_upper_strut)
        self.T_rbr_upper_strut_far_strut = np.zeros((3,1),dtype=np.float64)
        Q_rbl_upper_strut_jcl_strut_chassis = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut_chassis).T,multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]])]]),self.L_jcl_strut_chassis])
        self.F_rbl_upper_strut_jcl_strut_chassis = Q_rbl_upper_strut_jcl_strut_chassis[0:3,0:1]
        Te_rbl_upper_strut_jcl_strut_chassis = Q_rbl_upper_strut_jcl_strut_chassis[3:7,0:1]
        self.T_rbl_upper_strut_jcl_strut_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut_chassis])),self.F_rbl_upper_strut_jcl_strut_chassis]) + (0.5) * multi_dot([E(self.P_rbl_upper_strut),Te_rbl_upper_strut_jcl_strut_chassis]))
        Q_rbl_upper_strut_jcl_strut = (-1) * multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]]),multi_dot([A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]])],[multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]).T,A(self.P_rbl_lower_strut),self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]]),multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]).T,A(self.P_rbl_lower_strut),self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]]),(multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]).T,((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut]) + self.R_rbl_upper_strut)]) + multi_dot([B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut).T,A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]])),(multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]).T,((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut]) + self.R_rbl_upper_strut)]) + multi_dot([B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut).T,A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]]))]]),self.L_jcl_strut])
        self.F_rbl_upper_strut_jcl_strut = Q_rbl_upper_strut_jcl_strut[0:3,0:1]
        Te_rbl_upper_strut_jcl_strut = Q_rbl_upper_strut_jcl_strut[3:7,0:1]
        self.T_rbl_upper_strut_jcl_strut = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut])),self.F_rbl_upper_strut_jcl_strut]) + (0.5) * multi_dot([E(self.P_rbl_upper_strut),Te_rbl_upper_strut_jcl_strut]))
        self.F_rbl_upper_strut_fal_strut = (1.0/((multi_dot([((-1) * self.R_rbl_lower_strut.T + multi_dot([self.ubar_rbl_upper_strut_fal_strut.T,A(self.P_rbl_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbl_lower_strut_fal_strut.T,A(self.P_rbl_lower_strut).T]) + self.R_rbl_upper_strut.T),((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_fal_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_fal_strut]) + self.R_rbl_upper_strut)]))**(1.0/2.0))[0] * (config.UF_fal_strut_Fd((-1 * 1.0/((multi_dot([((-1) * self.R_rbl_lower_strut.T + multi_dot([self.ubar_rbl_upper_strut_fal_strut.T,A(self.P_rbl_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbl_lower_strut_fal_strut.T,A(self.P_rbl_lower_strut).T]) + self.R_rbl_upper_strut.T),((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_fal_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_fal_strut]) + self.R_rbl_upper_strut)]))**(1.0/2.0))[0]) * multi_dot([((-1) * self.R_rbl_lower_strut.T + multi_dot([self.ubar_rbl_upper_strut_fal_strut.T,A(self.P_rbl_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbl_lower_strut_fal_strut.T,A(self.P_rbl_lower_strut).T]) + self.R_rbl_upper_strut.T),((-1) * self.Rd_rbl_lower_strut + multi_dot([B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_fal_strut),self.Pd_rbl_upper_strut]) + (-1) * multi_dot([B(self.P_rbl_lower_strut,self.ubar_rbl_lower_strut_fal_strut),self.Pd_rbl_lower_strut]) + self.Rd_rbl_upper_strut)])) + config.UF_fal_strut_Fs((config.fal_strut_FL + (-1 * ((multi_dot([((-1) * self.R_rbl_lower_strut.T + multi_dot([self.ubar_rbl_upper_strut_fal_strut.T,A(self.P_rbl_upper_strut).T]) + (-1) * multi_dot([self.ubar_rbl_lower_strut_fal_strut.T,A(self.P_rbl_lower_strut).T]) + self.R_rbl_upper_strut.T),((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_fal_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_fal_strut]) + self.R_rbl_upper_strut)]))**(1.0/2.0))[0]))))) * ((-1) * self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_fal_strut]) + (-1) * multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_fal_strut]) + self.R_rbl_upper_strut)
        self.T_rbl_upper_strut_fal_strut = np.zeros((3,1),dtype=np.float64)
        Q_rbr_lower_strut_jcr_strut_rocker = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_lower_strut,self.ubar_rbr_lower_strut_jcr_strut_rocker).T,multi_dot([B(self.P_rbr_lower_strut,self.Mbar_rbr_lower_strut_jcr_strut_rocker[:,0:1]).T,A(self.P_rbr_rocker),self.Mbar_rbr_rocker_jcr_strut_rocker[:,0:1]])]]),self.L_jcr_strut_rocker])
        self.F_rbr_lower_strut_jcr_strut_rocker = Q_rbr_lower_strut_jcr_strut_rocker[0:3,0:1]
        Te_rbr_lower_strut_jcr_strut_rocker = Q_rbr_lower_strut_jcr_strut_rocker[3:7,0:1]
        self.T_rbr_lower_strut_jcr_strut_rocker = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut_rocker])),self.F_rbr_lower_strut_jcr_strut_rocker]) + (0.5) * multi_dot([E(self.P_rbr_lower_strut),Te_rbr_lower_strut_jcr_strut_rocker]))
        Q_rbl_lower_strut_jcl_strut_rocker = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_lower_strut,self.ubar_rbl_lower_strut_jcl_strut_rocker).T,multi_dot([B(self.P_rbl_lower_strut,self.Mbar_rbl_lower_strut_jcl_strut_rocker[:,0:1]).T,A(self.P_rbl_rocker),self.Mbar_rbl_rocker_jcl_strut_rocker[:,0:1]])]]),self.L_jcl_strut_rocker])
        self.F_rbl_lower_strut_jcl_strut_rocker = Q_rbl_lower_strut_jcl_strut_rocker[0:3,0:1]
        Te_rbl_lower_strut_jcl_strut_rocker = Q_rbl_lower_strut_jcl_strut_rocker[3:7,0:1]
        self.T_rbl_lower_strut_jcl_strut_rocker = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut_rocker])),self.F_rbl_lower_strut_jcl_strut_rocker]) + (0.5) * multi_dot([E(self.P_rbl_lower_strut),Te_rbl_lower_strut_jcl_strut_rocker]))

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
                        'F_rbr_tie_rod_jcr_tie_upright' : self.F_rbr_tie_rod_jcr_tie_upright,
                        'T_rbr_tie_rod_jcr_tie_upright' : self.T_rbr_tie_rod_jcr_tie_upright,
                        'F_rbr_tie_rod_jcr_tie_steering' : self.F_rbr_tie_rod_jcr_tie_steering,
                        'T_rbr_tie_rod_jcr_tie_steering' : self.T_rbr_tie_rod_jcr_tie_steering,
                        'F_rbl_tie_rod_jcl_tie_upright' : self.F_rbl_tie_rod_jcl_tie_upright,
                        'T_rbl_tie_rod_jcl_tie_upright' : self.T_rbl_tie_rod_jcl_tie_upright,
                        'F_rbl_tie_rod_jcl_tie_steering' : self.F_rbl_tie_rod_jcl_tie_steering,
                        'T_rbl_tie_rod_jcl_tie_steering' : self.T_rbl_tie_rod_jcl_tie_steering,
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
                        'F_rbr_upper_strut_far_strut' : self.F_rbr_upper_strut_far_strut,
                        'T_rbr_upper_strut_far_strut' : self.T_rbr_upper_strut_far_strut,
                        'F_rbl_upper_strut_jcl_strut_chassis' : self.F_rbl_upper_strut_jcl_strut_chassis,
                        'T_rbl_upper_strut_jcl_strut_chassis' : self.T_rbl_upper_strut_jcl_strut_chassis,
                        'F_rbl_upper_strut_jcl_strut' : self.F_rbl_upper_strut_jcl_strut,
                        'T_rbl_upper_strut_jcl_strut' : self.T_rbl_upper_strut_jcl_strut,
                        'F_rbl_upper_strut_fal_strut' : self.F_rbl_upper_strut_fal_strut,
                        'T_rbl_upper_strut_fal_strut' : self.T_rbl_upper_strut_fal_strut,
                        'F_rbr_lower_strut_jcr_strut_rocker' : self.F_rbr_lower_strut_jcr_strut_rocker,
                        'T_rbr_lower_strut_jcr_strut_rocker' : self.T_rbr_lower_strut_jcr_strut_rocker,
                        'F_rbl_lower_strut_jcl_strut_rocker' : self.F_rbl_lower_strut_jcl_strut_rocker,
                        'T_rbl_lower_strut_jcl_strut_rocker' : self.T_rbl_lower_strut_jcl_strut_rocker}

