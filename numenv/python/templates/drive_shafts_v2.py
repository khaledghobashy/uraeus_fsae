
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

        self.indicies_map = {'vbs_ground': 0, 'rbr_inner_shaft': 1, 'rbl_inner_shaft': 2, 'rbr_coupling_inner': 3, 'rbl_coupling_inner': 4, 'rbr_coupling_outer': 5, 'rbl_coupling_outer': 6, 'vbs_differential': 7, 'vbr_wheel_hub': 8, 'vbl_wheel_hub': 9}

        self.n  = 42
        self.nc = 42
        self.nrows = 30
        self.ncols = 2*6
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_rbr_inner_shaft_jcr_diff_joint', 'T_rbr_inner_shaft_jcr_diff_joint', 'F_rbr_inner_shaft_jcr_inner_cv', 'T_rbr_inner_shaft_jcr_inner_cv', 'F_rbl_inner_shaft_jcl_diff_joint', 'T_rbl_inner_shaft_jcl_diff_joint', 'F_rbl_inner_shaft_jcl_inner_cv', 'T_rbl_inner_shaft_jcl_inner_cv', 'F_rbr_coupling_inner_jcr_coupling_trans', 'T_rbr_coupling_inner_jcr_coupling_trans', 'F_rbl_coupling_inner_jcl_coupling_trans', 'T_rbl_coupling_inner_jcl_coupling_trans', 'F_rbr_coupling_outer_jcr_outer_cv', 'T_rbr_coupling_outer_jcr_outer_cv', 'F_rbl_coupling_outer_jcl_outer_cv', 'T_rbl_coupling_outer_jcl_outer_cv']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29])
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.rbr_inner_shaft*2, self.rbr_inner_shaft*2+1, self.vbs_differential*2, self.vbs_differential*2+1, self.rbr_inner_shaft*2, self.rbr_inner_shaft*2+1, self.vbs_differential*2, self.vbs_differential*2+1, self.rbr_inner_shaft*2, self.rbr_inner_shaft*2+1, self.vbs_differential*2, self.vbs_differential*2+1, self.rbr_inner_shaft*2, self.rbr_inner_shaft*2+1, self.rbr_coupling_inner*2, self.rbr_coupling_inner*2+1, self.rbr_inner_shaft*2, self.rbr_inner_shaft*2+1, self.rbr_coupling_inner*2, self.rbr_coupling_inner*2+1, self.rbl_inner_shaft*2, self.rbl_inner_shaft*2+1, self.vbs_differential*2, self.vbs_differential*2+1, self.rbl_inner_shaft*2, self.rbl_inner_shaft*2+1, self.vbs_differential*2, self.vbs_differential*2+1, self.rbl_inner_shaft*2, self.rbl_inner_shaft*2+1, self.vbs_differential*2, self.vbs_differential*2+1, self.rbl_inner_shaft*2, self.rbl_inner_shaft*2+1, self.rbl_coupling_inner*2, self.rbl_coupling_inner*2+1, self.rbl_inner_shaft*2, self.rbl_inner_shaft*2+1, self.rbl_coupling_inner*2, self.rbl_coupling_inner*2+1, self.rbr_coupling_inner*2, self.rbr_coupling_inner*2+1, self.rbr_coupling_outer*2, self.rbr_coupling_outer*2+1, self.rbr_coupling_inner*2, self.rbr_coupling_inner*2+1, self.rbr_coupling_outer*2, self.rbr_coupling_outer*2+1, self.rbr_coupling_inner*2, self.rbr_coupling_inner*2+1, self.rbr_coupling_outer*2, self.rbr_coupling_outer*2+1, self.rbr_coupling_inner*2, self.rbr_coupling_inner*2+1, self.rbr_coupling_outer*2, self.rbr_coupling_outer*2+1, self.rbr_coupling_inner*2, self.rbr_coupling_inner*2+1, self.rbr_coupling_outer*2, self.rbr_coupling_outer*2+1, self.rbl_coupling_inner*2, self.rbl_coupling_inner*2+1, self.rbl_coupling_outer*2, self.rbl_coupling_outer*2+1, self.rbl_coupling_inner*2, self.rbl_coupling_inner*2+1, self.rbl_coupling_outer*2, self.rbl_coupling_outer*2+1, self.rbl_coupling_inner*2, self.rbl_coupling_inner*2+1, self.rbl_coupling_outer*2, self.rbl_coupling_outer*2+1, self.rbl_coupling_inner*2, self.rbl_coupling_inner*2+1, self.rbl_coupling_outer*2, self.rbl_coupling_outer*2+1, self.rbl_coupling_inner*2, self.rbl_coupling_inner*2+1, self.rbl_coupling_outer*2, self.rbl_coupling_outer*2+1, self.rbr_coupling_outer*2, self.rbr_coupling_outer*2+1, self.vbr_wheel_hub*2, self.vbr_wheel_hub*2+1, self.rbr_coupling_outer*2, self.rbr_coupling_outer*2+1, self.vbr_wheel_hub*2, self.vbr_wheel_hub*2+1, self.rbl_coupling_outer*2, self.rbl_coupling_outer*2+1, self.vbl_wheel_hub*2, self.vbl_wheel_hub*2+1, self.rbl_coupling_outer*2, self.rbl_coupling_outer*2+1, self.vbl_wheel_hub*2, self.vbl_wheel_hub*2+1, self.rbr_inner_shaft*2, self.rbr_inner_shaft*2+1, self.rbl_inner_shaft*2, self.rbl_inner_shaft*2+1, self.rbr_coupling_inner*2, self.rbr_coupling_inner*2+1, self.rbl_coupling_inner*2, self.rbl_coupling_inner*2+1, self.rbr_coupling_outer*2, self.rbr_coupling_outer*2+1, self.rbl_coupling_outer*2, self.rbl_coupling_outer*2+1])

    def set_initial_states(self):
        self.q0  = np.concatenate([self.config.R_rbr_inner_shaft,
        self.config.P_rbr_inner_shaft,
        self.config.R_rbl_inner_shaft,
        self.config.P_rbl_inner_shaft,
        self.config.R_rbr_coupling_inner,
        self.config.P_rbr_coupling_inner,
        self.config.R_rbl_coupling_inner,
        self.config.P_rbl_coupling_inner,
        self.config.R_rbr_coupling_outer,
        self.config.P_rbr_coupling_outer,
        self.config.R_rbl_coupling_outer,
        self.config.P_rbl_coupling_outer])
        self.qd0 = np.concatenate([self.config.Rd_rbr_inner_shaft,
        self.config.Pd_rbr_inner_shaft,
        self.config.Rd_rbl_inner_shaft,
        self.config.Pd_rbl_inner_shaft,
        self.config.Rd_rbr_coupling_inner,
        self.config.Pd_rbr_coupling_inner,
        self.config.Rd_rbl_coupling_inner,
        self.config.Pd_rbl_coupling_inner,
        self.config.Rd_rbr_coupling_outer,
        self.config.Pd_rbr_coupling_outer,
        self.config.Rd_rbl_coupling_outer,
        self.config.Pd_rbl_coupling_outer])

        self.set_gen_coordinates(self.q0)
        self.set_gen_velocities(self.qd0)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbr_inner_shaft = indicies_map[p + 'rbr_inner_shaft']
        self.rbl_inner_shaft = indicies_map[p + 'rbl_inner_shaft']
        self.rbr_coupling_inner = indicies_map[p + 'rbr_coupling_inner']
        self.rbl_coupling_inner = indicies_map[p + 'rbl_coupling_inner']
        self.rbr_coupling_outer = indicies_map[p + 'rbr_coupling_outer']
        self.rbl_coupling_outer = indicies_map[p + 'rbl_coupling_outer']
        self.vbs_differential = indicies_map[interface_map[p + 'vbs_differential']]
        self.vbs_ground = indicies_map[interface_map[p + 'vbs_ground']]
        self.vbr_wheel_hub = indicies_map[interface_map[p + 'vbr_wheel_hub']]
        self.vbl_wheel_hub = indicies_map[interface_map[p + 'vbl_wheel_hub']]

    
    def eval_constants(self):
        config = self.config

        self.F_rbr_inner_shaft_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_inner_shaft]], dtype=np.float64)
        self.F_rbr_inner_shaft_far_drive = np.zeros((3,1),dtype=np.float64)
        self.F_rbl_inner_shaft_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_inner_shaft]], dtype=np.float64)
        self.F_rbl_inner_shaft_fal_drive = np.zeros((3,1),dtype=np.float64)
        self.F_rbr_coupling_inner_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_coupling_inner]], dtype=np.float64)
        self.F_rbl_coupling_inner_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_coupling_inner]], dtype=np.float64)
        self.F_rbr_coupling_outer_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_coupling_outer]], dtype=np.float64)
        self.F_rbl_coupling_outer_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_coupling_outer]], dtype=np.float64)

        self.vbar_rbr_inner_shaft_far_drive = multi_dot([A(config.P_rbr_inner_shaft).T,config.ax1_far_drive,(multi_dot([config.ax1_far_drive.T,A(config.P_rbr_inner_shaft),A(config.P_rbr_inner_shaft).T,config.ax1_far_drive]))**(-1.0/2.0)])
        self.Mbar_rbr_inner_shaft_far_drive = multi_dot([A(config.P_rbr_inner_shaft).T,triad(config.ax1_far_drive)])
        self.Mbar_vbs_ground_far_drive = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_far_drive)])
        self.Mbar_rbr_inner_shaft_jcr_diff_joint = multi_dot([A(config.P_rbr_inner_shaft).T,triad(config.ax1_jcr_diff_joint)])
        self.Mbar_vbs_differential_jcr_diff_joint = multi_dot([A(config.P_vbs_differential).T,triad(config.ax1_jcr_diff_joint)])
        self.ubar_rbr_inner_shaft_jcr_diff_joint = (multi_dot([A(config.P_rbr_inner_shaft).T,config.pt1_jcr_diff_joint]) + (-1) * multi_dot([A(config.P_rbr_inner_shaft).T,config.R_rbr_inner_shaft]))
        self.ubar_vbs_differential_jcr_diff_joint = (multi_dot([A(config.P_vbs_differential).T,config.pt1_jcr_diff_joint]) + (-1) * multi_dot([A(config.P_vbs_differential).T,config.R_vbs_differential]))
        self.Mbar_rbr_inner_shaft_jcr_inner_cv = multi_dot([A(config.P_rbr_inner_shaft).T,triad(config.ax1_jcr_inner_cv)])
        self.Mbar_rbr_coupling_inner_jcr_inner_cv = multi_dot([A(config.P_rbr_coupling_inner).T,triad(config.ax2_jcr_inner_cv,triad(config.ax1_jcr_inner_cv)[0:3,1:2])])
        self.ubar_rbr_inner_shaft_jcr_inner_cv = (multi_dot([A(config.P_rbr_inner_shaft).T,config.pt1_jcr_inner_cv]) + (-1) * multi_dot([A(config.P_rbr_inner_shaft).T,config.R_rbr_inner_shaft]))
        self.ubar_rbr_coupling_inner_jcr_inner_cv = (multi_dot([A(config.P_rbr_coupling_inner).T,config.pt1_jcr_inner_cv]) + (-1) * multi_dot([A(config.P_rbr_coupling_inner).T,config.R_rbr_coupling_inner]))
        self.vbar_rbl_inner_shaft_fal_drive = multi_dot([A(config.P_rbl_inner_shaft).T,config.ax1_fal_drive,(multi_dot([config.ax1_fal_drive.T,A(config.P_rbl_inner_shaft),A(config.P_rbl_inner_shaft).T,config.ax1_fal_drive]))**(-1.0/2.0)])
        self.Mbar_rbl_inner_shaft_fal_drive = multi_dot([A(config.P_rbl_inner_shaft).T,triad(config.ax1_fal_drive)])
        self.Mbar_vbs_ground_fal_drive = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_fal_drive)])
        self.Mbar_rbl_inner_shaft_jcl_diff_joint = multi_dot([A(config.P_rbl_inner_shaft).T,triad(config.ax1_jcl_diff_joint)])
        self.Mbar_vbs_differential_jcl_diff_joint = multi_dot([A(config.P_vbs_differential).T,triad(config.ax1_jcl_diff_joint)])
        self.ubar_rbl_inner_shaft_jcl_diff_joint = (multi_dot([A(config.P_rbl_inner_shaft).T,config.pt1_jcl_diff_joint]) + (-1) * multi_dot([A(config.P_rbl_inner_shaft).T,config.R_rbl_inner_shaft]))
        self.ubar_vbs_differential_jcl_diff_joint = (multi_dot([A(config.P_vbs_differential).T,config.pt1_jcl_diff_joint]) + (-1) * multi_dot([A(config.P_vbs_differential).T,config.R_vbs_differential]))
        self.Mbar_rbl_inner_shaft_jcl_inner_cv = multi_dot([A(config.P_rbl_inner_shaft).T,triad(config.ax1_jcl_inner_cv)])
        self.Mbar_rbl_coupling_inner_jcl_inner_cv = multi_dot([A(config.P_rbl_coupling_inner).T,triad(config.ax2_jcl_inner_cv,triad(config.ax1_jcl_inner_cv)[0:3,1:2])])
        self.ubar_rbl_inner_shaft_jcl_inner_cv = (multi_dot([A(config.P_rbl_inner_shaft).T,config.pt1_jcl_inner_cv]) + (-1) * multi_dot([A(config.P_rbl_inner_shaft).T,config.R_rbl_inner_shaft]))
        self.ubar_rbl_coupling_inner_jcl_inner_cv = (multi_dot([A(config.P_rbl_coupling_inner).T,config.pt1_jcl_inner_cv]) + (-1) * multi_dot([A(config.P_rbl_coupling_inner).T,config.R_rbl_coupling_inner]))
        self.Mbar_rbr_coupling_inner_jcr_coupling_trans = multi_dot([A(config.P_rbr_coupling_inner).T,triad(config.ax1_jcr_coupling_trans)])
        self.Mbar_rbr_coupling_outer_jcr_coupling_trans = multi_dot([A(config.P_rbr_coupling_outer).T,triad(config.ax1_jcr_coupling_trans)])
        self.ubar_rbr_coupling_inner_jcr_coupling_trans = (multi_dot([A(config.P_rbr_coupling_inner).T,config.pt1_jcr_coupling_trans]) + (-1) * multi_dot([A(config.P_rbr_coupling_inner).T,config.R_rbr_coupling_inner]))
        self.ubar_rbr_coupling_outer_jcr_coupling_trans = (multi_dot([A(config.P_rbr_coupling_outer).T,config.pt1_jcr_coupling_trans]) + (-1) * multi_dot([A(config.P_rbr_coupling_outer).T,config.R_rbr_coupling_outer]))
        self.Mbar_rbl_coupling_inner_jcl_coupling_trans = multi_dot([A(config.P_rbl_coupling_inner).T,triad(config.ax1_jcl_coupling_trans)])
        self.Mbar_rbl_coupling_outer_jcl_coupling_trans = multi_dot([A(config.P_rbl_coupling_outer).T,triad(config.ax1_jcl_coupling_trans)])
        self.ubar_rbl_coupling_inner_jcl_coupling_trans = (multi_dot([A(config.P_rbl_coupling_inner).T,config.pt1_jcl_coupling_trans]) + (-1) * multi_dot([A(config.P_rbl_coupling_inner).T,config.R_rbl_coupling_inner]))
        self.ubar_rbl_coupling_outer_jcl_coupling_trans = (multi_dot([A(config.P_rbl_coupling_outer).T,config.pt1_jcl_coupling_trans]) + (-1) * multi_dot([A(config.P_rbl_coupling_outer).T,config.R_rbl_coupling_outer]))
        self.Mbar_rbr_coupling_outer_jcr_outer_cv = multi_dot([A(config.P_rbr_coupling_outer).T,triad(config.ax1_jcr_outer_cv)])
        self.Mbar_vbr_wheel_hub_jcr_outer_cv = multi_dot([A(config.P_vbr_wheel_hub).T,triad(config.ax2_jcr_outer_cv,triad(config.ax1_jcr_outer_cv)[0:3,1:2])])
        self.ubar_rbr_coupling_outer_jcr_outer_cv = (multi_dot([A(config.P_rbr_coupling_outer).T,config.pt1_jcr_outer_cv]) + (-1) * multi_dot([A(config.P_rbr_coupling_outer).T,config.R_rbr_coupling_outer]))
        self.ubar_vbr_wheel_hub_jcr_outer_cv = (multi_dot([A(config.P_vbr_wheel_hub).T,config.pt1_jcr_outer_cv]) + (-1) * multi_dot([A(config.P_vbr_wheel_hub).T,config.R_vbr_wheel_hub]))
        self.Mbar_rbl_coupling_outer_jcl_outer_cv = multi_dot([A(config.P_rbl_coupling_outer).T,triad(config.ax1_jcl_outer_cv)])
        self.Mbar_vbl_wheel_hub_jcl_outer_cv = multi_dot([A(config.P_vbl_wheel_hub).T,triad(config.ax2_jcl_outer_cv,triad(config.ax1_jcl_outer_cv)[0:3,1:2])])
        self.ubar_rbl_coupling_outer_jcl_outer_cv = (multi_dot([A(config.P_rbl_coupling_outer).T,config.pt1_jcl_outer_cv]) + (-1) * multi_dot([A(config.P_rbl_coupling_outer).T,config.R_rbl_coupling_outer]))
        self.ubar_vbl_wheel_hub_jcl_outer_cv = (multi_dot([A(config.P_vbl_wheel_hub).T,config.pt1_jcl_outer_cv]) + (-1) * multi_dot([A(config.P_vbl_wheel_hub).T,config.R_vbl_wheel_hub]))

    
    def set_gen_coordinates(self,q):
        self.R_rbr_inner_shaft = q[0:3,0:1]
        self.P_rbr_inner_shaft = q[3:7,0:1]
        self.R_rbl_inner_shaft = q[7:10,0:1]
        self.P_rbl_inner_shaft = q[10:14,0:1]
        self.R_rbr_coupling_inner = q[14:17,0:1]
        self.P_rbr_coupling_inner = q[17:21,0:1]
        self.R_rbl_coupling_inner = q[21:24,0:1]
        self.P_rbl_coupling_inner = q[24:28,0:1]
        self.R_rbr_coupling_outer = q[28:31,0:1]
        self.P_rbr_coupling_outer = q[31:35,0:1]
        self.R_rbl_coupling_outer = q[35:38,0:1]
        self.P_rbl_coupling_outer = q[38:42,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbr_inner_shaft = qd[0:3,0:1]
        self.Pd_rbr_inner_shaft = qd[3:7,0:1]
        self.Rd_rbl_inner_shaft = qd[7:10,0:1]
        self.Pd_rbl_inner_shaft = qd[10:14,0:1]
        self.Rd_rbr_coupling_inner = qd[14:17,0:1]
        self.Pd_rbr_coupling_inner = qd[17:21,0:1]
        self.Rd_rbl_coupling_inner = qd[21:24,0:1]
        self.Pd_rbl_coupling_inner = qd[24:28,0:1]
        self.Rd_rbr_coupling_outer = qd[28:31,0:1]
        self.Pd_rbr_coupling_outer = qd[31:35,0:1]
        self.Rd_rbl_coupling_outer = qd[35:38,0:1]
        self.Pd_rbl_coupling_outer = qd[38:42,0:1]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_rbr_inner_shaft = qdd[0:3,0:1]
        self.Pdd_rbr_inner_shaft = qdd[3:7,0:1]
        self.Rdd_rbl_inner_shaft = qdd[7:10,0:1]
        self.Pdd_rbl_inner_shaft = qdd[10:14,0:1]
        self.Rdd_rbr_coupling_inner = qdd[14:17,0:1]
        self.Pdd_rbr_coupling_inner = qdd[17:21,0:1]
        self.Rdd_rbl_coupling_inner = qdd[21:24,0:1]
        self.Pdd_rbl_coupling_inner = qdd[24:28,0:1]
        self.Rdd_rbr_coupling_outer = qdd[28:31,0:1]
        self.Pdd_rbr_coupling_outer = qdd[31:35,0:1]
        self.Rdd_rbl_coupling_outer = qdd[35:38,0:1]
        self.Pdd_rbl_coupling_outer = qdd[38:42,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcr_diff_joint = Lambda[0:5,0:1]
        self.L_jcr_inner_cv = Lambda[5:9,0:1]
        self.L_jcl_diff_joint = Lambda[9:14,0:1]
        self.L_jcl_inner_cv = Lambda[14:18,0:1]
        self.L_jcr_coupling_trans = Lambda[18:23,0:1]
        self.L_jcl_coupling_trans = Lambda[23:28,0:1]
        self.L_jcr_outer_cv = Lambda[28:32,0:1]
        self.L_jcl_outer_cv = Lambda[32:36,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_rbr_inner_shaft
        x1 = (-1) * self.R_vbs_differential
        x2 = self.P_rbr_inner_shaft
        x3 = A(x2)
        x4 = A(self.P_vbs_differential)
        x5 = x3.T
        x6 = self.Mbar_vbs_differential_jcr_diff_joint[:,2:3]
        x7 = self.R_rbr_coupling_inner
        x8 = self.P_rbr_coupling_inner
        x9 = A(x8)
        x10 = self.R_rbl_inner_shaft
        x11 = self.P_rbl_inner_shaft
        x12 = A(x11)
        x13 = x12.T
        x14 = self.Mbar_vbs_differential_jcl_diff_joint[:,2:3]
        x15 = self.R_rbl_coupling_inner
        x16 = self.P_rbl_coupling_inner
        x17 = A(x16)
        x18 = self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,0:1].T
        x19 = x9.T
        x20 = self.P_rbr_coupling_outer
        x21 = A(x20)
        x22 = self.Mbar_rbr_coupling_outer_jcr_coupling_trans[:,2:3]
        x23 = self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,1:2].T
        x24 = self.R_rbr_coupling_outer
        x25 = (x7 + (-1) * x24 + multi_dot([x9,self.ubar_rbr_coupling_inner_jcr_coupling_trans]) + (-1) * multi_dot([x21,self.ubar_rbr_coupling_outer_jcr_coupling_trans]))
        x26 = self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,0:1].T
        x27 = x17.T
        x28 = self.P_rbl_coupling_outer
        x29 = A(x28)
        x30 = self.Mbar_rbl_coupling_outer_jcl_coupling_trans[:,2:3]
        x31 = self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,1:2].T
        x32 = self.R_rbl_coupling_outer
        x33 = (x15 + (-1) * x32 + multi_dot([x17,self.ubar_rbl_coupling_inner_jcl_coupling_trans]) + (-1) * multi_dot([x29,self.ubar_rbl_coupling_outer_jcl_coupling_trans]))
        x34 = A(self.P_vbr_wheel_hub)
        x35 = A(self.P_vbl_wheel_hub)
        x36 = (-1) * np.eye(1, dtype=np.float64)

        self.pos_eq_blocks = ((x0 + x1 + multi_dot([x3,self.ubar_rbr_inner_shaft_jcr_diff_joint]) + (-1) * multi_dot([x4,self.ubar_vbs_differential_jcr_diff_joint])),
        multi_dot([self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,0:1].T,x5,x4,x6]),
        multi_dot([self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,1:2].T,x5,x4,x6]),
        (x0 + (-1) * x7 + multi_dot([x3,self.ubar_rbr_inner_shaft_jcr_inner_cv]) + (-1) * multi_dot([x9,self.ubar_rbr_coupling_inner_jcr_inner_cv])),
        multi_dot([self.Mbar_rbr_inner_shaft_jcr_inner_cv[:,0:1].T,x5,x9,self.Mbar_rbr_coupling_inner_jcr_inner_cv[:,0:1]]),
        (x10 + x1 + multi_dot([x12,self.ubar_rbl_inner_shaft_jcl_diff_joint]) + (-1) * multi_dot([x4,self.ubar_vbs_differential_jcl_diff_joint])),
        multi_dot([self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,0:1].T,x13,x4,x14]),
        multi_dot([self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,1:2].T,x13,x4,x14]),
        (x10 + (-1) * x15 + multi_dot([x12,self.ubar_rbl_inner_shaft_jcl_inner_cv]) + (-1) * multi_dot([x17,self.ubar_rbl_coupling_inner_jcl_inner_cv])),
        multi_dot([self.Mbar_rbl_inner_shaft_jcl_inner_cv[:,0:1].T,x13,x17,self.Mbar_rbl_coupling_inner_jcl_inner_cv[:,0:1]]),
        multi_dot([x18,x19,x21,x22]),
        multi_dot([x23,x19,x21,x22]),
        multi_dot([x18,x19,x25]),
        multi_dot([x23,x19,x25]),
        multi_dot([x18,x19,x21,self.Mbar_rbr_coupling_outer_jcr_coupling_trans[:,1:2]]),
        multi_dot([x26,x27,x29,x30]),
        multi_dot([x31,x27,x29,x30]),
        multi_dot([x26,x27,x33]),
        multi_dot([x31,x27,x33]),
        multi_dot([x26,x27,x29,self.Mbar_rbl_coupling_outer_jcl_coupling_trans[:,1:2]]),
        (x24 + (-1) * self.R_vbr_wheel_hub + multi_dot([x21,self.ubar_rbr_coupling_outer_jcr_outer_cv]) + (-1) * multi_dot([x34,self.ubar_vbr_wheel_hub_jcr_outer_cv])),
        multi_dot([self.Mbar_rbr_coupling_outer_jcr_outer_cv[:,0:1].T,x21.T,x34,self.Mbar_vbr_wheel_hub_jcr_outer_cv[:,0:1]]),
        (x32 + (-1) * self.R_vbl_wheel_hub + multi_dot([x29,self.ubar_rbl_coupling_outer_jcl_outer_cv]) + (-1) * multi_dot([x35,self.ubar_vbl_wheel_hub_jcl_outer_cv])),
        multi_dot([self.Mbar_rbl_coupling_outer_jcl_outer_cv[:,0:1].T,x29.T,x35,self.Mbar_vbl_wheel_hub_jcl_outer_cv[:,0:1]]),
        (x36 + multi_dot([x2.T,x2])),
        (x36 + multi_dot([x11.T,x11])),
        (x36 + multi_dot([x8.T,x8])),
        (x36 + multi_dot([x16.T,x16])),
        (x36 + multi_dot([x20.T,x20])),
        (x36 + multi_dot([x28.T,x28])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = (v0,
        v1,
        v1,
        v0,
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
        v1,
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
        v1,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbr_inner_shaft
        a1 = self.Pd_vbs_differential
        a2 = self.Mbar_vbs_differential_jcr_diff_joint[:,2:3]
        a3 = a2.T
        a4 = self.P_vbs_differential
        a5 = A(a4).T
        a6 = self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,0:1]
        a7 = self.P_rbr_inner_shaft
        a8 = A(a7).T
        a9 = B(a1,a2)
        a10 = a0.T
        a11 = B(a4,a2)
        a12 = self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,1:2]
        a13 = self.Pd_rbr_coupling_inner
        a14 = self.Mbar_rbr_coupling_inner_jcr_inner_cv[:,0:1]
        a15 = self.P_rbr_coupling_inner
        a16 = A(a15).T
        a17 = self.Mbar_rbr_inner_shaft_jcr_inner_cv[:,0:1]
        a18 = self.Pd_rbl_inner_shaft
        a19 = self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,0:1]
        a20 = self.P_rbl_inner_shaft
        a21 = A(a20).T
        a22 = self.Mbar_vbs_differential_jcl_diff_joint[:,2:3]
        a23 = B(a1,a22)
        a24 = a22.T
        a25 = a18.T
        a26 = B(a4,a22)
        a27 = self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,1:2]
        a28 = self.Pd_rbl_coupling_inner
        a29 = self.Mbar_rbl_coupling_inner_jcl_inner_cv[:,0:1]
        a30 = self.P_rbl_coupling_inner
        a31 = A(a30).T
        a32 = self.Mbar_rbl_inner_shaft_jcl_inner_cv[:,0:1]
        a33 = self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,0:1]
        a34 = a33.T
        a35 = self.Pd_rbr_coupling_outer
        a36 = self.Mbar_rbr_coupling_outer_jcr_coupling_trans[:,2:3]
        a37 = B(a35,a36)
        a38 = a36.T
        a39 = self.P_rbr_coupling_outer
        a40 = A(a39).T
        a41 = B(a13,a33)
        a42 = a13.T
        a43 = B(a15,a33).T
        a44 = B(a39,a36)
        a45 = self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,1:2]
        a46 = a45.T
        a47 = B(a13,a45)
        a48 = B(a15,a45).T
        a49 = self.ubar_rbr_coupling_inner_jcr_coupling_trans
        a50 = self.ubar_rbr_coupling_outer_jcr_coupling_trans
        a51 = (multi_dot([B(a13,a49),a13]) + (-1) * multi_dot([B(a35,a50),a35]))
        a52 = (self.Rd_rbr_coupling_inner + (-1) * self.Rd_rbr_coupling_outer + multi_dot([B(a15,a49),a13]) + (-1) * multi_dot([B(a39,a50),a35]))
        a53 = (self.R_rbr_coupling_inner.T + (-1) * self.R_rbr_coupling_outer.T + multi_dot([a49.T,a16]) + (-1) * multi_dot([a50.T,a40]))
        a54 = self.Mbar_rbr_coupling_outer_jcr_coupling_trans[:,1:2]
        a55 = self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,0:1]
        a56 = a55.T
        a57 = self.Pd_rbl_coupling_outer
        a58 = self.Mbar_rbl_coupling_outer_jcl_coupling_trans[:,2:3]
        a59 = B(a57,a58)
        a60 = a58.T
        a61 = self.P_rbl_coupling_outer
        a62 = A(a61).T
        a63 = B(a28,a55)
        a64 = a28.T
        a65 = B(a30,a55).T
        a66 = B(a61,a58)
        a67 = self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,1:2]
        a68 = a67.T
        a69 = B(a28,a67)
        a70 = B(a30,a67).T
        a71 = self.ubar_rbl_coupling_inner_jcl_coupling_trans
        a72 = self.ubar_rbl_coupling_outer_jcl_coupling_trans
        a73 = (multi_dot([B(a28,a71),a28]) + (-1) * multi_dot([B(a57,a72),a57]))
        a74 = (self.Rd_rbl_coupling_inner + (-1) * self.Rd_rbl_coupling_outer + multi_dot([B(a30,a71),a28]) + (-1) * multi_dot([B(a61,a72),a57]))
        a75 = (self.R_rbl_coupling_inner.T + (-1) * self.R_rbl_coupling_outer.T + multi_dot([a71.T,a31]) + (-1) * multi_dot([a72.T,a62]))
        a76 = self.Mbar_rbl_coupling_outer_jcl_coupling_trans[:,1:2]
        a77 = self.Pd_vbr_wheel_hub
        a78 = self.Mbar_rbr_coupling_outer_jcr_outer_cv[:,0:1]
        a79 = self.Mbar_vbr_wheel_hub_jcr_outer_cv[:,0:1]
        a80 = self.P_vbr_wheel_hub
        a81 = a35.T
        a82 = self.Pd_vbl_wheel_hub
        a83 = self.Mbar_rbl_coupling_outer_jcl_outer_cv[:,0:1]
        a84 = self.Mbar_vbl_wheel_hub_jcl_outer_cv[:,0:1]
        a85 = self.P_vbl_wheel_hub
        a86 = a57.T

        self.acc_eq_blocks = ((multi_dot([B(a0,self.ubar_rbr_inner_shaft_jcr_diff_joint),a0]) + (-1) * multi_dot([B(a1,self.ubar_vbs_differential_jcr_diff_joint),a1])),
        (multi_dot([a3,a5,B(a0,a6),a0]) + multi_dot([a6.T,a8,a9,a1]) + (2) * multi_dot([a10,B(a7,a6).T,a11,a1])),
        (multi_dot([a3,a5,B(a0,a12),a0]) + multi_dot([a12.T,a8,a9,a1]) + (2) * multi_dot([a10,B(a7,a12).T,a11,a1])),
        (multi_dot([B(a0,self.ubar_rbr_inner_shaft_jcr_inner_cv),a0]) + (-1) * multi_dot([B(a13,self.ubar_rbr_coupling_inner_jcr_inner_cv),a13])),
        (multi_dot([a14.T,a16,B(a0,a17),a0]) + multi_dot([a17.T,a8,B(a13,a14),a13]) + (2) * multi_dot([a10,B(a7,a17).T,B(a15,a14),a13])),
        (multi_dot([B(a18,self.ubar_rbl_inner_shaft_jcl_diff_joint),a18]) + (-1) * multi_dot([B(a1,self.ubar_vbs_differential_jcl_diff_joint),a1])),
        (multi_dot([a19.T,a21,a23,a1]) + multi_dot([a24,a5,B(a18,a19),a18]) + (2) * multi_dot([a25,B(a20,a19).T,a26,a1])),
        (multi_dot([a27.T,a21,a23,a1]) + multi_dot([a24,a5,B(a18,a27),a18]) + (2) * multi_dot([a25,B(a20,a27).T,a26,a1])),
        (multi_dot([B(a18,self.ubar_rbl_inner_shaft_jcl_inner_cv),a18]) + (-1) * multi_dot([B(a28,self.ubar_rbl_coupling_inner_jcl_inner_cv),a28])),
        (multi_dot([a29.T,a31,B(a18,a32),a18]) + multi_dot([a32.T,a21,B(a28,a29),a28]) + (2) * multi_dot([a25,B(a20,a32).T,B(a30,a29),a28])),
        (multi_dot([a34,a16,a37,a35]) + multi_dot([a38,a40,a41,a13]) + (2) * multi_dot([a42,a43,a44,a35])),
        (multi_dot([a46,a16,a37,a35]) + multi_dot([a38,a40,a47,a13]) + (2) * multi_dot([a42,a48,a44,a35])),
        (multi_dot([a34,a16,a51]) + (2) * multi_dot([a42,a43,a52]) + multi_dot([a53,a41,a13])),
        (multi_dot([a46,a16,a51]) + (2) * multi_dot([a42,a48,a52]) + multi_dot([a53,a47,a13])),
        (multi_dot([a34,a16,B(a35,a54),a35]) + multi_dot([a54.T,a40,a41,a13]) + (2) * multi_dot([a42,a43,B(a39,a54),a35])),
        (multi_dot([a56,a31,a59,a57]) + multi_dot([a60,a62,a63,a28]) + (2) * multi_dot([a64,a65,a66,a57])),
        (multi_dot([a68,a31,a59,a57]) + multi_dot([a60,a62,a69,a28]) + (2) * multi_dot([a64,a70,a66,a57])),
        (multi_dot([a56,a31,a73]) + (2) * multi_dot([a64,a65,a74]) + multi_dot([a75,a63,a28])),
        (multi_dot([a68,a31,a73]) + (2) * multi_dot([a64,a70,a74]) + multi_dot([a75,a69,a28])),
        (multi_dot([a56,a31,B(a57,a76),a57]) + multi_dot([a76.T,a62,a63,a28]) + (2) * multi_dot([a64,a65,B(a61,a76),a57])),
        (multi_dot([B(a35,self.ubar_rbr_coupling_outer_jcr_outer_cv),a35]) + (-1) * multi_dot([B(a77,self.ubar_vbr_wheel_hub_jcr_outer_cv),a77])),
        (multi_dot([a78.T,a40,B(a77,a79),a77]) + multi_dot([a79.T,A(a80).T,B(a35,a78),a35]) + (2) * multi_dot([a81,B(a39,a78).T,B(a80,a79),a77])),
        (multi_dot([B(a57,self.ubar_rbl_coupling_outer_jcl_outer_cv),a57]) + (-1) * multi_dot([B(a82,self.ubar_vbl_wheel_hub_jcl_outer_cv),a82])),
        (multi_dot([a83.T,a62,B(a82,a84),a82]) + multi_dot([a84.T,A(a85).T,B(a57,a83),a57]) + (2) * multi_dot([a86,B(a61,a83).T,B(a85,a84),a82])),
        (2) * multi_dot([a10,a0]),
        (2) * multi_dot([a25,a18]),
        (2) * multi_dot([a42,a13]),
        (2) * multi_dot([a64,a28]),
        (2) * multi_dot([a81,a35]),
        (2) * multi_dot([a86,a57]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3, dtype=np.float64)
        j1 = self.P_rbr_inner_shaft
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = self.Mbar_vbs_differential_jcr_diff_joint[:,2:3]
        j4 = j3.T
        j5 = self.P_vbs_differential
        j6 = A(j5).T
        j7 = self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,0:1]
        j8 = self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,1:2]
        j9 = (-1) * j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.Mbar_rbr_coupling_inner_jcr_inner_cv[:,0:1]
        j13 = self.P_rbr_coupling_inner
        j14 = A(j13).T
        j15 = self.Mbar_rbr_inner_shaft_jcr_inner_cv[:,0:1]
        j16 = self.P_rbl_inner_shaft
        j17 = self.Mbar_vbs_differential_jcl_diff_joint[:,2:3]
        j18 = j17.T
        j19 = self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,0:1]
        j20 = self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,1:2]
        j21 = A(j16).T
        j22 = B(j5,j17)
        j23 = self.Mbar_rbl_coupling_inner_jcl_inner_cv[:,0:1]
        j24 = self.P_rbl_coupling_inner
        j25 = A(j24).T
        j26 = self.Mbar_rbl_inner_shaft_jcl_inner_cv[:,0:1]
        j27 = self.Mbar_rbr_coupling_outer_jcr_coupling_trans[:,2:3]
        j28 = j27.T
        j29 = self.P_rbr_coupling_outer
        j30 = A(j29).T
        j31 = self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,0:1]
        j32 = B(j13,j31)
        j33 = self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,1:2]
        j34 = B(j13,j33)
        j35 = j31.T
        j36 = multi_dot([j35,j14])
        j37 = self.ubar_rbr_coupling_inner_jcr_coupling_trans
        j38 = B(j13,j37)
        j39 = self.ubar_rbr_coupling_outer_jcr_coupling_trans
        j40 = (self.R_rbr_coupling_inner.T + (-1) * self.R_rbr_coupling_outer.T + multi_dot([j37.T,j14]) + (-1) * multi_dot([j39.T,j30]))
        j41 = j33.T
        j42 = multi_dot([j41,j14])
        j43 = self.Mbar_rbr_coupling_outer_jcr_coupling_trans[:,1:2]
        j44 = B(j29,j27)
        j45 = B(j29,j39)
        j46 = self.Mbar_rbl_coupling_outer_jcl_coupling_trans[:,2:3]
        j47 = j46.T
        j48 = self.P_rbl_coupling_outer
        j49 = A(j48).T
        j50 = self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,0:1]
        j51 = B(j24,j50)
        j52 = self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,1:2]
        j53 = B(j24,j52)
        j54 = j50.T
        j55 = multi_dot([j54,j25])
        j56 = self.ubar_rbl_coupling_inner_jcl_coupling_trans
        j57 = B(j24,j56)
        j58 = self.ubar_rbl_coupling_outer_jcl_coupling_trans
        j59 = (self.R_rbl_coupling_inner.T + (-1) * self.R_rbl_coupling_outer.T + multi_dot([j56.T,j25]) + (-1) * multi_dot([j58.T,j49]))
        j60 = j52.T
        j61 = multi_dot([j60,j25])
        j62 = self.Mbar_rbl_coupling_outer_jcl_coupling_trans[:,1:2]
        j63 = B(j48,j46)
        j64 = B(j48,j58)
        j65 = self.Mbar_vbr_wheel_hub_jcr_outer_cv[:,0:1]
        j66 = self.P_vbr_wheel_hub
        j67 = self.Mbar_rbr_coupling_outer_jcr_outer_cv[:,0:1]
        j68 = self.Mbar_vbl_wheel_hub_jcl_outer_cv[:,0:1]
        j69 = self.P_vbl_wheel_hub
        j70 = self.Mbar_rbl_coupling_outer_jcl_outer_cv[:,0:1]

        self.jac_eq_blocks = (j0,
        B(j1,self.ubar_rbr_inner_shaft_jcr_diff_joint),
        j9,
        (-1) * B(j5,self.ubar_vbs_differential_jcr_diff_joint),
        j2,
        multi_dot([j4,j6,B(j1,j7)]),
        j2,
        multi_dot([j7.T,j10,j11]),
        j2,
        multi_dot([j4,j6,B(j1,j8)]),
        j2,
        multi_dot([j8.T,j10,j11]),
        j0,
        B(j1,self.ubar_rbr_inner_shaft_jcr_inner_cv),
        j9,
        (-1) * B(j13,self.ubar_rbr_coupling_inner_jcr_inner_cv),
        j2,
        multi_dot([j12.T,j14,B(j1,j15)]),
        j2,
        multi_dot([j15.T,j10,B(j13,j12)]),
        j0,
        B(j16,self.ubar_rbl_inner_shaft_jcl_diff_joint),
        j9,
        (-1) * B(j5,self.ubar_vbs_differential_jcl_diff_joint),
        j2,
        multi_dot([j18,j6,B(j16,j19)]),
        j2,
        multi_dot([j19.T,j21,j22]),
        j2,
        multi_dot([j18,j6,B(j16,j20)]),
        j2,
        multi_dot([j20.T,j21,j22]),
        j0,
        B(j16,self.ubar_rbl_inner_shaft_jcl_inner_cv),
        j9,
        (-1) * B(j24,self.ubar_rbl_coupling_inner_jcl_inner_cv),
        j2,
        multi_dot([j23.T,j25,B(j16,j26)]),
        j2,
        multi_dot([j26.T,j21,B(j24,j23)]),
        j2,
        multi_dot([j28,j30,j32]),
        j2,
        multi_dot([j35,j14,j44]),
        j2,
        multi_dot([j28,j30,j34]),
        j2,
        multi_dot([j41,j14,j44]),
        j36,
        (multi_dot([j35,j14,j38]) + multi_dot([j40,j32])),
        (-1) * j36,
        (-1) * multi_dot([j35,j14,j45]),
        j42,
        (multi_dot([j41,j14,j38]) + multi_dot([j40,j34])),
        (-1) * j42,
        (-1) * multi_dot([j41,j14,j45]),
        j2,
        multi_dot([j43.T,j30,j32]),
        j2,
        multi_dot([j35,j14,B(j29,j43)]),
        j2,
        multi_dot([j47,j49,j51]),
        j2,
        multi_dot([j54,j25,j63]),
        j2,
        multi_dot([j47,j49,j53]),
        j2,
        multi_dot([j60,j25,j63]),
        j55,
        (multi_dot([j54,j25,j57]) + multi_dot([j59,j51])),
        (-1) * j55,
        (-1) * multi_dot([j54,j25,j64]),
        j61,
        (multi_dot([j60,j25,j57]) + multi_dot([j59,j53])),
        (-1) * j61,
        (-1) * multi_dot([j60,j25,j64]),
        j2,
        multi_dot([j62.T,j49,j51]),
        j2,
        multi_dot([j54,j25,B(j48,j62)]),
        j0,
        B(j29,self.ubar_rbr_coupling_outer_jcr_outer_cv),
        j9,
        (-1) * B(j66,self.ubar_vbr_wheel_hub_jcr_outer_cv),
        j2,
        multi_dot([j65.T,A(j66).T,B(j29,j67)]),
        j2,
        multi_dot([j67.T,j30,B(j66,j65)]),
        j0,
        B(j48,self.ubar_rbl_coupling_outer_jcl_outer_cv),
        j9,
        (-1) * B(j69,self.ubar_vbl_wheel_hub_jcl_outer_cv),
        j2,
        multi_dot([j68.T,A(j69).T,B(j48,j70)]),
        j2,
        multi_dot([j70.T,j49,B(j69,j68)]),
        j2,
        (2) * j1.T,
        j2,
        (2) * j16.T,
        j2,
        (2) * j13.T,
        j2,
        (2) * j24.T,
        j2,
        (2) * j29.T,
        j2,
        (2) * j48.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = np.eye(3, dtype=np.float64)
        m1 = G(self.P_rbr_inner_shaft)
        m2 = G(self.P_rbl_inner_shaft)
        m3 = G(self.P_rbr_coupling_inner)
        m4 = G(self.P_rbl_coupling_inner)
        m5 = G(self.P_rbr_coupling_outer)
        m6 = G(self.P_rbl_coupling_outer)

        self.mass_eq_blocks = (config.m_rbr_inner_shaft * m0,
        (4) * multi_dot([m1.T,config.Jbar_rbr_inner_shaft,m1]),
        config.m_rbl_inner_shaft * m0,
        (4) * multi_dot([m2.T,config.Jbar_rbl_inner_shaft,m2]),
        config.m_rbr_coupling_inner * m0,
        (4) * multi_dot([m3.T,config.Jbar_rbr_coupling_inner,m3]),
        config.m_rbl_coupling_inner * m0,
        (4) * multi_dot([m4.T,config.Jbar_rbl_coupling_inner,m4]),
        config.m_rbr_coupling_outer * m0,
        (4) * multi_dot([m5.T,config.Jbar_rbr_coupling_outer,m5]),
        config.m_rbl_coupling_outer * m0,
        (4) * multi_dot([m6.T,config.Jbar_rbl_coupling_outer,m6]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = np.zeros((3,1),dtype=np.float64)
        f1 = self.P_rbr_inner_shaft
        f2 = G(self.Pd_rbr_inner_shaft)
        f3 = self.P_rbl_inner_shaft
        f4 = G(self.Pd_rbl_inner_shaft)
        f5 = G(self.Pd_rbr_coupling_inner)
        f6 = G(self.Pd_rbl_coupling_inner)
        f7 = G(self.Pd_rbr_coupling_outer)
        f8 = G(self.Pd_rbl_coupling_outer)

        self.frc_eq_blocks = ((self.F_rbr_inner_shaft_gravity + f0),
        ((2 * config.UF_far_drive(t)) * multi_dot([G(f1).T,self.vbar_rbr_inner_shaft_far_drive]) + (8) * multi_dot([f2.T,config.Jbar_rbr_inner_shaft,f2,f1])),
        (self.F_rbl_inner_shaft_gravity + f0),
        ((2 * config.UF_fal_drive(t)) * multi_dot([G(f3).T,self.vbar_rbl_inner_shaft_fal_drive]) + (8) * multi_dot([f4.T,config.Jbar_rbl_inner_shaft,f4,f3])),
        self.F_rbr_coupling_inner_gravity,
        (8) * multi_dot([f5.T,config.Jbar_rbr_coupling_inner,f5,self.P_rbr_coupling_inner]),
        self.F_rbl_coupling_inner_gravity,
        (8) * multi_dot([f6.T,config.Jbar_rbl_coupling_inner,f6,self.P_rbl_coupling_inner]),
        self.F_rbr_coupling_outer_gravity,
        (8) * multi_dot([f7.T,config.Jbar_rbr_coupling_outer,f7,self.P_rbr_coupling_outer]),
        self.F_rbl_coupling_outer_gravity,
        (8) * multi_dot([f8.T,config.Jbar_rbl_coupling_outer,f8,self.P_rbl_coupling_outer]),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbr_inner_shaft_jcr_diff_joint = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_inner_shaft,self.ubar_rbr_inner_shaft_jcr_diff_joint).T,multi_dot([B(self.P_rbr_inner_shaft,self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,0:1]).T,A(self.P_vbs_differential),self.Mbar_vbs_differential_jcr_diff_joint[:,2:3]]),multi_dot([B(self.P_rbr_inner_shaft,self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,1:2]).T,A(self.P_vbs_differential),self.Mbar_vbs_differential_jcr_diff_joint[:,2:3]])]]),self.L_jcr_diff_joint])
        self.F_rbr_inner_shaft_jcr_diff_joint = Q_rbr_inner_shaft_jcr_diff_joint[0:3,0:1]
        Te_rbr_inner_shaft_jcr_diff_joint = Q_rbr_inner_shaft_jcr_diff_joint[3:7,0:1]
        self.T_rbr_inner_shaft_jcr_diff_joint = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_inner_shaft),self.ubar_rbr_inner_shaft_jcr_diff_joint])),self.F_rbr_inner_shaft_jcr_diff_joint]) + (0.5) * multi_dot([E(self.P_rbr_inner_shaft),Te_rbr_inner_shaft_jcr_diff_joint]))
        Q_rbr_inner_shaft_jcr_inner_cv = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_inner_shaft,self.ubar_rbr_inner_shaft_jcr_inner_cv).T,multi_dot([B(self.P_rbr_inner_shaft,self.Mbar_rbr_inner_shaft_jcr_inner_cv[:,0:1]).T,A(self.P_rbr_coupling_inner),self.Mbar_rbr_coupling_inner_jcr_inner_cv[:,0:1]])]]),self.L_jcr_inner_cv])
        self.F_rbr_inner_shaft_jcr_inner_cv = Q_rbr_inner_shaft_jcr_inner_cv[0:3,0:1]
        Te_rbr_inner_shaft_jcr_inner_cv = Q_rbr_inner_shaft_jcr_inner_cv[3:7,0:1]
        self.T_rbr_inner_shaft_jcr_inner_cv = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_inner_shaft),self.ubar_rbr_inner_shaft_jcr_inner_cv])),self.F_rbr_inner_shaft_jcr_inner_cv]) + (0.5) * multi_dot([E(self.P_rbr_inner_shaft),Te_rbr_inner_shaft_jcr_inner_cv]))
        Q_rbl_inner_shaft_jcl_diff_joint = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_inner_shaft,self.ubar_rbl_inner_shaft_jcl_diff_joint).T,multi_dot([B(self.P_rbl_inner_shaft,self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,0:1]).T,A(self.P_vbs_differential),self.Mbar_vbs_differential_jcl_diff_joint[:,2:3]]),multi_dot([B(self.P_rbl_inner_shaft,self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,1:2]).T,A(self.P_vbs_differential),self.Mbar_vbs_differential_jcl_diff_joint[:,2:3]])]]),self.L_jcl_diff_joint])
        self.F_rbl_inner_shaft_jcl_diff_joint = Q_rbl_inner_shaft_jcl_diff_joint[0:3,0:1]
        Te_rbl_inner_shaft_jcl_diff_joint = Q_rbl_inner_shaft_jcl_diff_joint[3:7,0:1]
        self.T_rbl_inner_shaft_jcl_diff_joint = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_inner_shaft),self.ubar_rbl_inner_shaft_jcl_diff_joint])),self.F_rbl_inner_shaft_jcl_diff_joint]) + (0.5) * multi_dot([E(self.P_rbl_inner_shaft),Te_rbl_inner_shaft_jcl_diff_joint]))
        Q_rbl_inner_shaft_jcl_inner_cv = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_inner_shaft,self.ubar_rbl_inner_shaft_jcl_inner_cv).T,multi_dot([B(self.P_rbl_inner_shaft,self.Mbar_rbl_inner_shaft_jcl_inner_cv[:,0:1]).T,A(self.P_rbl_coupling_inner),self.Mbar_rbl_coupling_inner_jcl_inner_cv[:,0:1]])]]),self.L_jcl_inner_cv])
        self.F_rbl_inner_shaft_jcl_inner_cv = Q_rbl_inner_shaft_jcl_inner_cv[0:3,0:1]
        Te_rbl_inner_shaft_jcl_inner_cv = Q_rbl_inner_shaft_jcl_inner_cv[3:7,0:1]
        self.T_rbl_inner_shaft_jcl_inner_cv = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_inner_shaft),self.ubar_rbl_inner_shaft_jcl_inner_cv])),self.F_rbl_inner_shaft_jcl_inner_cv]) + (0.5) * multi_dot([E(self.P_rbl_inner_shaft),Te_rbl_inner_shaft_jcl_inner_cv]))
        Q_rbr_coupling_inner_jcr_coupling_trans = (-1) * multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_rbr_coupling_inner),self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,0:1]]),multi_dot([A(self.P_rbr_coupling_inner),self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,1:2]]),np.zeros((1,3),dtype=np.float64).T],[multi_dot([B(self.P_rbr_coupling_inner,self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,0:1]).T,A(self.P_rbr_coupling_outer),self.Mbar_rbr_coupling_outer_jcr_coupling_trans[:,2:3]]),multi_dot([B(self.P_rbr_coupling_inner,self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,1:2]).T,A(self.P_rbr_coupling_outer),self.Mbar_rbr_coupling_outer_jcr_coupling_trans[:,2:3]]),(multi_dot([B(self.P_rbr_coupling_inner,self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,0:1]).T,((-1) * self.R_rbr_coupling_outer + multi_dot([A(self.P_rbr_coupling_inner),self.ubar_rbr_coupling_inner_jcr_coupling_trans]) + (-1) * multi_dot([A(self.P_rbr_coupling_outer),self.ubar_rbr_coupling_outer_jcr_coupling_trans]) + self.R_rbr_coupling_inner)]) + multi_dot([B(self.P_rbr_coupling_inner,self.ubar_rbr_coupling_inner_jcr_coupling_trans).T,A(self.P_rbr_coupling_inner),self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,0:1]])),(multi_dot([B(self.P_rbr_coupling_inner,self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,1:2]).T,((-1) * self.R_rbr_coupling_outer + multi_dot([A(self.P_rbr_coupling_inner),self.ubar_rbr_coupling_inner_jcr_coupling_trans]) + (-1) * multi_dot([A(self.P_rbr_coupling_outer),self.ubar_rbr_coupling_outer_jcr_coupling_trans]) + self.R_rbr_coupling_inner)]) + multi_dot([B(self.P_rbr_coupling_inner,self.ubar_rbr_coupling_inner_jcr_coupling_trans).T,A(self.P_rbr_coupling_inner),self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,1:2]])),multi_dot([B(self.P_rbr_coupling_inner,self.Mbar_rbr_coupling_inner_jcr_coupling_trans[:,0:1]).T,A(self.P_rbr_coupling_outer),self.Mbar_rbr_coupling_outer_jcr_coupling_trans[:,1:2]])]]),self.L_jcr_coupling_trans])
        self.F_rbr_coupling_inner_jcr_coupling_trans = Q_rbr_coupling_inner_jcr_coupling_trans[0:3,0:1]
        Te_rbr_coupling_inner_jcr_coupling_trans = Q_rbr_coupling_inner_jcr_coupling_trans[3:7,0:1]
        self.T_rbr_coupling_inner_jcr_coupling_trans = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_coupling_inner),self.ubar_rbr_coupling_inner_jcr_coupling_trans])),self.F_rbr_coupling_inner_jcr_coupling_trans]) + (0.5) * multi_dot([E(self.P_rbr_coupling_inner),Te_rbr_coupling_inner_jcr_coupling_trans]))
        Q_rbl_coupling_inner_jcl_coupling_trans = (-1) * multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_rbl_coupling_inner),self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,0:1]]),multi_dot([A(self.P_rbl_coupling_inner),self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,1:2]]),np.zeros((1,3),dtype=np.float64).T],[multi_dot([B(self.P_rbl_coupling_inner,self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,0:1]).T,A(self.P_rbl_coupling_outer),self.Mbar_rbl_coupling_outer_jcl_coupling_trans[:,2:3]]),multi_dot([B(self.P_rbl_coupling_inner,self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,1:2]).T,A(self.P_rbl_coupling_outer),self.Mbar_rbl_coupling_outer_jcl_coupling_trans[:,2:3]]),(multi_dot([B(self.P_rbl_coupling_inner,self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,0:1]).T,((-1) * self.R_rbl_coupling_outer + multi_dot([A(self.P_rbl_coupling_inner),self.ubar_rbl_coupling_inner_jcl_coupling_trans]) + (-1) * multi_dot([A(self.P_rbl_coupling_outer),self.ubar_rbl_coupling_outer_jcl_coupling_trans]) + self.R_rbl_coupling_inner)]) + multi_dot([B(self.P_rbl_coupling_inner,self.ubar_rbl_coupling_inner_jcl_coupling_trans).T,A(self.P_rbl_coupling_inner),self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,0:1]])),(multi_dot([B(self.P_rbl_coupling_inner,self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,1:2]).T,((-1) * self.R_rbl_coupling_outer + multi_dot([A(self.P_rbl_coupling_inner),self.ubar_rbl_coupling_inner_jcl_coupling_trans]) + (-1) * multi_dot([A(self.P_rbl_coupling_outer),self.ubar_rbl_coupling_outer_jcl_coupling_trans]) + self.R_rbl_coupling_inner)]) + multi_dot([B(self.P_rbl_coupling_inner,self.ubar_rbl_coupling_inner_jcl_coupling_trans).T,A(self.P_rbl_coupling_inner),self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,1:2]])),multi_dot([B(self.P_rbl_coupling_inner,self.Mbar_rbl_coupling_inner_jcl_coupling_trans[:,0:1]).T,A(self.P_rbl_coupling_outer),self.Mbar_rbl_coupling_outer_jcl_coupling_trans[:,1:2]])]]),self.L_jcl_coupling_trans])
        self.F_rbl_coupling_inner_jcl_coupling_trans = Q_rbl_coupling_inner_jcl_coupling_trans[0:3,0:1]
        Te_rbl_coupling_inner_jcl_coupling_trans = Q_rbl_coupling_inner_jcl_coupling_trans[3:7,0:1]
        self.T_rbl_coupling_inner_jcl_coupling_trans = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_coupling_inner),self.ubar_rbl_coupling_inner_jcl_coupling_trans])),self.F_rbl_coupling_inner_jcl_coupling_trans]) + (0.5) * multi_dot([E(self.P_rbl_coupling_inner),Te_rbl_coupling_inner_jcl_coupling_trans]))
        Q_rbr_coupling_outer_jcr_outer_cv = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_coupling_outer,self.ubar_rbr_coupling_outer_jcr_outer_cv).T,multi_dot([B(self.P_rbr_coupling_outer,self.Mbar_rbr_coupling_outer_jcr_outer_cv[:,0:1]).T,A(self.P_vbr_wheel_hub),self.Mbar_vbr_wheel_hub_jcr_outer_cv[:,0:1]])]]),self.L_jcr_outer_cv])
        self.F_rbr_coupling_outer_jcr_outer_cv = Q_rbr_coupling_outer_jcr_outer_cv[0:3,0:1]
        Te_rbr_coupling_outer_jcr_outer_cv = Q_rbr_coupling_outer_jcr_outer_cv[3:7,0:1]
        self.T_rbr_coupling_outer_jcr_outer_cv = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_coupling_outer),self.ubar_rbr_coupling_outer_jcr_outer_cv])),self.F_rbr_coupling_outer_jcr_outer_cv]) + (0.5) * multi_dot([E(self.P_rbr_coupling_outer),Te_rbr_coupling_outer_jcr_outer_cv]))
        Q_rbl_coupling_outer_jcl_outer_cv = (-1) * multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_coupling_outer,self.ubar_rbl_coupling_outer_jcl_outer_cv).T,multi_dot([B(self.P_rbl_coupling_outer,self.Mbar_rbl_coupling_outer_jcl_outer_cv[:,0:1]).T,A(self.P_vbl_wheel_hub),self.Mbar_vbl_wheel_hub_jcl_outer_cv[:,0:1]])]]),self.L_jcl_outer_cv])
        self.F_rbl_coupling_outer_jcl_outer_cv = Q_rbl_coupling_outer_jcl_outer_cv[0:3,0:1]
        Te_rbl_coupling_outer_jcl_outer_cv = Q_rbl_coupling_outer_jcl_outer_cv[3:7,0:1]
        self.T_rbl_coupling_outer_jcl_outer_cv = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_coupling_outer),self.ubar_rbl_coupling_outer_jcl_outer_cv])),self.F_rbl_coupling_outer_jcl_outer_cv]) + (0.5) * multi_dot([E(self.P_rbl_coupling_outer),Te_rbl_coupling_outer_jcl_outer_cv]))

        self.reactions = {'F_rbr_inner_shaft_jcr_diff_joint' : self.F_rbr_inner_shaft_jcr_diff_joint,
                        'T_rbr_inner_shaft_jcr_diff_joint' : self.T_rbr_inner_shaft_jcr_diff_joint,
                        'F_rbr_inner_shaft_jcr_inner_cv' : self.F_rbr_inner_shaft_jcr_inner_cv,
                        'T_rbr_inner_shaft_jcr_inner_cv' : self.T_rbr_inner_shaft_jcr_inner_cv,
                        'F_rbl_inner_shaft_jcl_diff_joint' : self.F_rbl_inner_shaft_jcl_diff_joint,
                        'T_rbl_inner_shaft_jcl_diff_joint' : self.T_rbl_inner_shaft_jcl_diff_joint,
                        'F_rbl_inner_shaft_jcl_inner_cv' : self.F_rbl_inner_shaft_jcl_inner_cv,
                        'T_rbl_inner_shaft_jcl_inner_cv' : self.T_rbl_inner_shaft_jcl_inner_cv,
                        'F_rbr_coupling_inner_jcr_coupling_trans' : self.F_rbr_coupling_inner_jcr_coupling_trans,
                        'T_rbr_coupling_inner_jcr_coupling_trans' : self.T_rbr_coupling_inner_jcr_coupling_trans,
                        'F_rbl_coupling_inner_jcl_coupling_trans' : self.F_rbl_coupling_inner_jcl_coupling_trans,
                        'T_rbl_coupling_inner_jcl_coupling_trans' : self.T_rbl_coupling_inner_jcl_coupling_trans,
                        'F_rbr_coupling_outer_jcr_outer_cv' : self.F_rbr_coupling_outer_jcr_outer_cv,
                        'T_rbr_coupling_outer_jcr_outer_cv' : self.T_rbr_coupling_outer_jcr_outer_cv,
                        'F_rbl_coupling_outer_jcl_outer_cv' : self.F_rbl_coupling_outer_jcl_outer_cv,
                        'T_rbl_coupling_outer_jcl_outer_cv' : self.T_rbl_coupling_outer_jcl_outer_cv}

