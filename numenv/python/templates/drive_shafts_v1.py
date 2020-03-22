
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

        self.indicies_map = {'vbs_ground': 0, 'rbr_inner_shaft': 1, 'rbl_inner_shaft': 2, 'rbr_coupling': 3, 'rbl_coupling': 4, 'vbs_differential': 5, 'vbr_wheel_hub': 6, 'vbl_wheel_hub': 7}

        self.n  = 28
        self.nc = 28
        self.nrows = 20
        self.ncols = 2*4
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_rbr_inner_shaft_jcr_diff_joint', 'T_rbr_inner_shaft_jcr_diff_joint', 'F_rbr_inner_shaft_jcr_inner_cv', 'T_rbr_inner_shaft_jcr_inner_cv', 'F_rbl_inner_shaft_jcl_diff_joint', 'T_rbl_inner_shaft_jcl_diff_joint', 'F_rbl_inner_shaft_jcl_inner_cv', 'T_rbl_inner_shaft_jcl_inner_cv', 'F_rbr_coupling_jcr_outer_cv', 'T_rbr_coupling_jcr_outer_cv', 'F_rbl_coupling_jcl_outer_cv', 'T_rbl_coupling_jcl_outer_cv']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.rbr_inner_shaft*2, self.rbr_inner_shaft*2+1, self.vbs_differential*2, self.vbs_differential*2+1, self.rbr_inner_shaft*2, self.rbr_inner_shaft*2+1, self.vbs_differential*2, self.vbs_differential*2+1, self.rbr_inner_shaft*2, self.rbr_inner_shaft*2+1, self.vbs_differential*2, self.vbs_differential*2+1, self.rbr_inner_shaft*2, self.rbr_inner_shaft*2+1, self.rbr_coupling*2, self.rbr_coupling*2+1, self.rbr_inner_shaft*2, self.rbr_inner_shaft*2+1, self.rbr_coupling*2, self.rbr_coupling*2+1, self.rbl_inner_shaft*2, self.rbl_inner_shaft*2+1, self.vbs_differential*2, self.vbs_differential*2+1, self.rbl_inner_shaft*2, self.rbl_inner_shaft*2+1, self.vbs_differential*2, self.vbs_differential*2+1, self.rbl_inner_shaft*2, self.rbl_inner_shaft*2+1, self.vbs_differential*2, self.vbs_differential*2+1, self.rbl_inner_shaft*2, self.rbl_inner_shaft*2+1, self.rbl_coupling*2, self.rbl_coupling*2+1, self.rbl_inner_shaft*2, self.rbl_inner_shaft*2+1, self.rbl_coupling*2, self.rbl_coupling*2+1, self.rbr_coupling*2, self.rbr_coupling*2+1, self.vbr_wheel_hub*2, self.vbr_wheel_hub*2+1, self.rbr_coupling*2, self.rbr_coupling*2+1, self.vbr_wheel_hub*2, self.vbr_wheel_hub*2+1, self.rbr_coupling*2, self.rbr_coupling*2+1, self.vbr_wheel_hub*2, self.vbr_wheel_hub*2+1, self.rbl_coupling*2, self.rbl_coupling*2+1, self.vbl_wheel_hub*2, self.vbl_wheel_hub*2+1, self.rbl_coupling*2, self.rbl_coupling*2+1, self.vbl_wheel_hub*2, self.vbl_wheel_hub*2+1, self.rbl_coupling*2, self.rbl_coupling*2+1, self.vbl_wheel_hub*2, self.vbl_wheel_hub*2+1, self.rbr_inner_shaft*2, self.rbr_inner_shaft*2+1, self.rbl_inner_shaft*2, self.rbl_inner_shaft*2+1, self.rbr_coupling*2, self.rbr_coupling*2+1, self.rbl_coupling*2, self.rbl_coupling*2+1], dtype=np.intc)

    def set_initial_states(self):
        self.q0  = np.concatenate([self.config.R_rbr_inner_shaft,
        self.config.P_rbr_inner_shaft,
        self.config.R_rbl_inner_shaft,
        self.config.P_rbl_inner_shaft,
        self.config.R_rbr_coupling,
        self.config.P_rbr_coupling,
        self.config.R_rbl_coupling,
        self.config.P_rbl_coupling])
        self.qd0 = np.concatenate([self.config.Rd_rbr_inner_shaft,
        self.config.Pd_rbr_inner_shaft,
        self.config.Rd_rbl_inner_shaft,
        self.config.Pd_rbl_inner_shaft,
        self.config.Rd_rbr_coupling,
        self.config.Pd_rbr_coupling,
        self.config.Rd_rbl_coupling,
        self.config.Pd_rbl_coupling])

        self.set_gen_coordinates(self.q0)
        self.set_gen_velocities(self.qd0)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbr_inner_shaft = indicies_map[p + 'rbr_inner_shaft']
        self.rbl_inner_shaft = indicies_map[p + 'rbl_inner_shaft']
        self.rbr_coupling = indicies_map[p + 'rbr_coupling']
        self.rbl_coupling = indicies_map[p + 'rbl_coupling']
        self.vbr_wheel_hub = indicies_map[interface_map[p + 'vbr_wheel_hub']]
        self.vbs_differential = indicies_map[interface_map[p + 'vbs_differential']]
        self.vbl_wheel_hub = indicies_map[interface_map[p + 'vbl_wheel_hub']]
        self.vbs_ground = indicies_map[interface_map[p + 'vbs_ground']]

    
    def eval_constants(self):
        config = self.config

        self.F_rbr_inner_shaft_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_inner_shaft]], dtype=np.float64)
        self.F_rbr_inner_shaft_far_drive = Z3x1
        self.F_rbl_inner_shaft_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_inner_shaft]], dtype=np.float64)
        self.F_rbl_inner_shaft_fal_drive = Z3x1
        self.F_rbr_coupling_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_coupling]], dtype=np.float64)
        self.F_rbl_coupling_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_coupling]], dtype=np.float64)

        self.vbar_rbr_inner_shaft_far_drive = multi_dot([A(config.P_rbr_inner_shaft).T,config.ax1_far_drive,(multi_dot([config.ax1_far_drive.T,A(config.P_rbr_inner_shaft),A(config.P_rbr_inner_shaft).T,config.ax1_far_drive]))**(-1.0/2.0)])
        self.Mbar_rbr_inner_shaft_far_drive = multi_dot([A(config.P_rbr_inner_shaft).T,triad(config.ax1_far_drive)])
        self.Mbar_vbs_ground_far_drive = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_far_drive)])
        self.Mbar_rbr_inner_shaft_jcr_diff_joint = multi_dot([A(config.P_rbr_inner_shaft).T,triad(config.ax1_jcr_diff_joint)])
        self.Mbar_vbs_differential_jcr_diff_joint = multi_dot([A(config.P_vbs_differential).T,triad(config.ax1_jcr_diff_joint)])
        self.ubar_rbr_inner_shaft_jcr_diff_joint = (multi_dot([A(config.P_rbr_inner_shaft).T,config.pt1_jcr_diff_joint]) + (-1) * multi_dot([A(config.P_rbr_inner_shaft).T,config.R_rbr_inner_shaft]))
        self.ubar_vbs_differential_jcr_diff_joint = (multi_dot([A(config.P_vbs_differential).T,config.pt1_jcr_diff_joint]) + (-1) * multi_dot([A(config.P_vbs_differential).T,config.R_vbs_differential]))
        self.Mbar_rbr_inner_shaft_jcr_inner_cv = multi_dot([A(config.P_rbr_inner_shaft).T,triad(config.ax1_jcr_inner_cv)])
        self.Mbar_rbr_coupling_jcr_inner_cv = multi_dot([A(config.P_rbr_coupling).T,triad(config.ax2_jcr_inner_cv,triad(config.ax1_jcr_inner_cv)[0:3,1:2])])
        self.ubar_rbr_inner_shaft_jcr_inner_cv = (multi_dot([A(config.P_rbr_inner_shaft).T,config.pt1_jcr_inner_cv]) + (-1) * multi_dot([A(config.P_rbr_inner_shaft).T,config.R_rbr_inner_shaft]))
        self.ubar_rbr_coupling_jcr_inner_cv = (multi_dot([A(config.P_rbr_coupling).T,config.pt1_jcr_inner_cv]) + (-1) * multi_dot([A(config.P_rbr_coupling).T,config.R_rbr_coupling]))
        self.vbar_rbl_inner_shaft_fal_drive = multi_dot([A(config.P_rbl_inner_shaft).T,config.ax1_fal_drive,(multi_dot([config.ax1_fal_drive.T,A(config.P_rbl_inner_shaft),A(config.P_rbl_inner_shaft).T,config.ax1_fal_drive]))**(-1.0/2.0)])
        self.Mbar_rbl_inner_shaft_fal_drive = multi_dot([A(config.P_rbl_inner_shaft).T,triad(config.ax1_fal_drive)])
        self.Mbar_vbs_ground_fal_drive = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_fal_drive)])
        self.Mbar_rbl_inner_shaft_jcl_diff_joint = multi_dot([A(config.P_rbl_inner_shaft).T,triad(config.ax1_jcl_diff_joint)])
        self.Mbar_vbs_differential_jcl_diff_joint = multi_dot([A(config.P_vbs_differential).T,triad(config.ax1_jcl_diff_joint)])
        self.ubar_rbl_inner_shaft_jcl_diff_joint = (multi_dot([A(config.P_rbl_inner_shaft).T,config.pt1_jcl_diff_joint]) + (-1) * multi_dot([A(config.P_rbl_inner_shaft).T,config.R_rbl_inner_shaft]))
        self.ubar_vbs_differential_jcl_diff_joint = (multi_dot([A(config.P_vbs_differential).T,config.pt1_jcl_diff_joint]) + (-1) * multi_dot([A(config.P_vbs_differential).T,config.R_vbs_differential]))
        self.Mbar_rbl_inner_shaft_jcl_inner_cv = multi_dot([A(config.P_rbl_inner_shaft).T,triad(config.ax1_jcl_inner_cv)])
        self.Mbar_rbl_coupling_jcl_inner_cv = multi_dot([A(config.P_rbl_coupling).T,triad(config.ax2_jcl_inner_cv,triad(config.ax1_jcl_inner_cv)[0:3,1:2])])
        self.ubar_rbl_inner_shaft_jcl_inner_cv = (multi_dot([A(config.P_rbl_inner_shaft).T,config.pt1_jcl_inner_cv]) + (-1) * multi_dot([A(config.P_rbl_inner_shaft).T,config.R_rbl_inner_shaft]))
        self.ubar_rbl_coupling_jcl_inner_cv = (multi_dot([A(config.P_rbl_coupling).T,config.pt1_jcl_inner_cv]) + (-1) * multi_dot([A(config.P_rbl_coupling).T,config.R_rbl_coupling]))
        self.Mbar_rbr_coupling_jcr_outer_cv = multi_dot([A(config.P_rbr_coupling).T,triad(config.ax1_jcr_outer_cv)])
        self.Mbar_vbr_wheel_hub_jcr_outer_cv = multi_dot([A(config.P_vbr_wheel_hub).T,triad(config.ax2_jcr_outer_cv,triad(config.ax1_jcr_outer_cv)[0:3,1:2])])
        self.ubar_rbr_coupling_jcr_outer_cv = (multi_dot([A(config.P_rbr_coupling).T,config.pt1_jcr_outer_cv]) + (-1) * multi_dot([A(config.P_rbr_coupling).T,config.R_rbr_coupling]))
        self.ubar_vbr_wheel_hub_jcr_outer_cv = (multi_dot([A(config.P_vbr_wheel_hub).T,config.pt1_jcr_outer_cv]) + (-1) * multi_dot([A(config.P_vbr_wheel_hub).T,config.R_vbr_wheel_hub]))
        self.Mbar_rbl_coupling_jcl_outer_cv = multi_dot([A(config.P_rbl_coupling).T,triad(config.ax1_jcl_outer_cv)])
        self.Mbar_vbl_wheel_hub_jcl_outer_cv = multi_dot([A(config.P_vbl_wheel_hub).T,triad(config.ax2_jcl_outer_cv,triad(config.ax1_jcl_outer_cv)[0:3,1:2])])
        self.ubar_rbl_coupling_jcl_outer_cv = (multi_dot([A(config.P_rbl_coupling).T,config.pt1_jcl_outer_cv]) + (-1) * multi_dot([A(config.P_rbl_coupling).T,config.R_rbl_coupling]))
        self.ubar_vbl_wheel_hub_jcl_outer_cv = (multi_dot([A(config.P_vbl_wheel_hub).T,config.pt1_jcl_outer_cv]) + (-1) * multi_dot([A(config.P_vbl_wheel_hub).T,config.R_vbl_wheel_hub]))

    
    def set_gen_coordinates(self,q):
        self.R_rbr_inner_shaft = q[0:3]
        self.P_rbr_inner_shaft = q[3:7]
        self.R_rbl_inner_shaft = q[7:10]
        self.P_rbl_inner_shaft = q[10:14]
        self.R_rbr_coupling = q[14:17]
        self.P_rbr_coupling = q[17:21]
        self.R_rbl_coupling = q[21:24]
        self.P_rbl_coupling = q[24:28]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbr_inner_shaft = qd[0:3]
        self.Pd_rbr_inner_shaft = qd[3:7]
        self.Rd_rbl_inner_shaft = qd[7:10]
        self.Pd_rbl_inner_shaft = qd[10:14]
        self.Rd_rbr_coupling = qd[14:17]
        self.Pd_rbr_coupling = qd[17:21]
        self.Rd_rbl_coupling = qd[21:24]
        self.Pd_rbl_coupling = qd[24:28]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_rbr_inner_shaft = qdd[0:3]
        self.Pdd_rbr_inner_shaft = qdd[3:7]
        self.Rdd_rbl_inner_shaft = qdd[7:10]
        self.Pdd_rbl_inner_shaft = qdd[10:14]
        self.Rdd_rbr_coupling = qdd[14:17]
        self.Pdd_rbr_coupling = qdd[17:21]
        self.Rdd_rbl_coupling = qdd[21:24]
        self.Pdd_rbl_coupling = qdd[24:28]

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcr_diff_joint = Lambda[0:5]
        self.L_jcr_inner_cv = Lambda[5:9]
        self.L_jcl_diff_joint = Lambda[9:14]
        self.L_jcl_inner_cv = Lambda[14:18]
        self.L_jcr_outer_cv = Lambda[18:21]
        self.L_jcl_outer_cv = Lambda[21:24]

    
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
        x7 = self.R_rbr_coupling
        x8 = self.P_rbr_coupling
        x9 = A(x8)
        x10 = self.R_rbl_inner_shaft
        x11 = self.P_rbl_inner_shaft
        x12 = A(x11)
        x13 = x12.T
        x14 = self.Mbar_vbs_differential_jcl_diff_joint[:,2:3]
        x15 = self.R_rbl_coupling
        x16 = self.P_rbl_coupling
        x17 = A(x16)
        x18 = self.Mbar_rbr_coupling_jcr_outer_cv[:,0:1].T
        x19 = x9.T
        x20 = A(self.P_vbr_wheel_hub)
        x21 = (x7 + (-1) * self.R_vbr_wheel_hub + multi_dot([x9,self.ubar_rbr_coupling_jcr_outer_cv]) + (-1) * multi_dot([x20,self.ubar_vbr_wheel_hub_jcr_outer_cv]))
        x22 = self.Mbar_rbl_coupling_jcl_outer_cv[:,0:1].T
        x23 = x17.T
        x24 = A(self.P_vbl_wheel_hub)
        x25 = (x15 + (-1) * self.R_vbl_wheel_hub + multi_dot([x17,self.ubar_rbl_coupling_jcl_outer_cv]) + (-1) * multi_dot([x24,self.ubar_vbl_wheel_hub_jcl_outer_cv]))
        x26 = (-1) * I1

        self.pos_eq_blocks = ((x0 + x1 + multi_dot([x3,self.ubar_rbr_inner_shaft_jcr_diff_joint]) + (-1) * multi_dot([x4,self.ubar_vbs_differential_jcr_diff_joint])),
        multi_dot([self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,0:1].T,x5,x4,x6]),
        multi_dot([self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,1:2].T,x5,x4,x6]),
        (x0 + (-1) * x7 + multi_dot([x3,self.ubar_rbr_inner_shaft_jcr_inner_cv]) + (-1) * multi_dot([x9,self.ubar_rbr_coupling_jcr_inner_cv])),
        multi_dot([self.Mbar_rbr_inner_shaft_jcr_inner_cv[:,0:1].T,x5,x9,self.Mbar_rbr_coupling_jcr_inner_cv[:,0:1]]),
        (x10 + x1 + multi_dot([x12,self.ubar_rbl_inner_shaft_jcl_diff_joint]) + (-1) * multi_dot([x4,self.ubar_vbs_differential_jcl_diff_joint])),
        multi_dot([self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,0:1].T,x13,x4,x14]),
        multi_dot([self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,1:2].T,x13,x4,x14]),
        (x10 + (-1) * x15 + multi_dot([x12,self.ubar_rbl_inner_shaft_jcl_inner_cv]) + (-1) * multi_dot([x17,self.ubar_rbl_coupling_jcl_inner_cv])),
        multi_dot([self.Mbar_rbl_inner_shaft_jcl_inner_cv[:,0:1].T,x13,x17,self.Mbar_rbl_coupling_jcl_inner_cv[:,0:1]]),
        multi_dot([x18,x19,x20,self.Mbar_vbr_wheel_hub_jcr_outer_cv[:,0:1]]),
        multi_dot([x18,x19,x21]),
        multi_dot([self.Mbar_rbr_coupling_jcr_outer_cv[:,1:2].T,x19,x21]),
        multi_dot([x22,x23,x24,self.Mbar_vbl_wheel_hub_jcl_outer_cv[:,0:1]]),
        multi_dot([x22,x23,x25]),
        multi_dot([self.Mbar_rbl_coupling_jcl_outer_cv[:,1:2].T,x23,x25]),
        (x26 + multi_dot([x2.T,x2])),
        (x26 + multi_dot([x11.T,x11])),
        (x26 + multi_dot([x8.T,x8])),
        (x26 + multi_dot([x16.T,x16])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = Z3x1
        v1 = Z1x1

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
        v1,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbr_inner_shaft
        a1 = self.Pd_vbs_differential
        a2 = self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,0:1]
        a3 = self.P_rbr_inner_shaft
        a4 = A(a3).T
        a5 = self.Mbar_vbs_differential_jcr_diff_joint[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_vbs_differential
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,1:2]
        a13 = self.Pd_rbr_coupling
        a14 = self.Mbar_rbr_inner_shaft_jcr_inner_cv[:,0:1]
        a15 = self.Mbar_rbr_coupling_jcr_inner_cv[:,0:1]
        a16 = self.P_rbr_coupling
        a17 = A(a16).T
        a18 = self.Pd_rbl_inner_shaft
        a19 = self.Mbar_vbs_differential_jcl_diff_joint[:,2:3]
        a20 = a19.T
        a21 = self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,0:1]
        a22 = self.P_rbl_inner_shaft
        a23 = A(a22).T
        a24 = B(a1,a19)
        a25 = a18.T
        a26 = B(a8,a19)
        a27 = self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,1:2]
        a28 = self.Pd_rbl_coupling
        a29 = self.Mbar_rbl_inner_shaft_jcl_inner_cv[:,0:1]
        a30 = self.Mbar_rbl_coupling_jcl_inner_cv[:,0:1]
        a31 = self.P_rbl_coupling
        a32 = A(a31).T
        a33 = self.Mbar_vbr_wheel_hub_jcr_outer_cv[:,0:1]
        a34 = self.P_vbr_wheel_hub
        a35 = A(a34).T
        a36 = self.Mbar_rbr_coupling_jcr_outer_cv[:,0:1]
        a37 = B(a13,a36)
        a38 = a36.T
        a39 = self.Pd_vbr_wheel_hub
        a40 = a13.T
        a41 = B(a16,a36).T
        a42 = self.ubar_rbr_coupling_jcr_outer_cv
        a43 = self.ubar_vbr_wheel_hub_jcr_outer_cv
        a44 = (multi_dot([B(a13,a42),a13]) + (-1) * multi_dot([B(a39,a43),a39]))
        a45 = (self.Rd_rbr_coupling + (-1) * self.Rd_vbr_wheel_hub + multi_dot([B(a16,a42),a13]) + (-1) * multi_dot([B(a34,a43),a39]))
        a46 = (self.R_rbr_coupling.T + (-1) * self.R_vbr_wheel_hub.T + multi_dot([a42.T,a17]) + (-1) * multi_dot([a43.T,a35]))
        a47 = self.Mbar_rbr_coupling_jcr_outer_cv[:,1:2]
        a48 = self.Mbar_rbl_coupling_jcl_outer_cv[:,0:1]
        a49 = a48.T
        a50 = self.Pd_vbl_wheel_hub
        a51 = self.Mbar_vbl_wheel_hub_jcl_outer_cv[:,0:1]
        a52 = self.P_vbl_wheel_hub
        a53 = A(a52).T
        a54 = B(a28,a48)
        a55 = a28.T
        a56 = B(a31,a48).T
        a57 = self.ubar_rbl_coupling_jcl_outer_cv
        a58 = self.ubar_vbl_wheel_hub_jcl_outer_cv
        a59 = (multi_dot([B(a28,a57),a28]) + (-1) * multi_dot([B(a50,a58),a50]))
        a60 = (self.Rd_rbl_coupling + (-1) * self.Rd_vbl_wheel_hub + multi_dot([B(a31,a57),a28]) + (-1) * multi_dot([B(a52,a58),a50]))
        a61 = (self.R_rbl_coupling.T + (-1) * self.R_vbl_wheel_hub.T + multi_dot([a57.T,a32]) + (-1) * multi_dot([a58.T,a53]))
        a62 = self.Mbar_rbl_coupling_jcl_outer_cv[:,1:2]

        self.acc_eq_blocks = ((multi_dot([B(a0,self.ubar_rbr_inner_shaft_jcr_diff_joint),a0]) + (-1) * multi_dot([B(a1,self.ubar_vbs_differential_jcr_diff_joint),a1])),
        (multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + (2) * multi_dot([a10,B(a3,a2).T,a11,a1])),
        (multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + (2) * multi_dot([a10,B(a3,a12).T,a11,a1])),
        (multi_dot([B(a0,self.ubar_rbr_inner_shaft_jcr_inner_cv),a0]) + (-1) * multi_dot([B(a13,self.ubar_rbr_coupling_jcr_inner_cv),a13])),
        (multi_dot([a14.T,a4,B(a13,a15),a13]) + multi_dot([a15.T,a17,B(a0,a14),a0]) + (2) * multi_dot([a10,B(a3,a14).T,B(a16,a15),a13])),
        (multi_dot([B(a18,self.ubar_rbl_inner_shaft_jcl_diff_joint),a18]) + (-1) * multi_dot([B(a1,self.ubar_vbs_differential_jcl_diff_joint),a1])),
        (multi_dot([a20,a9,B(a18,a21),a18]) + multi_dot([a21.T,a23,a24,a1]) + (2) * multi_dot([a25,B(a22,a21).T,a26,a1])),
        (multi_dot([a20,a9,B(a18,a27),a18]) + multi_dot([a27.T,a23,a24,a1]) + (2) * multi_dot([a25,B(a22,a27).T,a26,a1])),
        (multi_dot([B(a18,self.ubar_rbl_inner_shaft_jcl_inner_cv),a18]) + (-1) * multi_dot([B(a28,self.ubar_rbl_coupling_jcl_inner_cv),a28])),
        (multi_dot([a29.T,a23,B(a28,a30),a28]) + multi_dot([a30.T,a32,B(a18,a29),a18]) + (2) * multi_dot([a25,B(a22,a29).T,B(a31,a30),a28])),
        (multi_dot([a33.T,a35,a37,a13]) + multi_dot([a38,a17,B(a39,a33),a39]) + (2) * multi_dot([a40,a41,B(a34,a33),a39])),
        (multi_dot([a38,a17,a44]) + (2) * multi_dot([a40,a41,a45]) + multi_dot([a46,a37,a13])),
        (multi_dot([a47.T,a17,a44]) + (2) * multi_dot([a40,B(a16,a47).T,a45]) + multi_dot([a46,B(a13,a47),a13])),
        (multi_dot([a49,a32,B(a50,a51),a50]) + multi_dot([a51.T,a53,a54,a28]) + (2) * multi_dot([a55,a56,B(a52,a51),a50])),
        (multi_dot([a49,a32,a59]) + (2) * multi_dot([a55,a56,a60]) + multi_dot([a61,a54,a28])),
        (multi_dot([a62.T,a32,a59]) + (2) * multi_dot([a55,B(a31,a62).T,a60]) + multi_dot([a61,B(a28,a62),a28])),
        (2) * multi_dot([a10,a0]),
        (2) * multi_dot([a25,a18]),
        (2) * multi_dot([a40,a13]),
        (2) * multi_dot([a55,a28]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = I3
        j1 = self.P_rbr_inner_shaft
        j2 = Z1x3
        j3 = self.Mbar_vbs_differential_jcr_diff_joint[:,2:3]
        j4 = j3.T
        j5 = self.P_vbs_differential
        j6 = A(j5).T
        j7 = self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,0:1]
        j8 = self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,1:2]
        j9 = (-1) * j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.Mbar_rbr_coupling_jcr_inner_cv[:,0:1]
        j13 = self.P_rbr_coupling
        j14 = A(j13).T
        j15 = self.Mbar_rbr_inner_shaft_jcr_inner_cv[:,0:1]
        j16 = self.P_rbl_inner_shaft
        j17 = self.Mbar_vbs_differential_jcl_diff_joint[:,2:3]
        j18 = j17.T
        j19 = self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,0:1]
        j20 = self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,1:2]
        j21 = A(j16).T
        j22 = B(j5,j17)
        j23 = self.Mbar_rbl_coupling_jcl_inner_cv[:,0:1]
        j24 = self.P_rbl_coupling
        j25 = A(j24).T
        j26 = self.Mbar_rbl_inner_shaft_jcl_inner_cv[:,0:1]
        j27 = self.Mbar_vbr_wheel_hub_jcr_outer_cv[:,0:1]
        j28 = self.P_vbr_wheel_hub
        j29 = A(j28).T
        j30 = self.Mbar_rbr_coupling_jcr_outer_cv[:,0:1]
        j31 = B(j13,j30)
        j32 = j30.T
        j33 = multi_dot([j32,j14])
        j34 = self.ubar_rbr_coupling_jcr_outer_cv
        j35 = B(j13,j34)
        j36 = self.ubar_vbr_wheel_hub_jcr_outer_cv
        j37 = (self.R_rbr_coupling.T + (-1) * self.R_vbr_wheel_hub.T + multi_dot([j34.T,j14]) + (-1) * multi_dot([j36.T,j29]))
        j38 = self.Mbar_rbr_coupling_jcr_outer_cv[:,1:2]
        j39 = j38.T
        j40 = multi_dot([j39,j14])
        j41 = B(j28,j36)
        j42 = self.Mbar_vbl_wheel_hub_jcl_outer_cv[:,0:1]
        j43 = self.P_vbl_wheel_hub
        j44 = A(j43).T
        j45 = self.Mbar_rbl_coupling_jcl_outer_cv[:,0:1]
        j46 = B(j24,j45)
        j47 = j45.T
        j48 = multi_dot([j47,j25])
        j49 = self.ubar_rbl_coupling_jcl_outer_cv
        j50 = B(j24,j49)
        j51 = self.ubar_vbl_wheel_hub_jcl_outer_cv
        j52 = (self.R_rbl_coupling.T + (-1) * self.R_vbl_wheel_hub.T + multi_dot([j49.T,j25]) + (-1) * multi_dot([j51.T,j44]))
        j53 = self.Mbar_rbl_coupling_jcl_outer_cv[:,1:2]
        j54 = j53.T
        j55 = multi_dot([j54,j25])
        j56 = B(j43,j51)

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
        (-1) * B(j13,self.ubar_rbr_coupling_jcr_inner_cv),
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
        (-1) * B(j24,self.ubar_rbl_coupling_jcl_inner_cv),
        j2,
        multi_dot([j23.T,j25,B(j16,j26)]),
        j2,
        multi_dot([j26.T,j21,B(j24,j23)]),
        j2,
        multi_dot([j27.T,j29,j31]),
        j2,
        multi_dot([j32,j14,B(j28,j27)]),
        j33,
        (multi_dot([j32,j14,j35]) + multi_dot([j37,j31])),
        (-1) * j33,
        (-1) * multi_dot([j32,j14,j41]),
        j40,
        (multi_dot([j39,j14,j35]) + multi_dot([j37,B(j13,j38)])),
        (-1) * j40,
        (-1) * multi_dot([j39,j14,j41]),
        j2,
        multi_dot([j42.T,j44,j46]),
        j2,
        multi_dot([j47,j25,B(j43,j42)]),
        j48,
        (multi_dot([j47,j25,j50]) + multi_dot([j52,j46])),
        (-1) * j48,
        (-1) * multi_dot([j47,j25,j56]),
        j55,
        (multi_dot([j54,j25,j50]) + multi_dot([j52,B(j24,j53)])),
        (-1) * j55,
        (-1) * multi_dot([j54,j25,j56]),
        j2,
        (2) * j1.T,
        j2,
        (2) * j16.T,
        j2,
        (2) * j13.T,
        j2,
        (2) * j24.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = I3
        m1 = G(self.P_rbr_inner_shaft)
        m2 = G(self.P_rbl_inner_shaft)
        m3 = G(self.P_rbr_coupling)
        m4 = G(self.P_rbl_coupling)

        self.mass_eq_blocks = (config.m_rbr_inner_shaft * m0,
        (4) * multi_dot([m1.T,config.Jbar_rbr_inner_shaft,m1]),
        config.m_rbl_inner_shaft * m0,
        (4) * multi_dot([m2.T,config.Jbar_rbl_inner_shaft,m2]),
        config.m_rbr_coupling * m0,
        (4) * multi_dot([m3.T,config.Jbar_rbr_coupling,m3]),
        config.m_rbl_coupling * m0,
        (4) * multi_dot([m4.T,config.Jbar_rbl_coupling,m4]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = Z3x1
        f1 = self.P_rbr_inner_shaft
        f2 = G(self.Pd_rbr_inner_shaft)
        f3 = self.P_rbl_inner_shaft
        f4 = G(self.Pd_rbl_inner_shaft)
        f5 = G(self.Pd_rbr_coupling)
        f6 = G(self.Pd_rbl_coupling)

        self.frc_eq_blocks = ((self.F_rbr_inner_shaft_gravity + f0),
        ((2 * config.UF_far_drive(t)) * multi_dot([G(f1).T,self.vbar_rbr_inner_shaft_far_drive]) + (8) * multi_dot([f2.T,config.Jbar_rbr_inner_shaft,f2,f1])),
        (self.F_rbl_inner_shaft_gravity + f0),
        ((2 * config.UF_fal_drive(t)) * multi_dot([G(f3).T,self.vbar_rbl_inner_shaft_fal_drive]) + (8) * multi_dot([f4.T,config.Jbar_rbl_inner_shaft,f4,f3])),
        self.F_rbr_coupling_gravity,
        (8) * multi_dot([f5.T,config.Jbar_rbr_coupling,f5,self.P_rbr_coupling]),
        self.F_rbl_coupling_gravity,
        (8) * multi_dot([f6.T,config.Jbar_rbl_coupling,f6,self.P_rbl_coupling]),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbr_inner_shaft_jcr_diff_joint = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbr_inner_shaft,self.ubar_rbr_inner_shaft_jcr_diff_joint).T,multi_dot([B(self.P_rbr_inner_shaft,self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,0:1]).T,A(self.P_vbs_differential),self.Mbar_vbs_differential_jcr_diff_joint[:,2:3]]),multi_dot([B(self.P_rbr_inner_shaft,self.Mbar_rbr_inner_shaft_jcr_diff_joint[:,1:2]).T,A(self.P_vbs_differential),self.Mbar_vbs_differential_jcr_diff_joint[:,2:3]])]]),self.L_jcr_diff_joint])
        self.F_rbr_inner_shaft_jcr_diff_joint = Q_rbr_inner_shaft_jcr_diff_joint[0:3]
        Te_rbr_inner_shaft_jcr_diff_joint = Q_rbr_inner_shaft_jcr_diff_joint[3:7]
        self.T_rbr_inner_shaft_jcr_diff_joint = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_inner_shaft),self.ubar_rbr_inner_shaft_jcr_diff_joint])),self.F_rbr_inner_shaft_jcr_diff_joint]) + (0.5) * multi_dot([E(self.P_rbr_inner_shaft),Te_rbr_inner_shaft_jcr_diff_joint]))
        Q_rbr_inner_shaft_jcr_inner_cv = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbr_inner_shaft,self.ubar_rbr_inner_shaft_jcr_inner_cv).T,multi_dot([B(self.P_rbr_inner_shaft,self.Mbar_rbr_inner_shaft_jcr_inner_cv[:,0:1]).T,A(self.P_rbr_coupling),self.Mbar_rbr_coupling_jcr_inner_cv[:,0:1]])]]),self.L_jcr_inner_cv])
        self.F_rbr_inner_shaft_jcr_inner_cv = Q_rbr_inner_shaft_jcr_inner_cv[0:3]
        Te_rbr_inner_shaft_jcr_inner_cv = Q_rbr_inner_shaft_jcr_inner_cv[3:7]
        self.T_rbr_inner_shaft_jcr_inner_cv = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_inner_shaft),self.ubar_rbr_inner_shaft_jcr_inner_cv])),self.F_rbr_inner_shaft_jcr_inner_cv]) + (0.5) * multi_dot([E(self.P_rbr_inner_shaft),Te_rbr_inner_shaft_jcr_inner_cv]))
        Q_rbl_inner_shaft_jcl_diff_joint = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbl_inner_shaft,self.ubar_rbl_inner_shaft_jcl_diff_joint).T,multi_dot([B(self.P_rbl_inner_shaft,self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,0:1]).T,A(self.P_vbs_differential),self.Mbar_vbs_differential_jcl_diff_joint[:,2:3]]),multi_dot([B(self.P_rbl_inner_shaft,self.Mbar_rbl_inner_shaft_jcl_diff_joint[:,1:2]).T,A(self.P_vbs_differential),self.Mbar_vbs_differential_jcl_diff_joint[:,2:3]])]]),self.L_jcl_diff_joint])
        self.F_rbl_inner_shaft_jcl_diff_joint = Q_rbl_inner_shaft_jcl_diff_joint[0:3]
        Te_rbl_inner_shaft_jcl_diff_joint = Q_rbl_inner_shaft_jcl_diff_joint[3:7]
        self.T_rbl_inner_shaft_jcl_diff_joint = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_inner_shaft),self.ubar_rbl_inner_shaft_jcl_diff_joint])),self.F_rbl_inner_shaft_jcl_diff_joint]) + (0.5) * multi_dot([E(self.P_rbl_inner_shaft),Te_rbl_inner_shaft_jcl_diff_joint]))
        Q_rbl_inner_shaft_jcl_inner_cv = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbl_inner_shaft,self.ubar_rbl_inner_shaft_jcl_inner_cv).T,multi_dot([B(self.P_rbl_inner_shaft,self.Mbar_rbl_inner_shaft_jcl_inner_cv[:,0:1]).T,A(self.P_rbl_coupling),self.Mbar_rbl_coupling_jcl_inner_cv[:,0:1]])]]),self.L_jcl_inner_cv])
        self.F_rbl_inner_shaft_jcl_inner_cv = Q_rbl_inner_shaft_jcl_inner_cv[0:3]
        Te_rbl_inner_shaft_jcl_inner_cv = Q_rbl_inner_shaft_jcl_inner_cv[3:7]
        self.T_rbl_inner_shaft_jcl_inner_cv = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_inner_shaft),self.ubar_rbl_inner_shaft_jcl_inner_cv])),self.F_rbl_inner_shaft_jcl_inner_cv]) + (0.5) * multi_dot([E(self.P_rbl_inner_shaft),Te_rbl_inner_shaft_jcl_inner_cv]))
        Q_rbr_coupling_jcr_outer_cv = (-1) * multi_dot([np.bmat([[Z1x3.T,multi_dot([A(self.P_rbr_coupling),self.Mbar_rbr_coupling_jcr_outer_cv[:,0:1]]),multi_dot([A(self.P_rbr_coupling),self.Mbar_rbr_coupling_jcr_outer_cv[:,1:2]])],[multi_dot([B(self.P_rbr_coupling,self.Mbar_rbr_coupling_jcr_outer_cv[:,0:1]).T,A(self.P_vbr_wheel_hub),self.Mbar_vbr_wheel_hub_jcr_outer_cv[:,0:1]]),(multi_dot([B(self.P_rbr_coupling,self.Mbar_rbr_coupling_jcr_outer_cv[:,0:1]).T,((-1) * self.R_vbr_wheel_hub + multi_dot([A(self.P_rbr_coupling),self.ubar_rbr_coupling_jcr_outer_cv]) + (-1) * multi_dot([A(self.P_vbr_wheel_hub),self.ubar_vbr_wheel_hub_jcr_outer_cv]) + self.R_rbr_coupling)]) + multi_dot([B(self.P_rbr_coupling,self.ubar_rbr_coupling_jcr_outer_cv).T,A(self.P_rbr_coupling),self.Mbar_rbr_coupling_jcr_outer_cv[:,0:1]])),(multi_dot([B(self.P_rbr_coupling,self.Mbar_rbr_coupling_jcr_outer_cv[:,1:2]).T,((-1) * self.R_vbr_wheel_hub + multi_dot([A(self.P_rbr_coupling),self.ubar_rbr_coupling_jcr_outer_cv]) + (-1) * multi_dot([A(self.P_vbr_wheel_hub),self.ubar_vbr_wheel_hub_jcr_outer_cv]) + self.R_rbr_coupling)]) + multi_dot([B(self.P_rbr_coupling,self.ubar_rbr_coupling_jcr_outer_cv).T,A(self.P_rbr_coupling),self.Mbar_rbr_coupling_jcr_outer_cv[:,1:2]]))]]),self.L_jcr_outer_cv])
        self.F_rbr_coupling_jcr_outer_cv = Q_rbr_coupling_jcr_outer_cv[0:3]
        Te_rbr_coupling_jcr_outer_cv = Q_rbr_coupling_jcr_outer_cv[3:7]
        self.T_rbr_coupling_jcr_outer_cv = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_coupling),self.ubar_rbr_coupling_jcr_outer_cv])),self.F_rbr_coupling_jcr_outer_cv]) + (0.5) * multi_dot([E(self.P_rbr_coupling),Te_rbr_coupling_jcr_outer_cv]))
        Q_rbl_coupling_jcl_outer_cv = (-1) * multi_dot([np.bmat([[Z1x3.T,multi_dot([A(self.P_rbl_coupling),self.Mbar_rbl_coupling_jcl_outer_cv[:,0:1]]),multi_dot([A(self.P_rbl_coupling),self.Mbar_rbl_coupling_jcl_outer_cv[:,1:2]])],[multi_dot([B(self.P_rbl_coupling,self.Mbar_rbl_coupling_jcl_outer_cv[:,0:1]).T,A(self.P_vbl_wheel_hub),self.Mbar_vbl_wheel_hub_jcl_outer_cv[:,0:1]]),(multi_dot([B(self.P_rbl_coupling,self.Mbar_rbl_coupling_jcl_outer_cv[:,0:1]).T,((-1) * self.R_vbl_wheel_hub + multi_dot([A(self.P_rbl_coupling),self.ubar_rbl_coupling_jcl_outer_cv]) + (-1) * multi_dot([A(self.P_vbl_wheel_hub),self.ubar_vbl_wheel_hub_jcl_outer_cv]) + self.R_rbl_coupling)]) + multi_dot([B(self.P_rbl_coupling,self.ubar_rbl_coupling_jcl_outer_cv).T,A(self.P_rbl_coupling),self.Mbar_rbl_coupling_jcl_outer_cv[:,0:1]])),(multi_dot([B(self.P_rbl_coupling,self.Mbar_rbl_coupling_jcl_outer_cv[:,1:2]).T,((-1) * self.R_vbl_wheel_hub + multi_dot([A(self.P_rbl_coupling),self.ubar_rbl_coupling_jcl_outer_cv]) + (-1) * multi_dot([A(self.P_vbl_wheel_hub),self.ubar_vbl_wheel_hub_jcl_outer_cv]) + self.R_rbl_coupling)]) + multi_dot([B(self.P_rbl_coupling,self.ubar_rbl_coupling_jcl_outer_cv).T,A(self.P_rbl_coupling),self.Mbar_rbl_coupling_jcl_outer_cv[:,1:2]]))]]),self.L_jcl_outer_cv])
        self.F_rbl_coupling_jcl_outer_cv = Q_rbl_coupling_jcl_outer_cv[0:3]
        Te_rbl_coupling_jcl_outer_cv = Q_rbl_coupling_jcl_outer_cv[3:7]
        self.T_rbl_coupling_jcl_outer_cv = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_coupling),self.ubar_rbl_coupling_jcl_outer_cv])),self.F_rbl_coupling_jcl_outer_cv]) + (0.5) * multi_dot([E(self.P_rbl_coupling),Te_rbl_coupling_jcl_outer_cv]))

        self.reactions = {'F_rbr_inner_shaft_jcr_diff_joint' : self.F_rbr_inner_shaft_jcr_diff_joint,
                        'T_rbr_inner_shaft_jcr_diff_joint' : self.T_rbr_inner_shaft_jcr_diff_joint,
                        'F_rbr_inner_shaft_jcr_inner_cv' : self.F_rbr_inner_shaft_jcr_inner_cv,
                        'T_rbr_inner_shaft_jcr_inner_cv' : self.T_rbr_inner_shaft_jcr_inner_cv,
                        'F_rbl_inner_shaft_jcl_diff_joint' : self.F_rbl_inner_shaft_jcl_diff_joint,
                        'T_rbl_inner_shaft_jcl_diff_joint' : self.T_rbl_inner_shaft_jcl_diff_joint,
                        'F_rbl_inner_shaft_jcl_inner_cv' : self.F_rbl_inner_shaft_jcl_inner_cv,
                        'T_rbl_inner_shaft_jcl_inner_cv' : self.T_rbl_inner_shaft_jcl_inner_cv,
                        'F_rbr_coupling_jcr_outer_cv' : self.F_rbr_coupling_jcr_outer_cv,
                        'T_rbr_coupling_jcr_outer_cv' : self.T_rbr_coupling_jcr_outer_cv,
                        'F_rbl_coupling_jcl_outer_cv' : self.F_rbl_coupling_jcl_outer_cv,
                        'T_rbl_coupling_jcl_outer_cv' : self.T_rbl_coupling_jcl_outer_cv}

