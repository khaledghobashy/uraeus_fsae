
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

        self.indicies_map = {'vbs_ground': 0, 'rbs_coupler': 1, 'rbr_rocker': 2, 'rbl_rocker': 3, 'vbs_chassis': 4}

        self.n  = 21
        self.nc = 21
        self.nrows = 13
        self.ncols = 2*3
        self.rows = np.arange(self.nrows, dtype=np.intc)

        reactions_indicies = ['F_rbr_rocker_jcr_rocker_chassis', 'T_rbr_rocker_jcr_rocker_chassis', 'F_rbr_rocker_jcs_rocker_uni', 'T_rbr_rocker_jcs_rocker_uni', 'F_rbl_rocker_jcl_rocker_chassis', 'T_rbl_rocker_jcl_rocker_chassis', 'F_rbl_rocker_mcs_steer_act', 'T_rbl_rocker_mcs_steer_act', 'F_rbl_rocker_jcs_rocker_sph', 'T_rbl_rocker_jcs_rocker_sph']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 12, 12], dtype=np.intc)
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.rbr_rocker*2, self.rbr_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_coupler*2, self.rbs_coupler*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.rbs_coupler*2, self.rbs_coupler*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_coupler*2, self.rbs_coupler*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.rbs_coupler*2, self.rbs_coupler*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1], dtype=np.intc)

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
        np.concatenate([self.config.R_rbs_coupler,
        self.config.P_rbs_coupler,
        self.config.R_rbr_rocker,
        self.config.P_rbr_rocker,
        self.config.R_rbl_rocker,
        self.config.P_rbl_rocker], out=self._q)

        np.concatenate([self.config.Rd_rbs_coupler,
        self.config.Pd_rbs_coupler,
        self.config.Rd_rbr_rocker,
        self.config.Pd_rbr_rocker,
        self.config.Rd_rbl_rocker,
        self.config.Pd_rbl_rocker], out=self._qd)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbs_coupler = indicies_map[p + 'rbs_coupler']
        self.rbr_rocker = indicies_map[p + 'rbr_rocker']
        self.rbl_rocker = indicies_map[p + 'rbl_rocker']
        self.vbs_chassis = indicies_map[interface_map[p + 'vbs_chassis']]
        self.vbs_ground = indicies_map[interface_map[p + 'vbs_ground']]

    
    def eval_constants(self):
        config = self.config

        self.F_rbs_coupler_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_coupler]], dtype=np.float64)
        self.F_rbr_rocker_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_rocker]], dtype=np.float64)
        self.F_rbl_rocker_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_rocker]], dtype=np.float64)

        self.Mbar_rbr_rocker_jcr_rocker_chassis = multi_dot([A(config.P_rbr_rocker).T,triad(config.ax1_jcr_rocker_chassis)])
        self.Mbar_vbs_chassis_jcr_rocker_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcr_rocker_chassis)])
        self.ubar_rbr_rocker_jcr_rocker_chassis = (multi_dot([A(config.P_rbr_rocker).T,config.pt1_jcr_rocker_chassis]) + (-1) * multi_dot([A(config.P_rbr_rocker).T,config.R_rbr_rocker]))
        self.ubar_vbs_chassis_jcr_rocker_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_rocker_chassis]) + (-1) * multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbr_rocker_jcs_rocker_uni = multi_dot([A(config.P_rbr_rocker).T,triad(config.ax1_jcs_rocker_uni)])
        self.Mbar_rbs_coupler_jcs_rocker_uni = multi_dot([A(config.P_rbs_coupler).T,triad(config.ax2_jcs_rocker_uni,triad(config.ax1_jcs_rocker_uni)[0:3,1:2])])
        self.ubar_rbr_rocker_jcs_rocker_uni = (multi_dot([A(config.P_rbr_rocker).T,config.pt1_jcs_rocker_uni]) + (-1) * multi_dot([A(config.P_rbr_rocker).T,config.R_rbr_rocker]))
        self.ubar_rbs_coupler_jcs_rocker_uni = (multi_dot([A(config.P_rbs_coupler).T,config.pt1_jcs_rocker_uni]) + (-1) * multi_dot([A(config.P_rbs_coupler).T,config.R_rbs_coupler]))
        self.Mbar_rbl_rocker_jcl_rocker_chassis = multi_dot([A(config.P_rbl_rocker).T,triad(config.ax1_jcl_rocker_chassis)])
        self.Mbar_vbs_chassis_jcl_rocker_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcl_rocker_chassis)])
        self.ubar_rbl_rocker_jcl_rocker_chassis = (multi_dot([A(config.P_rbl_rocker).T,config.pt1_jcl_rocker_chassis]) + (-1) * multi_dot([A(config.P_rbl_rocker).T,config.R_rbl_rocker]))
        self.ubar_vbs_chassis_jcl_rocker_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_rocker_chassis]) + (-1) * multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_rocker_jcl_rocker_chassis = multi_dot([A(config.P_rbl_rocker).T,triad(config.ax1_jcl_rocker_chassis)])
        self.Mbar_vbs_chassis_jcl_rocker_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcl_rocker_chassis)])
        self.Mbar_rbl_rocker_jcs_rocker_sph = multi_dot([A(config.P_rbl_rocker).T,triad(config.ax1_jcs_rocker_sph)])
        self.Mbar_rbs_coupler_jcs_rocker_sph = multi_dot([A(config.P_rbs_coupler).T,triad(config.ax1_jcs_rocker_sph)])
        self.ubar_rbl_rocker_jcs_rocker_sph = (multi_dot([A(config.P_rbl_rocker).T,config.pt1_jcs_rocker_sph]) + (-1) * multi_dot([A(config.P_rbl_rocker).T,config.R_rbl_rocker]))
        self.ubar_rbs_coupler_jcs_rocker_sph = (multi_dot([A(config.P_rbs_coupler).T,config.pt1_jcs_rocker_sph]) + (-1) * multi_dot([A(config.P_rbs_coupler).T,config.R_rbs_coupler]))

    
    def _map_gen_coordinates(self):
        q = self._q
        self.R_rbs_coupler = q[0:3]
        self.P_rbs_coupler = q[3:7]
        self.R_rbr_rocker = q[7:10]
        self.P_rbr_rocker = q[10:14]
        self.R_rbl_rocker = q[14:17]
        self.P_rbl_rocker = q[17:21]

    
    def _map_gen_velocities(self):
        qd = self._qd
        self.Rd_rbs_coupler = qd[0:3]
        self.Pd_rbs_coupler = qd[3:7]
        self.Rd_rbr_rocker = qd[7:10]
        self.Pd_rbr_rocker = qd[10:14]
        self.Rd_rbl_rocker = qd[14:17]
        self.Pd_rbl_rocker = qd[17:21]

    
    def _map_gen_accelerations(self):
        qdd = self._qdd
        self.Rdd_rbs_coupler = qdd[0:3]
        self.Pdd_rbs_coupler = qdd[3:7]
        self.Rdd_rbr_rocker = qdd[7:10]
        self.Pdd_rbr_rocker = qdd[10:14]
        self.Rdd_rbl_rocker = qdd[14:17]
        self.Pdd_rbl_rocker = qdd[17:21]

    
    def _map_lagrange_multipliers(self):
        Lambda = self._lgr
        self.L_jcr_rocker_chassis = Lambda[0:5]
        self.L_jcs_rocker_uni = Lambda[5:9]
        self.L_jcl_rocker_chassis = Lambda[9:14]
        self.L_mcs_steer_act = Lambda[14:15]
        self.L_jcs_rocker_sph = Lambda[15:18]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_rbr_rocker
        x1 = (-1) * self.R_vbs_chassis
        x2 = self.P_rbr_rocker
        x3 = A(x2)
        x4 = A(self.P_vbs_chassis)
        x5 = x3.T
        x6 = self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]
        x7 = (-1) * self.R_rbs_coupler
        x8 = self.P_rbs_coupler
        x9 = A(x8)
        x10 = self.R_rbl_rocker
        x11 = self.P_rbl_rocker
        x12 = A(x11)
        x13 = x12.T
        x14 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]
        x15 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,0:1]
        x16 = (-1) * I1

        self.pos_eq_blocks = ((x0 + x1 + multi_dot([x3,self.ubar_rbr_rocker_jcr_rocker_chassis]) + (-1) * multi_dot([x4,self.ubar_vbs_chassis_jcr_rocker_chassis])),
        multi_dot([self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1].T,x5,x4,x6]),
        multi_dot([self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2].T,x5,x4,x6]),
        (x0 + x7 + multi_dot([x3,self.ubar_rbr_rocker_jcs_rocker_uni]) + (-1) * multi_dot([x9,self.ubar_rbs_coupler_jcs_rocker_uni])),
        multi_dot([self.Mbar_rbr_rocker_jcs_rocker_uni[:,0:1].T,x5,x9,self.Mbar_rbs_coupler_jcs_rocker_uni[:,0:1]]),
        (x10 + x1 + multi_dot([x12,self.ubar_rbl_rocker_jcl_rocker_chassis]) + (-1) * multi_dot([x4,self.ubar_vbs_chassis_jcl_rocker_chassis])),
        multi_dot([self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1].T,x13,x4,x14]),
        multi_dot([self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2].T,x13,x4,x14]),
        (cos(config.UF_mcs_steer_act(t)) * multi_dot([self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2].T,x13,x4,x15]) + (-1 * sin(config.UF_mcs_steer_act(t))) * multi_dot([self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1].T,x13,x4,x15])),
        (x10 + x7 + multi_dot([x12,self.ubar_rbl_rocker_jcs_rocker_sph]) + (-1) * multi_dot([x9,self.ubar_rbs_coupler_jcs_rocker_sph])),
        (x16 + multi_dot([x8.T,x8])),
        (x16 + multi_dot([x2.T,x2])),
        (x16 + multi_dot([x11.T,x11])),)

    
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
        (v1 + (-1 * derivative(config.UF_mcs_steer_act, t, 0.1, 1)) * I1),
        v0,
        v1,
        v1,
        v1,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbr_rocker
        a1 = self.Pd_vbs_chassis
        a2 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1]
        a3 = self.P_rbr_rocker
        a4 = A(a3).T
        a5 = self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_vbs_chassis
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2]
        a13 = self.Pd_rbs_coupler
        a14 = self.Mbar_rbr_rocker_jcs_rocker_uni[:,0:1]
        a15 = self.Mbar_rbs_coupler_jcs_rocker_uni[:,0:1]
        a16 = self.P_rbs_coupler
        a17 = self.Pd_rbl_rocker
        a18 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]
        a19 = self.P_rbl_rocker
        a20 = A(a19).T
        a21 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]
        a22 = B(a1,a21)
        a23 = a21.T
        a24 = a17.T
        a25 = B(a8,a21)
        a26 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]
        a27 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,0:1]
        a28 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]
        a29 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]

        self.acc_eq_blocks = ((multi_dot([B(a0,self.ubar_rbr_rocker_jcr_rocker_chassis),a0]) + (-1) * multi_dot([B(a1,self.ubar_vbs_chassis_jcr_rocker_chassis),a1])),
        (multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + (2) * multi_dot([a10,B(a3,a2).T,a11,a1])),
        (multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + (2) * multi_dot([a10,B(a3,a12).T,a11,a1])),
        (multi_dot([B(a0,self.ubar_rbr_rocker_jcs_rocker_uni),a0]) + (-1) * multi_dot([B(a13,self.ubar_rbs_coupler_jcs_rocker_uni),a13])),
        (multi_dot([a14.T,a4,B(a13,a15),a13]) + multi_dot([a15.T,A(a16).T,B(a0,a14),a0]) + (2) * multi_dot([a10,B(a3,a14).T,B(a16,a15),a13])),
        (multi_dot([B(a17,self.ubar_rbl_rocker_jcl_rocker_chassis),a17]) + (-1) * multi_dot([B(a1,self.ubar_vbs_chassis_jcl_rocker_chassis),a1])),
        (multi_dot([a18.T,a20,a22,a1]) + multi_dot([a23,a9,B(a17,a18),a17]) + (2) * multi_dot([a24,B(a19,a18).T,a25,a1])),
        (multi_dot([a26.T,a20,a22,a1]) + multi_dot([a23,a9,B(a17,a26),a17]) + (2) * multi_dot([a24,B(a19,a26).T,a25,a1])),
        ((-1 * derivative(config.UF_mcs_steer_act, t, 0.1, 2)) * I1 + multi_dot([a27.T,a9,(cos(config.UF_mcs_steer_act(t)) * B(a17,a28) + (-1 * sin(config.UF_mcs_steer_act(t))) * B(a17,a29)),a17]) + multi_dot([(cos(config.UF_mcs_steer_act(t)) * multi_dot([a28.T,a20]) + (-1 * sin(config.UF_mcs_steer_act(t))) * multi_dot([a29.T,a20])),B(a1,a27),a1]) + (2) * multi_dot([(cos(config.UF_mcs_steer_act(t)) * multi_dot([a24,B(a19,a28).T]) + (-1 * sin(config.UF_mcs_steer_act(t))) * multi_dot([a24,B(a19,a29).T])),B(a8,a27),a1])),
        (multi_dot([B(a17,self.ubar_rbl_rocker_jcs_rocker_sph),a17]) + (-1) * multi_dot([B(a13,self.ubar_rbs_coupler_jcs_rocker_sph),a13])),
        (2) * multi_dot([a13.T,a13]),
        (2) * multi_dot([a10,a0]),
        (2) * multi_dot([a24,a17]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = I3
        j1 = self.P_rbr_rocker
        j2 = Z1x3
        j3 = self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]
        j4 = j3.T
        j5 = self.P_vbs_chassis
        j6 = A(j5).T
        j7 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1]
        j8 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2]
        j9 = (-1) * j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.Mbar_rbs_coupler_jcs_rocker_uni[:,0:1]
        j13 = self.P_rbs_coupler
        j14 = self.Mbar_rbr_rocker_jcs_rocker_uni[:,0:1]
        j15 = self.P_rbl_rocker
        j16 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]
        j17 = j16.T
        j18 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]
        j19 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]
        j20 = A(j15).T
        j21 = B(j5,j16)
        j22 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,0:1]
        j23 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]
        j24 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]

        self.jac_eq_blocks = (j0,
        B(j1,self.ubar_rbr_rocker_jcr_rocker_chassis),
        j9,
        (-1) * B(j5,self.ubar_vbs_chassis_jcr_rocker_chassis),
        j2,
        multi_dot([j4,j6,B(j1,j7)]),
        j2,
        multi_dot([j7.T,j10,j11]),
        j2,
        multi_dot([j4,j6,B(j1,j8)]),
        j2,
        multi_dot([j8.T,j10,j11]),
        j9,
        (-1) * B(j13,self.ubar_rbs_coupler_jcs_rocker_uni),
        j0,
        B(j1,self.ubar_rbr_rocker_jcs_rocker_uni),
        j2,
        multi_dot([j14.T,j10,B(j13,j12)]),
        j2,
        multi_dot([j12.T,A(j13).T,B(j1,j14)]),
        j0,
        B(j15,self.ubar_rbl_rocker_jcl_rocker_chassis),
        j9,
        (-1) * B(j5,self.ubar_vbs_chassis_jcl_rocker_chassis),
        j2,
        multi_dot([j17,j6,B(j15,j18)]),
        j2,
        multi_dot([j18.T,j20,j21]),
        j2,
        multi_dot([j17,j6,B(j15,j19)]),
        j2,
        multi_dot([j19.T,j20,j21]),
        j2,
        multi_dot([j22.T,j6,(cos(config.UF_mcs_steer_act(t)) * B(j15,j23) + (-1 * sin(config.UF_mcs_steer_act(t))) * B(j15,j24))]),
        j2,
        multi_dot([(cos(config.UF_mcs_steer_act(t)) * multi_dot([j23.T,j20]) + (-1 * sin(config.UF_mcs_steer_act(t))) * multi_dot([j24.T,j20])),B(j5,j22)]),
        j9,
        (-1) * B(j13,self.ubar_rbs_coupler_jcs_rocker_sph),
        j0,
        B(j15,self.ubar_rbl_rocker_jcs_rocker_sph),
        j2,
        (2) * j13.T,
        j2,
        (2) * j1.T,
        j2,
        (2) * j15.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = I3
        m1 = G(self.P_rbs_coupler)
        m2 = G(self.P_rbr_rocker)
        m3 = G(self.P_rbl_rocker)

        self.mass_eq_blocks = (config.m_rbs_coupler * m0,
        (4) * multi_dot([m1.T,config.Jbar_rbs_coupler,m1]),
        config.m_rbr_rocker * m0,
        (4) * multi_dot([m2.T,config.Jbar_rbr_rocker,m2]),
        config.m_rbl_rocker * m0,
        (4) * multi_dot([m3.T,config.Jbar_rbl_rocker,m3]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_coupler)
        f1 = G(self.Pd_rbr_rocker)
        f2 = G(self.Pd_rbl_rocker)

        self.frc_eq_blocks = (self.F_rbs_coupler_gravity,
        (8) * multi_dot([f0.T,config.Jbar_rbs_coupler,f0,self.P_rbs_coupler]),
        self.F_rbr_rocker_gravity,
        (8) * multi_dot([f1.T,config.Jbar_rbr_rocker,f1,self.P_rbr_rocker]),
        self.F_rbl_rocker_gravity,
        (8) * multi_dot([f2.T,config.Jbar_rbl_rocker,f2,self.P_rbl_rocker]),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbr_rocker_jcr_rocker_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbr_rocker,self.ubar_rbr_rocker_jcr_rocker_chassis).T,multi_dot([B(self.P_rbr_rocker,self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]]),multi_dot([B(self.P_rbr_rocker,self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]])]]),self.L_jcr_rocker_chassis])
        self.F_rbr_rocker_jcr_rocker_chassis = Q_rbr_rocker_jcr_rocker_chassis[0:3]
        Te_rbr_rocker_jcr_rocker_chassis = Q_rbr_rocker_jcr_rocker_chassis[3:7]
        self.T_rbr_rocker_jcr_rocker_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_rocker),self.ubar_rbr_rocker_jcr_rocker_chassis])),self.F_rbr_rocker_jcr_rocker_chassis]) + (0.5) * multi_dot([E(self.P_rbr_rocker),Te_rbr_rocker_jcr_rocker_chassis]))
        Q_rbr_rocker_jcs_rocker_uni = (-1) * multi_dot([np.bmat([[I3,Z1x3.T],[B(self.P_rbr_rocker,self.ubar_rbr_rocker_jcs_rocker_uni).T,multi_dot([B(self.P_rbr_rocker,self.Mbar_rbr_rocker_jcs_rocker_uni[:,0:1]).T,A(self.P_rbs_coupler),self.Mbar_rbs_coupler_jcs_rocker_uni[:,0:1]])]]),self.L_jcs_rocker_uni])
        self.F_rbr_rocker_jcs_rocker_uni = Q_rbr_rocker_jcs_rocker_uni[0:3]
        Te_rbr_rocker_jcs_rocker_uni = Q_rbr_rocker_jcs_rocker_uni[3:7]
        self.T_rbr_rocker_jcs_rocker_uni = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbr_rocker),self.ubar_rbr_rocker_jcs_rocker_uni])),self.F_rbr_rocker_jcs_rocker_uni]) + (0.5) * multi_dot([E(self.P_rbr_rocker),Te_rbr_rocker_jcs_rocker_uni]))
        Q_rbl_rocker_jcl_rocker_chassis = (-1) * multi_dot([np.bmat([[I3,Z1x3.T,Z1x3.T],[B(self.P_rbl_rocker,self.ubar_rbl_rocker_jcl_rocker_chassis).T,multi_dot([B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]]),multi_dot([B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]])]]),self.L_jcl_rocker_chassis])
        self.F_rbl_rocker_jcl_rocker_chassis = Q_rbl_rocker_jcl_rocker_chassis[0:3]
        Te_rbl_rocker_jcl_rocker_chassis = Q_rbl_rocker_jcl_rocker_chassis[3:7]
        self.T_rbl_rocker_jcl_rocker_chassis = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_rocker),self.ubar_rbl_rocker_jcl_rocker_chassis])),self.F_rbl_rocker_jcl_rocker_chassis]) + (0.5) * multi_dot([E(self.P_rbl_rocker),Te_rbl_rocker_jcl_rocker_chassis]))
        Q_rbl_rocker_mcs_steer_act = (-1) * multi_dot([np.bmat([[Z1x3.T],[multi_dot([((-1 * sin(config.UF_mcs_steer_act(t))) * B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]).T + cos(config.UF_mcs_steer_act(t)) * B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]).T),A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_rocker_chassis[:,0:1]])]]),self.L_mcs_steer_act])
        self.F_rbl_rocker_mcs_steer_act = Q_rbl_rocker_mcs_steer_act[0:3]
        Te_rbl_rocker_mcs_steer_act = Q_rbl_rocker_mcs_steer_act[3:7]
        self.T_rbl_rocker_mcs_steer_act = (0.5) * multi_dot([E(self.P_rbl_rocker),Te_rbl_rocker_mcs_steer_act])
        Q_rbl_rocker_jcs_rocker_sph = (-1) * multi_dot([np.bmat([[I3],[B(self.P_rbl_rocker,self.ubar_rbl_rocker_jcs_rocker_sph).T]]),self.L_jcs_rocker_sph])
        self.F_rbl_rocker_jcs_rocker_sph = Q_rbl_rocker_jcs_rocker_sph[0:3]
        Te_rbl_rocker_jcs_rocker_sph = Q_rbl_rocker_jcs_rocker_sph[3:7]
        self.T_rbl_rocker_jcs_rocker_sph = ((-1) * multi_dot([skew(multi_dot([A(self.P_rbl_rocker),self.ubar_rbl_rocker_jcs_rocker_sph])),self.F_rbl_rocker_jcs_rocker_sph]) + (0.5) * multi_dot([E(self.P_rbl_rocker),Te_rbl_rocker_jcs_rocker_sph]))

        self.reactions = {'F_rbr_rocker_jcr_rocker_chassis' : self.F_rbr_rocker_jcr_rocker_chassis,
                        'T_rbr_rocker_jcr_rocker_chassis' : self.T_rbr_rocker_jcr_rocker_chassis,
                        'F_rbr_rocker_jcs_rocker_uni' : self.F_rbr_rocker_jcs_rocker_uni,
                        'T_rbr_rocker_jcs_rocker_uni' : self.T_rbr_rocker_jcs_rocker_uni,
                        'F_rbl_rocker_jcl_rocker_chassis' : self.F_rbl_rocker_jcl_rocker_chassis,
                        'T_rbl_rocker_jcl_rocker_chassis' : self.T_rbl_rocker_jcl_rocker_chassis,
                        'F_rbl_rocker_mcs_steer_act' : self.F_rbl_rocker_mcs_steer_act,
                        'T_rbl_rocker_mcs_steer_act' : self.T_rbl_rocker_mcs_steer_act,
                        'F_rbl_rocker_jcs_rocker_sph' : self.F_rbl_rocker_jcs_rocker_sph,
                        'T_rbl_rocker_jcs_rocker_sph' : self.T_rbl_rocker_jcs_rocker_sph}

