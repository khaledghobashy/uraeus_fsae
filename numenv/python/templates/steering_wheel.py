
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

        self.indicies_map = {'vbs_ground': 0, 'rbs_input_shaft': 1, 'rbs_connect_shaft': 2, 'rbs_output_shaft': 3, 'vbs_chassis': 4}

        self.n  = 21
        self.nc = 21
        self.nrows = 15
        self.ncols = 2*3
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_rbs_input_shaft_jcs_input_bearing', 'T_rbs_input_shaft_jcs_input_bearing', 'F_rbs_input_shaft_mcs_hand_wheel', 'T_rbs_input_shaft_mcs_hand_wheel', 'F_rbs_input_shaft_jcs_input_connect', 'T_rbs_input_shaft_jcs_input_connect', 'F_rbs_output_shaft_jcs_output_connect', 'T_rbs_output_shaft_jcs_output_connect', 'F_rbs_output_shaft_jcs_output_bearing', 'T_rbs_output_shaft_jcs_output_bearing']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 13, 13, 14, 14])
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.rbs_input_shaft*2, self.rbs_input_shaft*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_input_shaft*2, self.rbs_input_shaft*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_input_shaft*2, self.rbs_input_shaft*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_input_shaft*2, self.rbs_input_shaft*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_input_shaft*2, self.rbs_input_shaft*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_input_shaft*2, self.rbs_input_shaft*2+1, self.rbs_connect_shaft*2, self.rbs_connect_shaft*2+1, self.rbs_input_shaft*2, self.rbs_input_shaft*2+1, self.rbs_connect_shaft*2, self.rbs_connect_shaft*2+1, self.rbs_connect_shaft*2, self.rbs_connect_shaft*2+1, self.rbs_output_shaft*2, self.rbs_output_shaft*2+1, self.rbs_connect_shaft*2, self.rbs_connect_shaft*2+1, self.rbs_output_shaft*2, self.rbs_output_shaft*2+1, self.rbs_output_shaft*2, self.rbs_output_shaft*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_output_shaft*2, self.rbs_output_shaft*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_output_shaft*2, self.rbs_output_shaft*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_input_shaft*2, self.rbs_input_shaft*2+1, self.rbs_connect_shaft*2, self.rbs_connect_shaft*2+1, self.rbs_output_shaft*2, self.rbs_output_shaft*2+1])

    def set_initial_states(self):
        self.q0  = np.concatenate([self.config.R_rbs_input_shaft,
        self.config.P_rbs_input_shaft,
        self.config.R_rbs_connect_shaft,
        self.config.P_rbs_connect_shaft,
        self.config.R_rbs_output_shaft,
        self.config.P_rbs_output_shaft])
        self.qd0 = np.concatenate([self.config.Rd_rbs_input_shaft,
        self.config.Pd_rbs_input_shaft,
        self.config.Rd_rbs_connect_shaft,
        self.config.Pd_rbs_connect_shaft,
        self.config.Rd_rbs_output_shaft,
        self.config.Pd_rbs_output_shaft])

        self.set_gen_coordinates(self.q0)
        self.set_gen_velocities(self.qd0)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbs_input_shaft = indicies_map[p + 'rbs_input_shaft']
        self.rbs_connect_shaft = indicies_map[p + 'rbs_connect_shaft']
        self.rbs_output_shaft = indicies_map[p + 'rbs_output_shaft']
        self.vbs_chassis = indicies_map[interface_map[p + 'vbs_chassis']]
        self.vbs_ground = indicies_map[interface_map[p + 'vbs_ground']]

    
    def eval_constants(self):
        config = self.config

        self.F_rbs_input_shaft_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_input_shaft]], dtype=np.float64)
        self.F_rbs_connect_shaft_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_connect_shaft]], dtype=np.float64)
        self.F_rbs_output_shaft_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_output_shaft]], dtype=np.float64)

        self.Mbar_rbs_input_shaft_jcs_input_bearing = multi_dot([A(config.P_rbs_input_shaft).T,triad(config.ax1_jcs_input_bearing)])
        self.Mbar_vbs_chassis_jcs_input_bearing = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcs_input_bearing)])
        self.ubar_rbs_input_shaft_jcs_input_bearing = (multi_dot([A(config.P_rbs_input_shaft).T,config.pt1_jcs_input_bearing]) + -1*multi_dot([A(config.P_rbs_input_shaft).T,config.R_rbs_input_shaft]))
        self.ubar_vbs_chassis_jcs_input_bearing = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcs_input_bearing]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbs_input_shaft_jcs_input_bearing = multi_dot([A(config.P_rbs_input_shaft).T,triad(config.ax1_jcs_input_bearing)])
        self.Mbar_vbs_chassis_jcs_input_bearing = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcs_input_bearing)])
        self.Mbar_rbs_input_shaft_jcs_input_connect = multi_dot([A(config.P_rbs_input_shaft).T,triad(config.ax1_jcs_input_connect)])
        self.Mbar_rbs_connect_shaft_jcs_input_connect = multi_dot([A(config.P_rbs_connect_shaft).T,triad(config.ax2_jcs_input_connect,triad(config.ax1_jcs_input_connect)[0:3,1:2])])
        self.ubar_rbs_input_shaft_jcs_input_connect = (multi_dot([A(config.P_rbs_input_shaft).T,config.pt1_jcs_input_connect]) + -1*multi_dot([A(config.P_rbs_input_shaft).T,config.R_rbs_input_shaft]))
        self.ubar_rbs_connect_shaft_jcs_input_connect = (multi_dot([A(config.P_rbs_connect_shaft).T,config.pt1_jcs_input_connect]) + -1*multi_dot([A(config.P_rbs_connect_shaft).T,config.R_rbs_connect_shaft]))
        self.Mbar_rbs_output_shaft_jcs_output_connect = multi_dot([A(config.P_rbs_output_shaft).T,triad(config.ax1_jcs_output_connect)])
        self.Mbar_rbs_connect_shaft_jcs_output_connect = multi_dot([A(config.P_rbs_connect_shaft).T,triad(config.ax2_jcs_output_connect,triad(config.ax1_jcs_output_connect)[0:3,1:2])])
        self.ubar_rbs_output_shaft_jcs_output_connect = (multi_dot([A(config.P_rbs_output_shaft).T,config.pt1_jcs_output_connect]) + -1*multi_dot([A(config.P_rbs_output_shaft).T,config.R_rbs_output_shaft]))
        self.ubar_rbs_connect_shaft_jcs_output_connect = (multi_dot([A(config.P_rbs_connect_shaft).T,config.pt1_jcs_output_connect]) + -1*multi_dot([A(config.P_rbs_connect_shaft).T,config.R_rbs_connect_shaft]))
        self.Mbar_rbs_output_shaft_jcs_output_bearing = multi_dot([A(config.P_rbs_output_shaft).T,triad(config.ax1_jcs_output_bearing)])
        self.Mbar_vbs_chassis_jcs_output_bearing = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcs_output_bearing)])
        self.ubar_rbs_output_shaft_jcs_output_bearing = (multi_dot([A(config.P_rbs_output_shaft).T,config.pt1_jcs_output_bearing]) + -1*multi_dot([A(config.P_rbs_output_shaft).T,config.R_rbs_output_shaft]))
        self.ubar_vbs_chassis_jcs_output_bearing = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcs_output_bearing]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))

    
    def set_gen_coordinates(self,q):
        self.R_rbs_input_shaft = q[0:3,0:1]
        self.P_rbs_input_shaft = q[3:7,0:1]
        self.R_rbs_connect_shaft = q[7:10,0:1]
        self.P_rbs_connect_shaft = q[10:14,0:1]
        self.R_rbs_output_shaft = q[14:17,0:1]
        self.P_rbs_output_shaft = q[17:21,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbs_input_shaft = qd[0:3,0:1]
        self.Pd_rbs_input_shaft = qd[3:7,0:1]
        self.Rd_rbs_connect_shaft = qd[7:10,0:1]
        self.Pd_rbs_connect_shaft = qd[10:14,0:1]
        self.Rd_rbs_output_shaft = qd[14:17,0:1]
        self.Pd_rbs_output_shaft = qd[17:21,0:1]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_rbs_input_shaft = qdd[0:3,0:1]
        self.Pdd_rbs_input_shaft = qdd[3:7,0:1]
        self.Rdd_rbs_connect_shaft = qdd[7:10,0:1]
        self.Pdd_rbs_connect_shaft = qdd[10:14,0:1]
        self.Rdd_rbs_output_shaft = qdd[14:17,0:1]
        self.Pdd_rbs_output_shaft = qdd[17:21,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcs_input_bearing = Lambda[0:4,0:1]
        self.L_mcs_hand_wheel = Lambda[4:5,0:1]
        self.L_jcs_input_connect = Lambda[5:9,0:1]
        self.L_jcs_output_connect = Lambda[9:13,0:1]
        self.L_jcs_output_bearing = Lambda[13:18,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.Mbar_rbs_input_shaft_jcs_input_bearing[:,0:1].T
        x1 = self.P_rbs_input_shaft
        x2 = A(x1)
        x3 = x2.T
        x4 = A(self.P_vbs_chassis)
        x5 = self.Mbar_vbs_chassis_jcs_input_bearing[:,2:3]
        x6 = self.Mbar_rbs_input_shaft_jcs_input_bearing[:,1:2].T
        x7 = self.R_rbs_input_shaft
        x8 = -1*self.R_vbs_chassis
        x9 = (x7 + x8 + multi_dot([x2,self.ubar_rbs_input_shaft_jcs_input_bearing]) + -1*multi_dot([x4,self.ubar_vbs_chassis_jcs_input_bearing]))
        x10 = self.Mbar_vbs_chassis_jcs_input_bearing[:,0:1]
        x11 = -1*self.R_rbs_connect_shaft
        x12 = self.P_rbs_connect_shaft
        x13 = A(x12)
        x14 = self.R_rbs_output_shaft
        x15 = self.P_rbs_output_shaft
        x16 = A(x15)
        x17 = x16.T
        x18 = self.Mbar_vbs_chassis_jcs_output_bearing[:,2:3]
        x19 = -1*np.eye(1, dtype=np.float64)

        self.pos_eq_blocks = (multi_dot([x0,x3,x4,x5]),
        multi_dot([x6,x3,x4,x5]),
        multi_dot([x0,x3,x9]),
        multi_dot([x6,x3,x9]),
        (cos(config.UF_mcs_hand_wheel(t))*multi_dot([self.Mbar_rbs_input_shaft_jcs_input_bearing[:,1:2].T,x3,x4,x10]) + -1*sin(config.UF_mcs_hand_wheel(t))*multi_dot([self.Mbar_rbs_input_shaft_jcs_input_bearing[:,0:1].T,x3,x4,x10])),
        (x7 + x11 + multi_dot([x2,self.ubar_rbs_input_shaft_jcs_input_connect]) + -1*multi_dot([x13,self.ubar_rbs_connect_shaft_jcs_input_connect])),
        multi_dot([self.Mbar_rbs_input_shaft_jcs_input_connect[:,0:1].T,x3,x13,self.Mbar_rbs_connect_shaft_jcs_input_connect[:,0:1]]),
        (x14 + x11 + multi_dot([x16,self.ubar_rbs_output_shaft_jcs_output_connect]) + -1*multi_dot([x13,self.ubar_rbs_connect_shaft_jcs_output_connect])),
        multi_dot([self.Mbar_rbs_output_shaft_jcs_output_connect[:,0:1].T,x17,x13,self.Mbar_rbs_connect_shaft_jcs_output_connect[:,0:1]]),
        (x14 + x8 + multi_dot([x16,self.ubar_rbs_output_shaft_jcs_output_bearing]) + -1*multi_dot([x4,self.ubar_vbs_chassis_jcs_output_bearing])),
        multi_dot([self.Mbar_rbs_output_shaft_jcs_output_bearing[:,0:1].T,x17,x4,x18]),
        multi_dot([self.Mbar_rbs_output_shaft_jcs_output_bearing[:,1:2].T,x17,x4,x18]),
        (x19 + multi_dot([x1.T,x1])),
        (x19 + multi_dot([x12.T,x12])),
        (x19 + multi_dot([x15.T,x15])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((1,1),dtype=np.float64)
        v1 = np.zeros((3,1),dtype=np.float64)

        self.vel_eq_blocks = (v0,
        v0,
        v0,
        v0,
        (v0 + -1*derivative(config.UF_mcs_hand_wheel, t, 0.1, 1)*np.eye(1, dtype=np.float64)),
        v1,
        v0,
        v1,
        v0,
        v1,
        v0,
        v0,
        v0,
        v0,
        v0,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Mbar_vbs_chassis_jcs_input_bearing[:,2:3]
        a1 = a0.T
        a2 = self.P_vbs_chassis
        a3 = A(a2).T
        a4 = self.Pd_rbs_input_shaft
        a5 = self.Mbar_rbs_input_shaft_jcs_input_bearing[:,0:1]
        a6 = B(a4,a5)
        a7 = a5.T
        a8 = self.P_rbs_input_shaft
        a9 = A(a8).T
        a10 = self.Pd_vbs_chassis
        a11 = B(a10,a0)
        a12 = a4.T
        a13 = B(a8,a5).T
        a14 = B(a2,a0)
        a15 = self.Mbar_rbs_input_shaft_jcs_input_bearing[:,1:2]
        a16 = B(a4,a15)
        a17 = a15.T
        a18 = B(a8,a15).T
        a19 = self.ubar_rbs_input_shaft_jcs_input_bearing
        a20 = self.ubar_vbs_chassis_jcs_input_bearing
        a21 = (multi_dot([B(a4,a19),a4]) + -1*multi_dot([B(a10,a20),a10]))
        a22 = (self.Rd_rbs_input_shaft + -1*self.Rd_vbs_chassis + multi_dot([B(a8,a19),a4]) + -1*multi_dot([B(a2,a20),a10]))
        a23 = (self.R_rbs_input_shaft.T + -1*self.R_vbs_chassis.T + multi_dot([a19.T,a9]) + -1*multi_dot([a20.T,a3]))
        a24 = self.Mbar_vbs_chassis_jcs_input_bearing[:,0:1]
        a25 = self.Mbar_rbs_input_shaft_jcs_input_bearing[:,1:2]
        a26 = self.Mbar_rbs_input_shaft_jcs_input_bearing[:,0:1]
        a27 = self.Pd_rbs_connect_shaft
        a28 = self.Mbar_rbs_input_shaft_jcs_input_connect[:,0:1]
        a29 = self.Mbar_rbs_connect_shaft_jcs_input_connect[:,0:1]
        a30 = self.P_rbs_connect_shaft
        a31 = A(a30).T
        a32 = self.Pd_rbs_output_shaft
        a33 = self.Mbar_rbs_connect_shaft_jcs_output_connect[:,0:1]
        a34 = self.Mbar_rbs_output_shaft_jcs_output_connect[:,0:1]
        a35 = self.P_rbs_output_shaft
        a36 = A(a35).T
        a37 = a32.T
        a38 = self.Mbar_rbs_output_shaft_jcs_output_bearing[:,0:1]
        a39 = self.Mbar_vbs_chassis_jcs_output_bearing[:,2:3]
        a40 = B(a10,a39)
        a41 = a39.T
        a42 = B(a2,a39)
        a43 = self.Mbar_rbs_output_shaft_jcs_output_bearing[:,1:2]

        self.acc_eq_blocks = ((multi_dot([a1,a3,a6,a4]) + multi_dot([a7,a9,a11,a10]) + 2*multi_dot([a12,a13,a14,a10])),
        (multi_dot([a1,a3,a16,a4]) + multi_dot([a17,a9,a11,a10]) + 2*multi_dot([a12,a18,a14,a10])),
        (multi_dot([a7,a9,a21]) + 2*multi_dot([a12,a13,a22]) + multi_dot([a23,a6,a4])),
        (multi_dot([a17,a9,a21]) + 2*multi_dot([a12,a18,a22]) + multi_dot([a23,a16,a4])),
        (-1*derivative(config.UF_mcs_hand_wheel, t, 0.1, 2)*np.eye(1, dtype=np.float64) + multi_dot([a24.T,a3,(cos(config.UF_mcs_hand_wheel(t))*B(a4,a25) + -1*sin(config.UF_mcs_hand_wheel(t))*B(a4,a26)),a4]) + multi_dot([(cos(config.UF_mcs_hand_wheel(t))*multi_dot([a25.T,a9]) + -1*sin(config.UF_mcs_hand_wheel(t))*multi_dot([a26.T,a9])),B(a10,a24),a10]) + 2*multi_dot([(cos(config.UF_mcs_hand_wheel(t))*multi_dot([a12,B(a8,a25).T]) + -1*sin(config.UF_mcs_hand_wheel(t))*multi_dot([a12,B(a8,a26).T])),B(a2,a24),a10])),
        (multi_dot([B(a4,self.ubar_rbs_input_shaft_jcs_input_connect),a4]) + -1*multi_dot([B(a27,self.ubar_rbs_connect_shaft_jcs_input_connect),a27])),
        (multi_dot([a28.T,a9,B(a27,a29),a27]) + multi_dot([a29.T,a31,B(a4,a28),a4]) + 2*multi_dot([a12,B(a8,a28).T,B(a30,a29),a27])),
        (multi_dot([B(a32,self.ubar_rbs_output_shaft_jcs_output_connect),a32]) + -1*multi_dot([B(a27,self.ubar_rbs_connect_shaft_jcs_output_connect),a27])),
        (multi_dot([a33.T,a31,B(a32,a34),a32]) + multi_dot([a34.T,a36,B(a27,a33),a27]) + 2*multi_dot([a37,B(a35,a34).T,B(a30,a33),a27])),
        (multi_dot([B(a32,self.ubar_rbs_output_shaft_jcs_output_bearing),a32]) + -1*multi_dot([B(a10,self.ubar_vbs_chassis_jcs_output_bearing),a10])),
        (multi_dot([a38.T,a36,a40,a10]) + multi_dot([a41,a3,B(a32,a38),a32]) + 2*multi_dot([a37,B(a35,a38).T,a42,a10])),
        (multi_dot([a43.T,a36,a40,a10]) + multi_dot([a41,a3,B(a32,a43),a32]) + 2*multi_dot([a37,B(a35,a43).T,a42,a10])),
        2*multi_dot([a12,a4]),
        2*multi_dot([a27.T,a27]),
        2*multi_dot([a37,a32]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,3),dtype=np.float64)
        j1 = self.Mbar_vbs_chassis_jcs_input_bearing[:,2:3]
        j2 = j1.T
        j3 = self.P_vbs_chassis
        j4 = A(j3).T
        j5 = self.P_rbs_input_shaft
        j6 = self.Mbar_rbs_input_shaft_jcs_input_bearing[:,0:1]
        j7 = B(j5,j6)
        j8 = self.Mbar_rbs_input_shaft_jcs_input_bearing[:,1:2]
        j9 = B(j5,j8)
        j10 = j6.T
        j11 = A(j5).T
        j12 = multi_dot([j10,j11])
        j13 = self.ubar_rbs_input_shaft_jcs_input_bearing
        j14 = B(j5,j13)
        j15 = self.ubar_vbs_chassis_jcs_input_bearing
        j16 = (self.R_rbs_input_shaft.T + -1*self.R_vbs_chassis.T + multi_dot([j13.T,j11]) + -1*multi_dot([j15.T,j4]))
        j17 = j8.T
        j18 = multi_dot([j17,j11])
        j19 = B(j3,j1)
        j20 = B(j3,j15)
        j21 = self.Mbar_vbs_chassis_jcs_input_bearing[:,0:1]
        j22 = self.Mbar_rbs_input_shaft_jcs_input_bearing[:,1:2]
        j23 = self.Mbar_rbs_input_shaft_jcs_input_bearing[:,0:1]
        j24 = np.eye(3, dtype=np.float64)
        j25 = self.Mbar_rbs_connect_shaft_jcs_input_connect[:,0:1]
        j26 = self.P_rbs_connect_shaft
        j27 = A(j26).T
        j28 = self.Mbar_rbs_input_shaft_jcs_input_connect[:,0:1]
        j29 = -1*j24
        j30 = self.P_rbs_output_shaft
        j31 = self.Mbar_rbs_connect_shaft_jcs_output_connect[:,0:1]
        j32 = self.Mbar_rbs_output_shaft_jcs_output_connect[:,0:1]
        j33 = A(j30).T
        j34 = self.Mbar_vbs_chassis_jcs_output_bearing[:,2:3]
        j35 = j34.T
        j36 = self.Mbar_rbs_output_shaft_jcs_output_bearing[:,0:1]
        j37 = self.Mbar_rbs_output_shaft_jcs_output_bearing[:,1:2]
        j38 = B(j3,j34)

        self.jac_eq_blocks = (j0,
        multi_dot([j2,j4,j7]),
        j0,
        multi_dot([j10,j11,j19]),
        j0,
        multi_dot([j2,j4,j9]),
        j0,
        multi_dot([j17,j11,j19]),
        j12,
        (multi_dot([j10,j11,j14]) + multi_dot([j16,j7])),
        -1*j12,
        -1*multi_dot([j10,j11,j20]),
        j18,
        (multi_dot([j17,j11,j14]) + multi_dot([j16,j9])),
        -1*j18,
        -1*multi_dot([j17,j11,j20]),
        j0,
        multi_dot([j21.T,j4,(cos(config.UF_mcs_hand_wheel(t))*B(j5,j22) + -1*sin(config.UF_mcs_hand_wheel(t))*B(j5,j23))]),
        j0,
        multi_dot([(cos(config.UF_mcs_hand_wheel(t))*multi_dot([j22.T,j11]) + -1*sin(config.UF_mcs_hand_wheel(t))*multi_dot([j23.T,j11])),B(j3,j21)]),
        j24,
        B(j5,self.ubar_rbs_input_shaft_jcs_input_connect),
        j29,
        -1*B(j26,self.ubar_rbs_connect_shaft_jcs_input_connect),
        j0,
        multi_dot([j25.T,j27,B(j5,j28)]),
        j0,
        multi_dot([j28.T,j11,B(j26,j25)]),
        j29,
        -1*B(j26,self.ubar_rbs_connect_shaft_jcs_output_connect),
        j24,
        B(j30,self.ubar_rbs_output_shaft_jcs_output_connect),
        j0,
        multi_dot([j32.T,j33,B(j26,j31)]),
        j0,
        multi_dot([j31.T,j27,B(j30,j32)]),
        j24,
        B(j30,self.ubar_rbs_output_shaft_jcs_output_bearing),
        j29,
        -1*B(j3,self.ubar_vbs_chassis_jcs_output_bearing),
        j0,
        multi_dot([j35,j4,B(j30,j36)]),
        j0,
        multi_dot([j36.T,j33,j38]),
        j0,
        multi_dot([j35,j4,B(j30,j37)]),
        j0,
        multi_dot([j37.T,j33,j38]),
        j0,
        2*j5.T,
        j0,
        2*j26.T,
        j0,
        2*j30.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = np.eye(3, dtype=np.float64)
        m1 = G(self.P_rbs_input_shaft)
        m2 = G(self.P_rbs_connect_shaft)
        m3 = G(self.P_rbs_output_shaft)

        self.mass_eq_blocks = (config.m_rbs_input_shaft*m0,
        4*multi_dot([m1.T,config.Jbar_rbs_input_shaft,m1]),
        config.m_rbs_connect_shaft*m0,
        4*multi_dot([m2.T,config.Jbar_rbs_connect_shaft,m2]),
        config.m_rbs_output_shaft*m0,
        4*multi_dot([m3.T,config.Jbar_rbs_output_shaft,m3]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_input_shaft)
        f1 = G(self.Pd_rbs_connect_shaft)
        f2 = G(self.Pd_rbs_output_shaft)

        self.frc_eq_blocks = (self.F_rbs_input_shaft_gravity,
        8*multi_dot([f0.T,config.Jbar_rbs_input_shaft,f0,self.P_rbs_input_shaft]),
        self.F_rbs_connect_shaft_gravity,
        8*multi_dot([f1.T,config.Jbar_rbs_connect_shaft,f1,self.P_rbs_connect_shaft]),
        self.F_rbs_output_shaft_gravity,
        8*multi_dot([f2.T,config.Jbar_rbs_output_shaft,f2,self.P_rbs_output_shaft]),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbs_input_shaft_jcs_input_bearing = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_rbs_input_shaft),self.Mbar_rbs_input_shaft_jcs_input_bearing[:,0:1]]),multi_dot([A(self.P_rbs_input_shaft),self.Mbar_rbs_input_shaft_jcs_input_bearing[:,1:2]])],[multi_dot([B(self.P_rbs_input_shaft,self.Mbar_rbs_input_shaft_jcs_input_bearing[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcs_input_bearing[:,2:3]]),multi_dot([B(self.P_rbs_input_shaft,self.Mbar_rbs_input_shaft_jcs_input_bearing[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcs_input_bearing[:,2:3]]),(multi_dot([B(self.P_rbs_input_shaft,self.Mbar_rbs_input_shaft_jcs_input_bearing[:,0:1]).T,(-1*self.R_vbs_chassis + multi_dot([A(self.P_rbs_input_shaft),self.ubar_rbs_input_shaft_jcs_input_bearing]) + -1*multi_dot([A(self.P_vbs_chassis),self.ubar_vbs_chassis_jcs_input_bearing]) + self.R_rbs_input_shaft)]) + multi_dot([B(self.P_rbs_input_shaft,self.ubar_rbs_input_shaft_jcs_input_bearing).T,A(self.P_rbs_input_shaft),self.Mbar_rbs_input_shaft_jcs_input_bearing[:,0:1]])),(multi_dot([B(self.P_rbs_input_shaft,self.Mbar_rbs_input_shaft_jcs_input_bearing[:,1:2]).T,(-1*self.R_vbs_chassis + multi_dot([A(self.P_rbs_input_shaft),self.ubar_rbs_input_shaft_jcs_input_bearing]) + -1*multi_dot([A(self.P_vbs_chassis),self.ubar_vbs_chassis_jcs_input_bearing]) + self.R_rbs_input_shaft)]) + multi_dot([B(self.P_rbs_input_shaft,self.ubar_rbs_input_shaft_jcs_input_bearing).T,A(self.P_rbs_input_shaft),self.Mbar_rbs_input_shaft_jcs_input_bearing[:,1:2]]))]]),self.L_jcs_input_bearing])
        self.F_rbs_input_shaft_jcs_input_bearing = Q_rbs_input_shaft_jcs_input_bearing[0:3,0:1]
        Te_rbs_input_shaft_jcs_input_bearing = Q_rbs_input_shaft_jcs_input_bearing[3:7,0:1]
        self.T_rbs_input_shaft_jcs_input_bearing = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_input_shaft),self.ubar_rbs_input_shaft_jcs_input_bearing])),self.F_rbs_input_shaft_jcs_input_bearing]) + 0.5*multi_dot([E(self.P_rbs_input_shaft),Te_rbs_input_shaft_jcs_input_bearing]))
        Q_rbs_input_shaft_mcs_hand_wheel = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T],[multi_dot([(-1*sin(config.UF_mcs_hand_wheel(t))*B(self.P_rbs_input_shaft,self.Mbar_rbs_input_shaft_jcs_input_bearing[:,0:1]).T + cos(config.UF_mcs_hand_wheel(t))*B(self.P_rbs_input_shaft,self.Mbar_rbs_input_shaft_jcs_input_bearing[:,1:2]).T),A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcs_input_bearing[:,0:1]])]]),self.L_mcs_hand_wheel])
        self.F_rbs_input_shaft_mcs_hand_wheel = Q_rbs_input_shaft_mcs_hand_wheel[0:3,0:1]
        Te_rbs_input_shaft_mcs_hand_wheel = Q_rbs_input_shaft_mcs_hand_wheel[3:7,0:1]
        self.T_rbs_input_shaft_mcs_hand_wheel = 0.5*multi_dot([E(self.P_rbs_input_shaft),Te_rbs_input_shaft_mcs_hand_wheel])
        Q_rbs_input_shaft_jcs_input_connect = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbs_input_shaft,self.ubar_rbs_input_shaft_jcs_input_connect).T,multi_dot([B(self.P_rbs_input_shaft,self.Mbar_rbs_input_shaft_jcs_input_connect[:,0:1]).T,A(self.P_rbs_connect_shaft),self.Mbar_rbs_connect_shaft_jcs_input_connect[:,0:1]])]]),self.L_jcs_input_connect])
        self.F_rbs_input_shaft_jcs_input_connect = Q_rbs_input_shaft_jcs_input_connect[0:3,0:1]
        Te_rbs_input_shaft_jcs_input_connect = Q_rbs_input_shaft_jcs_input_connect[3:7,0:1]
        self.T_rbs_input_shaft_jcs_input_connect = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_input_shaft),self.ubar_rbs_input_shaft_jcs_input_connect])),self.F_rbs_input_shaft_jcs_input_connect]) + 0.5*multi_dot([E(self.P_rbs_input_shaft),Te_rbs_input_shaft_jcs_input_connect]))
        Q_rbs_output_shaft_jcs_output_connect = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbs_output_shaft,self.ubar_rbs_output_shaft_jcs_output_connect).T,multi_dot([B(self.P_rbs_output_shaft,self.Mbar_rbs_output_shaft_jcs_output_connect[:,0:1]).T,A(self.P_rbs_connect_shaft),self.Mbar_rbs_connect_shaft_jcs_output_connect[:,0:1]])]]),self.L_jcs_output_connect])
        self.F_rbs_output_shaft_jcs_output_connect = Q_rbs_output_shaft_jcs_output_connect[0:3,0:1]
        Te_rbs_output_shaft_jcs_output_connect = Q_rbs_output_shaft_jcs_output_connect[3:7,0:1]
        self.T_rbs_output_shaft_jcs_output_connect = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_output_shaft),self.ubar_rbs_output_shaft_jcs_output_connect])),self.F_rbs_output_shaft_jcs_output_connect]) + 0.5*multi_dot([E(self.P_rbs_output_shaft),Te_rbs_output_shaft_jcs_output_connect]))
        Q_rbs_output_shaft_jcs_output_bearing = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbs_output_shaft,self.ubar_rbs_output_shaft_jcs_output_bearing).T,multi_dot([B(self.P_rbs_output_shaft,self.Mbar_rbs_output_shaft_jcs_output_bearing[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcs_output_bearing[:,2:3]]),multi_dot([B(self.P_rbs_output_shaft,self.Mbar_rbs_output_shaft_jcs_output_bearing[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcs_output_bearing[:,2:3]])]]),self.L_jcs_output_bearing])
        self.F_rbs_output_shaft_jcs_output_bearing = Q_rbs_output_shaft_jcs_output_bearing[0:3,0:1]
        Te_rbs_output_shaft_jcs_output_bearing = Q_rbs_output_shaft_jcs_output_bearing[3:7,0:1]
        self.T_rbs_output_shaft_jcs_output_bearing = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_output_shaft),self.ubar_rbs_output_shaft_jcs_output_bearing])),self.F_rbs_output_shaft_jcs_output_bearing]) + 0.5*multi_dot([E(self.P_rbs_output_shaft),Te_rbs_output_shaft_jcs_output_bearing]))

        self.reactions = {'F_rbs_input_shaft_jcs_input_bearing' : self.F_rbs_input_shaft_jcs_input_bearing,
                        'T_rbs_input_shaft_jcs_input_bearing' : self.T_rbs_input_shaft_jcs_input_bearing,
                        'F_rbs_input_shaft_mcs_hand_wheel' : self.F_rbs_input_shaft_mcs_hand_wheel,
                        'T_rbs_input_shaft_mcs_hand_wheel' : self.T_rbs_input_shaft_mcs_hand_wheel,
                        'F_rbs_input_shaft_jcs_input_connect' : self.F_rbs_input_shaft_jcs_input_connect,
                        'T_rbs_input_shaft_jcs_input_connect' : self.T_rbs_input_shaft_jcs_input_connect,
                        'F_rbs_output_shaft_jcs_output_connect' : self.F_rbs_output_shaft_jcs_output_connect,
                        'T_rbs_output_shaft_jcs_output_connect' : self.T_rbs_output_shaft_jcs_output_connect,
                        'F_rbs_output_shaft_jcs_output_bearing' : self.F_rbs_output_shaft_jcs_output_bearing,
                        'T_rbs_output_shaft_jcs_output_bearing' : self.T_rbs_output_shaft_jcs_output_bearing}

