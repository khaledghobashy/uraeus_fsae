
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

        self.indicies_map = {'vbs_ground': 0, 'rbs_rack': 1, 'vbs_chassis': 2}

        self.n  = 7
        self.nc = 7
        self.nrows = 7
        self.ncols = 2*1
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_rbs_rack_jcs_rack', 'T_rbs_rack_jcs_rack', 'F_rbs_rack_mcs_rack_act', 'T_rbs_rack_mcs_rack_act']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6])
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.rbs_rack*2, self.rbs_rack*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_rack*2, self.rbs_rack*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_rack*2, self.rbs_rack*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_rack*2, self.rbs_rack*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_rack*2, self.rbs_rack*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_rack*2, self.rbs_rack*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_rack*2, self.rbs_rack*2+1])

    def set_initial_states(self):
        self.q0  = np.concatenate([self.config.R_rbs_rack,
        self.config.P_rbs_rack])
        self.qd0 = np.concatenate([self.config.Rd_rbs_rack,
        self.config.Pd_rbs_rack])

        self.set_gen_coordinates(self.q0)
        self.set_gen_velocities(self.qd0)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbs_rack = indicies_map[p + 'rbs_rack']
        self.vbs_ground = indicies_map[interface_map[p + 'vbs_ground']]
        self.vbs_chassis = indicies_map[interface_map[p + 'vbs_chassis']]

    
    def eval_constants(self):
        config = self.config

        self.F_rbs_rack_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_rack]], dtype=np.float64)

        self.Mbar_rbs_rack_jcs_rack = multi_dot([A(config.P_rbs_rack).T,triad(config.ax1_jcs_rack)])
        self.Mbar_vbs_chassis_jcs_rack = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcs_rack)])
        self.ubar_rbs_rack_jcs_rack = (multi_dot([A(config.P_rbs_rack).T,config.pt1_jcs_rack]) + -1*multi_dot([A(config.P_rbs_rack).T,config.R_rbs_rack]))
        self.ubar_vbs_chassis_jcs_rack = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcs_rack]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbs_rack_jcs_rack = multi_dot([A(config.P_rbs_rack).T,triad(config.ax1_jcs_rack)])
        self.Mbar_vbs_chassis_jcs_rack = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcs_rack)])
        self.ubar_rbs_rack_jcs_rack = (multi_dot([A(config.P_rbs_rack).T,config.pt1_jcs_rack]) + -1*multi_dot([A(config.P_rbs_rack).T,config.R_rbs_rack]))
        self.ubar_vbs_chassis_jcs_rack = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcs_rack]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))

    
    def set_gen_coordinates(self,q):
        self.R_rbs_rack = q[0:3,0:1]
        self.P_rbs_rack = q[3:7,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbs_rack = qd[0:3,0:1]
        self.Pd_rbs_rack = qd[3:7,0:1]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_rbs_rack = qdd[0:3,0:1]
        self.Pdd_rbs_rack = qdd[3:7,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcs_rack = Lambda[0:5,0:1]
        self.L_mcs_rack_act = Lambda[5:6,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.Mbar_rbs_rack_jcs_rack[:,0:1].T
        x1 = self.P_rbs_rack
        x2 = A(x1)
        x3 = x2.T
        x4 = A(self.P_vbs_chassis)
        x5 = self.Mbar_vbs_chassis_jcs_rack[:,2:3]
        x6 = self.Mbar_rbs_rack_jcs_rack[:,1:2].T
        x7 = (self.R_rbs_rack + -1*self.R_vbs_chassis + multi_dot([x2,self.ubar_rbs_rack_jcs_rack]) + -1*multi_dot([x4,self.ubar_vbs_chassis_jcs_rack]))
        x8 = np.eye(1, dtype=np.float64)

        self.pos_eq_blocks = (multi_dot([x0,x3,x4,x5]),
        multi_dot([x6,x3,x4,x5]),
        multi_dot([x0,x3,x7]),
        multi_dot([x6,x3,x7]),
        multi_dot([x0,x3,x4,self.Mbar_vbs_chassis_jcs_rack[:,1:2]]),
        (-1*config.UF_mcs_rack_act(t)*x8 + multi_dot([self.Mbar_rbs_rack_jcs_rack[:,2:3].T,x3,x7])),
        (-1*x8 + multi_dot([x1.T,x1])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = (v0,
        v0,
        v0,
        v0,
        v0,
        (v0 + -1*derivative(config.UF_mcs_rack_act, t, 0.1, 1)*np.eye(1, dtype=np.float64)),
        v0,)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Mbar_rbs_rack_jcs_rack[:,0:1]
        a1 = a0.T
        a2 = self.P_rbs_rack
        a3 = A(a2).T
        a4 = self.Pd_vbs_chassis
        a5 = self.Mbar_vbs_chassis_jcs_rack[:,2:3]
        a6 = B(a4,a5)
        a7 = a5.T
        a8 = self.P_vbs_chassis
        a9 = A(a8).T
        a10 = self.Pd_rbs_rack
        a11 = B(a10,a0)
        a12 = a10.T
        a13 = B(a2,a0).T
        a14 = B(a8,a5)
        a15 = self.Mbar_rbs_rack_jcs_rack[:,1:2]
        a16 = a15.T
        a17 = B(a10,a15)
        a18 = B(a2,a15).T
        a19 = self.ubar_rbs_rack_jcs_rack
        a20 = self.ubar_vbs_chassis_jcs_rack
        a21 = (multi_dot([B(a10,a19),a10]) + -1*multi_dot([B(a4,a20),a4]))
        a22 = (self.Rd_rbs_rack + -1*self.Rd_vbs_chassis + multi_dot([B(a2,a19),a10]) + -1*multi_dot([B(a8,a20),a4]))
        a23 = (self.R_rbs_rack.T + -1*self.R_vbs_chassis.T + multi_dot([a19.T,a3]) + -1*multi_dot([a20.T,a9]))
        a24 = self.Mbar_vbs_chassis_jcs_rack[:,1:2]
        a25 = self.Mbar_rbs_rack_jcs_rack[:,2:3]

        self.acc_eq_blocks = ((multi_dot([a1,a3,a6,a4]) + multi_dot([a7,a9,a11,a10]) + 2*multi_dot([a12,a13,a14,a4])),
        (multi_dot([a16,a3,a6,a4]) + multi_dot([a7,a9,a17,a10]) + 2*multi_dot([a12,a18,a14,a4])),
        (multi_dot([a1,a3,a21]) + 2*multi_dot([a12,a13,a22]) + multi_dot([a23,a11,a10])),
        (multi_dot([a16,a3,a21]) + 2*multi_dot([a12,a18,a22]) + multi_dot([a23,a17,a10])),
        (multi_dot([a1,a3,B(a4,a24),a4]) + multi_dot([a24.T,a9,a11,a10]) + 2*multi_dot([a12,a13,B(a8,a24),a4])),
        (-1*derivative(config.UF_mcs_rack_act, t, 0.1, 2)*np.eye(1, dtype=np.float64) + multi_dot([a25.T,a3,a21]) + 2*multi_dot([a12,B(a2,a25).T,a22]) + multi_dot([a23,B(a10,a25),a10])),
        2*multi_dot([a12,a10]),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,3),dtype=np.float64)
        j1 = self.Mbar_vbs_chassis_jcs_rack[:,2:3]
        j2 = j1.T
        j3 = self.P_vbs_chassis
        j4 = A(j3).T
        j5 = self.P_rbs_rack
        j6 = self.Mbar_rbs_rack_jcs_rack[:,0:1]
        j7 = B(j5,j6)
        j8 = self.Mbar_rbs_rack_jcs_rack[:,1:2]
        j9 = B(j5,j8)
        j10 = j6.T
        j11 = A(j5).T
        j12 = multi_dot([j10,j11])
        j13 = self.ubar_rbs_rack_jcs_rack
        j14 = B(j5,j13)
        j15 = self.ubar_vbs_chassis_jcs_rack
        j16 = (self.R_rbs_rack.T + -1*self.R_vbs_chassis.T + multi_dot([j13.T,j11]) + -1*multi_dot([j15.T,j4]))
        j17 = j8.T
        j18 = multi_dot([j17,j11])
        j19 = self.Mbar_vbs_chassis_jcs_rack[:,1:2]
        j20 = B(j3,j1)
        j21 = B(j3,j15)
        j22 = self.Mbar_rbs_rack_jcs_rack[:,2:3]
        j23 = j22.T
        j24 = multi_dot([j23,j11])

        self.jac_eq_blocks = (j0,
        multi_dot([j2,j4,j7]),
        j0,
        multi_dot([j10,j11,j20]),
        j0,
        multi_dot([j2,j4,j9]),
        j0,
        multi_dot([j17,j11,j20]),
        j12,
        (multi_dot([j10,j11,j14]) + multi_dot([j16,j7])),
        -1*j12,
        -1*multi_dot([j10,j11,j21]),
        j18,
        (multi_dot([j17,j11,j14]) + multi_dot([j16,j9])),
        -1*j18,
        -1*multi_dot([j17,j11,j21]),
        j0,
        multi_dot([j19.T,j4,j7]),
        j0,
        multi_dot([j10,j11,B(j3,j19)]),
        j24,
        (multi_dot([j23,j11,j14]) + multi_dot([j16,B(j5,j22)])),
        -1*j24,
        -1*multi_dot([j23,j11,j21]),
        j0,
        2*j5.T,)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = G(self.P_rbs_rack)

        self.mass_eq_blocks = (config.m_rbs_rack*np.eye(3, dtype=np.float64),
        4*multi_dot([m0.T,config.Jbar_rbs_rack,m0]),)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_rack)

        self.frc_eq_blocks = (self.F_rbs_rack_gravity,
        8*multi_dot([f0.T,config.Jbar_rbs_rack,f0,self.P_rbs_rack]),)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbs_rack_jcs_rack = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_rbs_rack),self.Mbar_rbs_rack_jcs_rack[:,0:1]]),multi_dot([A(self.P_rbs_rack),self.Mbar_rbs_rack_jcs_rack[:,1:2]]),np.zeros((1,3),dtype=np.float64).T],[multi_dot([B(self.P_rbs_rack,self.Mbar_rbs_rack_jcs_rack[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcs_rack[:,2:3]]),multi_dot([B(self.P_rbs_rack,self.Mbar_rbs_rack_jcs_rack[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcs_rack[:,2:3]]),(multi_dot([B(self.P_rbs_rack,self.Mbar_rbs_rack_jcs_rack[:,0:1]).T,(-1*self.R_vbs_chassis + multi_dot([A(self.P_rbs_rack),self.ubar_rbs_rack_jcs_rack]) + -1*multi_dot([A(self.P_vbs_chassis),self.ubar_vbs_chassis_jcs_rack]) + self.R_rbs_rack)]) + multi_dot([B(self.P_rbs_rack,self.ubar_rbs_rack_jcs_rack).T,A(self.P_rbs_rack),self.Mbar_rbs_rack_jcs_rack[:,0:1]])),(multi_dot([B(self.P_rbs_rack,self.Mbar_rbs_rack_jcs_rack[:,1:2]).T,(-1*self.R_vbs_chassis + multi_dot([A(self.P_rbs_rack),self.ubar_rbs_rack_jcs_rack]) + -1*multi_dot([A(self.P_vbs_chassis),self.ubar_vbs_chassis_jcs_rack]) + self.R_rbs_rack)]) + multi_dot([B(self.P_rbs_rack,self.ubar_rbs_rack_jcs_rack).T,A(self.P_rbs_rack),self.Mbar_rbs_rack_jcs_rack[:,1:2]])),multi_dot([B(self.P_rbs_rack,self.Mbar_rbs_rack_jcs_rack[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcs_rack[:,1:2]])]]),self.L_jcs_rack])
        self.F_rbs_rack_jcs_rack = Q_rbs_rack_jcs_rack[0:3,0:1]
        Te_rbs_rack_jcs_rack = Q_rbs_rack_jcs_rack[3:7,0:1]
        self.T_rbs_rack_jcs_rack = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_rack),self.ubar_rbs_rack_jcs_rack])),self.F_rbs_rack_jcs_rack]) + 0.5*multi_dot([E(self.P_rbs_rack),Te_rbs_rack_jcs_rack]))
        Q_rbs_rack_mcs_rack_act = -1*multi_dot([np.bmat([[multi_dot([A(self.P_rbs_rack),self.Mbar_rbs_rack_jcs_rack[:,2:3]])],[(multi_dot([B(self.P_rbs_rack,self.Mbar_rbs_rack_jcs_rack[:,2:3]).T,(-1*self.R_vbs_chassis + multi_dot([A(self.P_rbs_rack),self.ubar_rbs_rack_jcs_rack]) + -1*multi_dot([A(self.P_vbs_chassis),self.ubar_vbs_chassis_jcs_rack]) + self.R_rbs_rack)]) + multi_dot([B(self.P_rbs_rack,self.ubar_rbs_rack_jcs_rack).T,A(self.P_rbs_rack),self.Mbar_rbs_rack_jcs_rack[:,2:3]]))]]),self.L_mcs_rack_act])
        self.F_rbs_rack_mcs_rack_act = Q_rbs_rack_mcs_rack_act[0:3,0:1]
        Te_rbs_rack_mcs_rack_act = Q_rbs_rack_mcs_rack_act[3:7,0:1]
        self.T_rbs_rack_mcs_rack_act = 0.5*multi_dot([E(self.P_rbs_rack),Te_rbs_rack_mcs_rack_act])

        self.reactions = {'F_rbs_rack_jcs_rack' : self.F_rbs_rack_jcs_rack,
                        'T_rbs_rack_jcs_rack' : self.T_rbs_rack_jcs_rack,
                        'F_rbs_rack_mcs_rack_act' : self.F_rbs_rack_mcs_rack_act,
                        'T_rbs_rack_mcs_rack_act' : self.T_rbs_rack_mcs_rack_act}

