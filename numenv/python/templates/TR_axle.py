
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

        self.indicies_map = {'vbs_ground': 0, 'vbr_wheel_hub': 1, 'vbl_wheel_hub': 2, 'vbr_wheel_upright': 3, 'vbl_wheel_upright': 4}

        self.n  = 0
        self.nc = 2
        self.nrows = 2
        self.ncols = 2*0
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_vbr_wheel_hub_mcr_wheel_lock', 'T_vbr_wheel_hub_mcr_wheel_lock', 'F_vbl_wheel_hub_mcl_wheel_lock', 'T_vbl_wheel_hub_mcl_wheel_lock']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.vbr_wheel_hub*2, self.vbr_wheel_hub*2+1, self.vbr_wheel_upright*2, self.vbr_wheel_upright*2+1, self.vbl_wheel_hub*2, self.vbl_wheel_hub*2+1, self.vbl_wheel_upright*2, self.vbl_wheel_upright*2+1])

    def set_initial_states(self):
        self.q0  = np.concatenate([[]])
        self.qd0 = np.concatenate([[]])

        self.set_gen_coordinates(self.q0)
        self.set_gen_velocities(self.qd0)

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
    
        self.vbr_wheel_upright = indicies_map[interface_map[p + 'vbr_wheel_upright']]
        self.vbl_wheel_hub = indicies_map[interface_map[p + 'vbl_wheel_hub']]
        self.vbr_wheel_hub = indicies_map[interface_map[p + 'vbr_wheel_hub']]
        self.vbl_wheel_upright = indicies_map[interface_map[p + 'vbl_wheel_upright']]
        self.vbs_ground = indicies_map[interface_map[p + 'vbs_ground']]

    
    def eval_constants(self):
        config = self.config

    

        self.Mbar_vbr_wheel_hub_jcr_hub_bearing = multi_dot([A(config.P_vbr_wheel_hub).T,triad(config.ax1_jcr_hub_bearing)])
        self.Mbar_vbr_wheel_upright_jcr_hub_bearing = multi_dot([A(config.P_vbr_wheel_upright).T,triad(config.ax1_jcr_hub_bearing)])
        self.Mbar_vbl_wheel_hub_jcl_hub_bearing = multi_dot([A(config.P_vbl_wheel_hub).T,triad(config.ax1_jcl_hub_bearing)])
        self.Mbar_vbl_wheel_upright_jcl_hub_bearing = multi_dot([A(config.P_vbl_wheel_upright).T,triad(config.ax1_jcl_hub_bearing)])

    
    def set_gen_coordinates(self,q):
        pass

    
    def set_gen_velocities(self,qd):
        pass

    
    def set_gen_accelerations(self,qdd):
        pass

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_mcr_wheel_lock = Lambda[0:1,0:1]
        self.L_mcl_wheel_lock = Lambda[1:2,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = A(self.P_vbr_wheel_hub).T
        x1 = A(self.P_vbr_wheel_upright)
        x2 = self.Mbar_vbr_wheel_upright_jcr_hub_bearing[:,0:1]
        x3 = A(self.P_vbl_wheel_hub).T
        x4 = A(self.P_vbl_wheel_upright)
        x5 = self.Mbar_vbl_wheel_upright_jcl_hub_bearing[:,0:1]

        self.pos_eq_blocks = ((cos(config.UF_mcr_wheel_lock(t)) * multi_dot([self.Mbar_vbr_wheel_hub_jcr_hub_bearing[:,1:2].T,x0,x1,x2]) + (-1 * sin(config.UF_mcr_wheel_lock(t))) * multi_dot([self.Mbar_vbr_wheel_hub_jcr_hub_bearing[:,0:1].T,x0,x1,x2])),
        (cos(config.UF_mcl_wheel_lock(t)) * multi_dot([self.Mbar_vbl_wheel_hub_jcl_hub_bearing[:,1:2].T,x3,x4,x5]) + (-1 * sin(config.UF_mcl_wheel_lock(t))) * multi_dot([self.Mbar_vbl_wheel_hub_jcl_hub_bearing[:,0:1].T,x3,x4,x5])),)

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((1,1),dtype=np.float64)
        v1 = np.eye(1, dtype=np.float64)

        self.vel_eq_blocks = ((v0 + (-1 * derivative(config.UF_mcr_wheel_lock, t, 0.1, 1)) * v1),
        (v0 + (-1 * derivative(config.UF_mcl_wheel_lock, t, 0.1, 1)) * v1),)

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = np.eye(1, dtype=np.float64)
        a1 = self.Mbar_vbr_wheel_upright_jcr_hub_bearing[:,0:1]
        a2 = self.P_vbr_wheel_upright
        a3 = self.Pd_vbr_wheel_hub
        a4 = self.Mbar_vbr_wheel_hub_jcr_hub_bearing[:,1:2]
        a5 = self.Mbar_vbr_wheel_hub_jcr_hub_bearing[:,0:1]
        a6 = self.P_vbr_wheel_hub
        a7 = A(a6).T
        a8 = self.Pd_vbr_wheel_upright
        a9 = a3.T
        a10 = self.Mbar_vbl_wheel_upright_jcl_hub_bearing[:,0:1]
        a11 = self.P_vbl_wheel_upright
        a12 = self.Pd_vbl_wheel_hub
        a13 = self.Mbar_vbl_wheel_hub_jcl_hub_bearing[:,1:2]
        a14 = self.Mbar_vbl_wheel_hub_jcl_hub_bearing[:,0:1]
        a15 = self.P_vbl_wheel_hub
        a16 = A(a15).T
        a17 = self.Pd_vbl_wheel_upright
        a18 = a12.T

        self.acc_eq_blocks = (((-1 * derivative(config.UF_mcr_wheel_lock, t, 0.1, 2)) * a0 + multi_dot([a1.T,A(a2).T,(cos(config.UF_mcr_wheel_lock(t)) * B(a3,a4) + (-1 * sin(config.UF_mcr_wheel_lock(t))) * B(a3,a5)),a3]) + multi_dot([(cos(config.UF_mcr_wheel_lock(t)) * multi_dot([a4.T,a7]) + (-1 * sin(config.UF_mcr_wheel_lock(t))) * multi_dot([a5.T,a7])),B(a8,a1),a8]) + (2) * multi_dot([(cos(config.UF_mcr_wheel_lock(t)) * multi_dot([a9,B(a6,a4).T]) + (-1 * sin(config.UF_mcr_wheel_lock(t))) * multi_dot([a9,B(a6,a5).T])),B(a2,a1),a8])),
        ((-1 * derivative(config.UF_mcl_wheel_lock, t, 0.1, 2)) * a0 + multi_dot([a10.T,A(a11).T,(cos(config.UF_mcl_wheel_lock(t)) * B(a12,a13) + (-1 * sin(config.UF_mcl_wheel_lock(t))) * B(a12,a14)),a12]) + multi_dot([(cos(config.UF_mcl_wheel_lock(t)) * multi_dot([a13.T,a16]) + (-1 * sin(config.UF_mcl_wheel_lock(t))) * multi_dot([a14.T,a16])),B(a17,a10),a17]) + (2) * multi_dot([(cos(config.UF_mcl_wheel_lock(t)) * multi_dot([a18,B(a15,a13).T]) + (-1 * sin(config.UF_mcl_wheel_lock(t))) * multi_dot([a18,B(a15,a14).T])),B(a11,a10),a17])),)

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,3),dtype=np.float64)
        j1 = self.Mbar_vbr_wheel_upright_jcr_hub_bearing[:,0:1]
        j2 = self.P_vbr_wheel_upright
        j3 = self.P_vbr_wheel_hub
        j4 = self.Mbar_vbr_wheel_hub_jcr_hub_bearing[:,1:2]
        j5 = self.Mbar_vbr_wheel_hub_jcr_hub_bearing[:,0:1]
        j6 = A(j3).T
        j7 = self.Mbar_vbl_wheel_upright_jcl_hub_bearing[:,0:1]
        j8 = self.P_vbl_wheel_upright
        j9 = self.P_vbl_wheel_hub
        j10 = self.Mbar_vbl_wheel_hub_jcl_hub_bearing[:,1:2]
        j11 = self.Mbar_vbl_wheel_hub_jcl_hub_bearing[:,0:1]
        j12 = A(j9).T

        self.jac_eq_blocks = (j0,
        multi_dot([j1.T,A(j2).T,(cos(config.UF_mcr_wheel_lock(t)) * B(j3,j4) + (-1 * sin(config.UF_mcr_wheel_lock(t))) * B(j3,j5))]),
        j0,
        multi_dot([(cos(config.UF_mcr_wheel_lock(t)) * multi_dot([j4.T,j6]) + (-1 * sin(config.UF_mcr_wheel_lock(t))) * multi_dot([j5.T,j6])),B(j2,j1)]),
        j0,
        multi_dot([j7.T,A(j8).T,(cos(config.UF_mcl_wheel_lock(t)) * B(j9,j10) + (-1 * sin(config.UF_mcl_wheel_lock(t))) * B(j9,j11))]),
        j0,
        multi_dot([(cos(config.UF_mcl_wheel_lock(t)) * multi_dot([j10.T,j12]) + (-1 * sin(config.UF_mcl_wheel_lock(t))) * multi_dot([j11.T,j12])),B(j8,j7)]),)

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

    

        self.mass_eq_blocks = (,)

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

    

        self.frc_eq_blocks = (,)

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_vbr_wheel_hub_mcr_wheel_lock = (-1) * multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T],[multi_dot([((-1 * sin(config.UF_mcr_wheel_lock(t))) * B(self.P_vbr_wheel_hub,self.Mbar_vbr_wheel_hub_jcr_hub_bearing[:,0:1]).T + cos(config.UF_mcr_wheel_lock(t)) * B(self.P_vbr_wheel_hub,self.Mbar_vbr_wheel_hub_jcr_hub_bearing[:,1:2]).T),A(self.P_vbr_wheel_upright),self.Mbar_vbr_wheel_upright_jcr_hub_bearing[:,0:1]])]]),self.L_mcr_wheel_lock])
        self.F_vbr_wheel_hub_mcr_wheel_lock = Q_vbr_wheel_hub_mcr_wheel_lock[0:3,0:1]
        Te_vbr_wheel_hub_mcr_wheel_lock = Q_vbr_wheel_hub_mcr_wheel_lock[3:7,0:1]
        self.T_vbr_wheel_hub_mcr_wheel_lock = (0.5) * multi_dot([E(self.P_vbr_wheel_hub),Te_vbr_wheel_hub_mcr_wheel_lock])
        Q_vbl_wheel_hub_mcl_wheel_lock = (-1) * multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T],[multi_dot([((-1 * sin(config.UF_mcl_wheel_lock(t))) * B(self.P_vbl_wheel_hub,self.Mbar_vbl_wheel_hub_jcl_hub_bearing[:,0:1]).T + cos(config.UF_mcl_wheel_lock(t)) * B(self.P_vbl_wheel_hub,self.Mbar_vbl_wheel_hub_jcl_hub_bearing[:,1:2]).T),A(self.P_vbl_wheel_upright),self.Mbar_vbl_wheel_upright_jcl_hub_bearing[:,0:1]])]]),self.L_mcl_wheel_lock])
        self.F_vbl_wheel_hub_mcl_wheel_lock = Q_vbl_wheel_hub_mcl_wheel_lock[0:3,0:1]
        Te_vbl_wheel_hub_mcl_wheel_lock = Q_vbl_wheel_hub_mcl_wheel_lock[3:7,0:1]
        self.T_vbl_wheel_hub_mcl_wheel_lock = (0.5) * multi_dot([E(self.P_vbl_wheel_hub),Te_vbl_wheel_hub_mcl_wheel_lock])

        self.reactions = {'F_vbr_wheel_hub_mcr_wheel_lock' : self.F_vbr_wheel_hub_mcr_wheel_lock,
                        'T_vbr_wheel_hub_mcr_wheel_lock' : self.T_vbr_wheel_hub_mcr_wheel_lock,
                        'F_vbl_wheel_hub_mcl_wheel_lock' : self.F_vbl_wheel_hub_mcl_wheel_lock,
                        'T_vbl_wheel_hub_mcl_wheel_lock' : self.T_vbl_wheel_hub_mcl_wheel_lock}

