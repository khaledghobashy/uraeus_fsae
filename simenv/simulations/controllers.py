import numpy as np
from uraeus.nmbd.python.engine.numerics.math_funcs import A

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

class longitudinal_control(object):

    def __init__(self):

        self._desired_speed = (30 / 3.6) * 1e3
        self._errors_array = []

        self.Kp = 5*1e-3
        self.Ki = 1*1e-5
        self.Kd = 0

        self._sum_int = 0

    def _get_error(self, P_ch, Rd_ch):
        v_c = abs(A(P_ch).T @ Rd_ch)[0,0]
        v_r = self._desired_speed
        err = v_r - v_c
        self._errors_array.append(err)
        
    def get_torque_factor(self, P_ch, Rd_ch):
        self._get_error(P_ch, Rd_ch)
        err = self._errors_array[-1]

        self._sum_int += err*1e-3

        P = self.Kp * err
        I = self.Ki * self._sum_int #sum(np.array(self._errors_array)*1e-3)
        #D = self.Kd * np.diff(self._errors_array)[-1]

        factor = P + I #* D

        if factor > 1.3:
            self._sum_int -= err*1e-3

        factor = clamp(factor, -1.2, 1.2)
        print('E = %s'%err)
        print('P = %s'%P)
        print('I = %s'%I)
        print('F = %s\n'%factor)

        return factor

