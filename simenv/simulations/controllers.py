import numpy as np
from uraeus.nmbd.python.engine.numerics.math_funcs import A

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

class speed_controller(object):

    def __init__(self, desired_speed, sample_time, gains=[]):

        self.desired_speed = (desired_speed  / 3.6) * 1e3
        self.dt = sample_time

        if not gains:
            self.Kp = 5*1e-3
            self.Ki = 3*1e-4
            self.Kd = 0
        else:
            self.Kp, self.Ki, self.Kd = gains
        
        self._errors_array = []
        self._sum_int = 0
        self._last_err = 0

    def _get_error(self, P_ch, Rd_ch):
        v_c = abs(A(P_ch).T @ Rd_ch)[0,0]
        v_r = self.desired_speed
        err = v_r - v_c
        self._errors_array.append(err)
        return err
        
    def get_torque_factor(self, P_ch, Rd_ch):
        err = self._get_error(P_ch, Rd_ch)

        self._sum_int += err * self.dt

        P = self.Kp * err
        I = self.Ki * self._sum_int
        D = self.Kd * (err - self._last_err)
        
        self._last_err = err
        
        factor = P + I + D
        if factor > 1.2 or factor < -1.2:
            self._sum_int -= err * self.dt
        
        factor = clamp(factor, -1.2, 1.2)
        
        print('E = %s'%err)
        print('P = %s'%P)
        print('I = %s'%I)
        print('D = %s'%D)
        print('F = %s\n'%factor)

        return factor

