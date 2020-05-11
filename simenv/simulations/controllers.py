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
        print('vel = %s'%v_c)
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

class stanley_controller(object):

    def __init__(self, way_points):
       self._waypoints = way_points
       self._gain = 0.3
       self._k_soft = 1

    
    def get_steer_angle(self, r_ax1, P_ch, vel):

        vel = abs(vel)
        k = self._gain
        k_soft = self._k_soft

        x_ax1, y_ax1, _ = r_ax1.flat[:]

        yaw_ch = self.get_yaw_angle(P_ch)
        err, yaw_path = self.get_waypoint(x_ax1, y_ax1, yaw_ch)

        yaw_diff = (yaw_ch - yaw_path)
        print(yaw_diff)
        if yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        if yaw_diff < - np.pi:
            yaw_diff += 2 * np.pi
        
        heading_factor = yaw_diff
        crosstrack_factor = np.arctan2(k * err, k_soft + vel)
        delta = yaw_diff + crosstrack_factor

        delta = clamp(delta, np.deg2rad(-60), np.deg2rad(60))

        print('x_ax1, y_ax1 = %s'%((x_ax1, y_ax1),))
        print('vel = %s'%vel)
        print('heading_factor = %s'%heading_factor)
        print('crosstrack_factor = %s'%crosstrack_factor)
        print('E = %s'%err)
        print('Yaw_Ch, Yaw_Path = %s'%((yaw_ch, yaw_path),))
        print('delta = %s\n'%delta)

        return delta
    
    def get_yaw_angle(self, P_ch):
        w, x, y, z = P_ch.flat[:]
        angle = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        return angle
    
    def get_waypoint2(self, x, y):
        min_idx       = 0
        min_dist      = float("inf")
        for i in range(self._waypoints.shape[0]):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - x,
                    self._waypoints[i][1] - y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        print('min_dist = %s'%min_dist)
        print('min_idx = %s'%min_idx)
        
        err = min_dist
        yaw_path = np.arctan(self._waypoints[min_idx][2])
        if min_idx != 0:
            yaw_path = np.arctan2(self._waypoints[min_idx][1]-self._waypoints[min_idx-1][1], self._waypoints[min_idx][0]-self._waypoints[min_idx-1][0])


        return err, yaw_path
    
    def get_waypoint(self, x, y, yaw):

        cx = self._waypoints[:,0]
        cy = self._waypoints[:,1]

        # Search nearest point index
        dx = [x - icx for icx in cx]
        dy = [y - icy for icy in cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        front_axle_vec = [-np.cos(yaw + np.pi / 2),
                          -np.sin(yaw + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        print('target_idx = %s'%target_idx)

        yaw_path = self._waypoints[target_idx][2]

        return error_front_axle, yaw_path

