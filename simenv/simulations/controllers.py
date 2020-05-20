import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from uraeus.nmbd.python.engine.numerics.math_funcs import A, E


def clamp(n, nmin, nmax):
    return max(min(nmax, n), nmin)

def normalize(v):
    n = v / np.linalg.norm(v)
    return n


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
        if factor > 1 or factor < -1:
            self._sum_int -= err * self.dt
        
        factor = clamp(factor, -1, 1)
        
        #self._print_states(err, P, I, D, factor)

        return factor
    
    def _print_states(self, err, P, I, D, factor):
        print('E = %s'%err)
        print('P = %s'%P)
        print('I = %s'%I)
        print('D = %s'%D)
        print('F = %s\n'%factor)


class trajectory(object):

    def __init__(self, waypoints):

        # waypoints' array containig the x and y coordinates of the desired 
        # path. shape = (n, 2)
        self._waypoints = waypoints

        # Evaluating path heading vectors as vector point from current point to
        # the next point as: array([[x[i+1] - x[i]], [y[i+1] - y[i]]])
        x_diffs = np.diff(waypoints[:,0])[:,None]
        y_diffs = np.diff(waypoints[:,1])[:,None]

        self._path_headings = [normalize(np.array([x,  y])) for x,y in zip(x_diffs, y_diffs)] 
        self._path_normals  = [normalize(np.array([y, -x])) for x,y in zip(x_diffs, y_diffs)]

        self._radii = waypoints[:,2] #abs(((1 + interpolate.splev(x_new, tck, der=2)**2)**(3/2))/(interpolate.splev(x_new, tck, der=2)))

        # variable holding the current waypoint index
        self._idx = 0

    def get_heading_error(self, P_ch):

        idx = self._idx

        try:
            path_heading = self._path_headings[idx]
        except(IndexError):
            path_heading = self._path_headings[idx-1]
        
        # getting the chassis heading as a 2d vector of two components [x, y]
        chassis_heading = (A(P_ch) @ np.array([[-1], [0], [0]]))[0:2]
        # normalizing the vector to get non-scaled angle
        chassis_heading = normalize(chassis_heading)

        # The cross funtion returns a scaler for 2d vectors that represent the
        # z-axis value of the resultant normal vector. This captures the 
        # direction of rotation needed too.
        angle = np.cross(path_heading.flat[:], chassis_heading.flat[:])

        return angle
    
    
    def get_crosstrack_error(self, x, y):
        
        path_x = self._waypoints[:,0]
        path_y = self._waypoints[:,1]

        # Search nearest point index as the smallest hypotenuse fromed by the
        # x and y differences between the reference point and the path
        dx = x - path_x # broadcasting rule (float - array)
        dy = y - path_y # broadcasting rule (float - array)
        d  = np.hypot(dx, dy)

        # target waypoint index
        self._idx = idx = np.argmin(d)

        # error vector point from path point to the vehicle reference point
        error_vector = np.array([dx[idx], dy[idx]])

        # projecting the error on the path normal, as the error is the normal 
        # distance between the path and the reference point, relative to path
        try:
            path_normal = self._path_normals[idx]
        except(IndexError):
            # assuming forward path vector in case of indexError. 
            path_normal = self._path_normals[idx-1]
                
        # error value
        error = float(-np.dot(error_vector.T, path_normal))

        return error
    
    def get_radius(self):
        return self._radii[self._idx]


class stanley_controller(object):

    def __init__(self, waypoints, gain=1):

        self.trajectory = trajectory(waypoints)
        
        # crossfactor error gain
        self._gain = gain
        # softning gain for lower vehicle speeds (1 m/s)
        self._k_soft = 1e3
        self._kd_yawrate = 0

        self.error_array = [0]
        self.steering_angles = [0, 0]

    def get_steer_factor(self, r_ax1, P_ch, Pd_ch, vel):

        k = self._gain # crossfactor gain
        k_soft = self._k_soft # softning gain
        kd_yawrate = self._kd_yawrate
       
        # longitudinal velocity of the front axle
        vel = abs(vel)
        # x and y coordinates of the reference point at the front axle
        x_ax1, y_ax1, _ = r_ax1.flat[:]
        
        crosstrack_error = self.trajectory.get_crosstrack_error(x_ax1, y_ax1)
        heading_error = self.trajectory.get_heading_error(P_ch) 
        
        crosstrack_factor = np.arctan2(k * crosstrack_error, (k_soft + vel))
        
        steadystate_yawrate = vel / self.trajectory.get_radius()
        yaw_damping_factor = kd_yawrate * (self._get_yaw_rate(P_ch, Pd_ch) - steadystate_yawrate)

        # wheels steering angle needed
        delta = heading_error + crosstrack_factor + yaw_damping_factor
        # clamping the value to the applicable angular boundries
        delta = clamp(delta, np.deg2rad(-60), np.deg2rad(60))

        self.steering_angles.append(delta)
        
        print('vel = %s'%vel)
        print('x_ax1, y_ax1 = %s'%((x_ax1, y_ax1),))
        print('target_idx = %s'%self.trajectory._idx)
        print('heading_error = %s'%heading_error)
        print('crosstrack_error = %s'%crosstrack_error)
        print('crosstrack_factor = %s'%crosstrack_factor)
        print('yaw_damp_factor = %s'%yaw_damping_factor)
        print('delta = %s\n'%delta)

        self.error_array.append(crosstrack_error)

        return delta
    
    def _get_yaw_angle(self, P_ch):
        w, x, y, z = P_ch.flat[:]
        angle = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        return angle
    
    def _get_yaw_rate(self, P_ch, Pd_ch):
        yaw_rate = 2*E(P_ch)@Pd_ch # Global
        return yaw_rate[2,0]

    


