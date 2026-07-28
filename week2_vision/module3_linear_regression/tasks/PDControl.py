from math import hypot


def shortest_yaw_error(target, current):
    """Return signed error (degrees) using the shortest rotation direction."""
    return ((target - current) + 180) % 360 - 180

class PDController:
    def __init__(self, kp, kd, max_output, wrap=False):
        self.kp = kp
        self.kd = kd
        self.max_output = max_output
        self.setpoint = 0.0
        self.prev_position = 0.0
        self.wrap = wrap

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def calculate_p(self, current_position):
        self.prev_position = current_position
        return shortest_yaw_error(self.setpoint, current_position) * self.kp

    def calculate_d(self, current_velocity):
        return -current_velocity * self.kd

    def calculate_d_position(self, current_position, delta_t):
        return (self.prev_position - current_position) / delta_t * self.kd

    def _calculate(self, current_position, current_velocity):
        return self.calculate_p(current_position) + self.calculate_d(current_velocity)

    def _calculate_position(self, current_position, delta_t):
        return self.calculate_p(current_position) + self.calculate_d_position(current_position, delta_t)

    def calculate(self, current_position, current_velocity):
        return max(-self.max_output, min(self.max_output, self._calculate(current_position, current_velocity)))

    def calculate_position(self, current_position, delta_t):
        return max(-self.max_output, min(self.max_output, self._calculate_position(current_position, delta_t)))

    def get_error(self):
        return self.prev_position - self.setpoint

MAX_SPEED    = 1   # PCMD pitch/roll magnitude
MAX_THROTTLE = 1
MAX_YAW = 0.5
KP_TRANS = 0.75
KD_TRANS = 0.5
KP_ALT = 3
KD_ALT = 1
KP_YAW = 0.01

class FullController:
    def __init__(self, kp_alt=KP_ALT, kd_alt=KD_ALT, kp_trans=KP_TRANS, kd_trans=KD_TRANS, kp_yaw=KP_YAW, max_throttle=MAX_THROTTLE, max_speed=MAX_SPEED, max_yaw=MAX_YAW):
        self.fwd = PDController(kp_trans, kd_trans, max_speed, wrap=False)
        self.rgt = PDController(kp_trans, kd_trans, max_speed, wrap=False)
        self.yaw = PDController(kp_yaw, 0.0, max_yaw, wrap=True)
        self.alt = PDController(kp_alt, kd_alt, max_throttle, wrap=False)

    def calculate(self, _fwd=0, _rgt=0, _yaw=0, _alt=0, _fwd_vel=0, _rgt_vel=0, _alt_vel=0, dt=0.05):
        return [self.fwd.calculate(_fwd, _fwd_vel), self.rgt.calculate(_rgt, _rgt_vel), self.yaw.calculate_position(_yaw, dt), self.alt.calculate(_alt, _alt_vel)]

    def get_trans_error(self):
        return hypot(self.fwd.get_error(), self.rgt.get_error(), self.alt.get_error())

    def set_setpoint(self, _fwd=0, _rgt=0, _yaw=0, _alt=0):
        self.fwd.set_setpoint(_fwd)
        self.rgt.set_setpoint(_rgt)
        self.alt.set_setpoint(_alt)
        self.yaw.set_setpoint(_yaw)