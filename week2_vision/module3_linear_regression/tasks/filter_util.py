import threading
import time
from typing import Union

import numpy as np


class VectorOneEuroFilter:
    def __init__(self, t0, x0: np.ndarray, dx0=None, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_prev = np.array(x0, dtype=float)
        self.dx_prev = np.zeros_like(self.x_prev) if dx0 is None else np.array(dx0, dtype=float)
        self.t_prev = float(t0)

    def _smoothing_factor(self, te, cutoff):
        r = 2 * np.pi * cutoff * te
        return r / (r + 1)

    def __call__(self, t, x: np.ndarray):
        te = t - self.t_prev
        if te <= 0:
            return self.x_prev

        x = np.array(x, dtype=float)

        edx = (x - self.x_prev) / te
        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx = a_d * edx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx)

        a_x = self._smoothing_factor(te, cutoff)
        x_filtered = a_x * x + (1.0 - a_x) * self.x_prev

        self.t_prev = t
        self.x_prev = x_filtered
        self.dx_prev = dx

        return x_filtered


class ExponentialLowPassFilter:
    """
    An exponential low-pass filter for smoothing noisy signals.
    """

    def __init__(self, alpha: float):
        """
        Initialize the filter.

        Args:
            alpha (float): The smoothing factor, between 0.0 and 1.0.
                           - 1.0 means no filtering (raw signal passes through).
                           - Values closer to 0.0 mean heavy filtering/smoothing.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")

        self.alpha = alpha
        self.last_estimate = None

    def __call__(self, current_value: float) -> float:
        """
        Calculate and return the new filtered value.

        Args:
            current_value (float): The latest raw measurement.

        Returns:
            float: The filtered estimate.
        """
        # If this is the first measurement, trust it completely to avoid an initial lag spike
        if self.last_estimate is None:
            self.last_estimate = current_value
        else:
            # Apply the exponential moving average formula
            self.last_estimate = (self.alpha * current_value) + ((1.0 - self.alpha) * self.last_estimate)

        return self.last_estimate

    def reset(self):
        """
        Reset the filter's state. Useful if the signal is lost and re-acquired.
        """
        self.last_estimate = None


class VectorExponentialLowPassFilter:
    """
    An exponential low-pass filter compatible with scalars and NumPy arrays.
    """

    def __init__(self, alpha: float):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")

        self.alpha = alpha
        self.last_estimate = None

    def __call__(self, current_value: np.ndarray) -> np.ndarray:
        # Convert inputs to numpy arrays if they are lists for safety
        if isinstance(current_value, list):
            current_value = np.array(current_value)

        # First pass initialization
        if self.last_estimate is None:
            # Use copy() to prevent mutating the original input array later
            self.last_estimate = current_value.copy() if isinstance(current_value, np.ndarray) else current_value
        else:
            # NumPy broadcasting handles the element-wise math automatically
            self.last_estimate = (self.alpha * current_value) + ((1.0 - self.alpha) * self.last_estimate)

        return self.last_estimate

    def reset(self):
        self.last_estimate = None


class BooleanDebouncer:
    def __init__(self, delay_seconds: float = 0.05):
        self.delay = delay_seconds
        self._stable_value = False
        self._last_raw_value = False
        self._change_time = time.time()

    def update(self, raw_value: bool) -> bool:
        now = time.time()
        if raw_value != self._last_raw_value:
            self._last_raw_value = raw_value
            self._change_time = now
        elif (now - self._change_time) >= self.delay:
            self._stable_value = raw_value
        return self._stable_value

    @property
    def value(self) -> bool:
        return self._stable_value


class GatePositionKalmanFilter:
    def __init__(self, dt, initial_position, r_base=0.1, q_base=0.01):
        """
        Initializes the Linear Kalman Filter for gate height tracking.
        dt: Time step between predictions (seconds)
        initial_position: First rough guess of the gate position
        r_base: Baseline measurement noise multiplier
        """
        # State vector: x = [height, velocity]^T
        self.x = np.array([[initial_position],
                           [0.0]])

        # State transition matrix (F) - Constant Velocity Model
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])

        # Measurement matrix (H) - We only measure height
        self.H = np.array([[1.0, 0.0]])

        # State Covariance matrix (P) - High initial uncertainty
        self.P = np.array([[500.0, 0.0],
                           [0.0, 500.0]])

        # Process Noise Covariance (Q) - Adjust based on expected drone drift
        self.Q = np.array([[q_base, 0.0],
                           [0.0, q_base]])

        # Base multiplier for measurement noise
        self.r_base = r_base

    def predict(self, dt):
        """Predicts the next state. Call this every flight controller loop."""
        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])
        # x = F * x
        self.x = np.dot(self.F, self.x)

        # P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.x[0, 0]  # Return predicted height

    def update(self, measured_position, distance_to_gate, current_r_base):
        # Apply the specific r_base for the current vision mode
        R = np.array([[current_r_base * (distance_to_gate ** 2)]])
        # Dynamic R: Noise increases quadratically with distance
        R = np.array([[self.r_base * (distance_to_gate ** 2)]])

        # Measurement residual: y = z - H * x
        z = np.array([[measured_position]])
        y = z - np.dot(self.H, self.x)

        # Innovation covariance: S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + R

        # Kalman Gain: K = P * H^T * S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update state: x = x + K * y
        self.x = self.x + np.dot(K, y)

        # Update covariance: P = (I - K * H) * P
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return self.x[0, 0]  # Return updated height

    def reset(self, initial_position=0.0):
        """Resets the filter to a new initial position."""
        self.x = np.array([[initial_position],
                           [0.0]])
        self.P = np.array([[500.0, 0.0],
                           [0.0, 500.0]])


import numpy as np


class DistanceSensorFusionFilter:
    def __init__(self, dt, initial_distance, initial_velocity=0.0):
        self.lock = threading.Lock()

        # State vector: [Distance, Velocity]
        self.x = np.array([[initial_distance],
                           [initial_velocity]])

        self.F = np.array([[1.0, dt],
                           [0.0, 1.0]])

        # H is now an Identity matrix because we measure both state variables
        self.H = np.array([[1.0, 0.0],
                           [0.0, 1.0]])

        self.P = np.array([[100.0, 0.0],
                           [0.0, 100.0]])

        # Q can be lower now because we aren't relying purely on the model for velocity
        self.Q = np.array([[0.01, 0.0],
                           [0.0, 0.05]])

        # How much we trust the IMU (tune this based on your ARK sensor's specs)
        self.r_imu = 0.05

    def predict(self, dt):
        with self.lock:
            self.F = np.array([[1.0, dt],
                               [0.0, 1.0]])
            self.x = np.dot(self.F, self.x)
            self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
            current_distance = self.x[0, 0]
        return current_distance  # Return predicted distance

    def update(self, camera_distance, imu_velocity, current_r_base):
        camera_noise = current_r_base * (camera_distance ** 2)
        R = np.array([[camera_noise, 0.0],
                      [0.0, self.r_imu]])
        # Measurement vector from our two different sensors
        z = np.array([[camera_distance],
                      [imu_velocity]])
        with self.lock:
            y = z - np.dot(self.H, self.x)
            S = np.dot(np.dot(self.H, self.P), self.H.T) + R
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
            self.x = self.x + np.dot(K, y)
            I = np.eye(self.P.shape[0])
            self.P = np.dot((I - np.dot(K, self.H)), self.P)
            current_distance = self.x[0, 0]
        return current_distance  # Return fused distance

    def update_velocity_only(self, imu_velocity):
        """
        Runs a 1D update when only IMU data is available (No camera frame).
        """
        # H is 1x2: [0, 1] maps to [Distance, Velocity]
        H_vel = np.array([[0.0, 1.0]])

        # R is 1x1: Just the IMU noise
        R_vel = np.array([[self.r_imu]])

        # z is 1x1: The measurement
        z = np.array([[imu_velocity]])
        with self.lock:
            y = z - np.dot(H_vel, self.x)
            S = np.dot(np.dot(H_vel, self.P), H_vel.T) + R_vel
            K = np.dot(np.dot(self.P, H_vel.T), np.linalg.inv(S))
            self.x = self.x + np.dot(K, y)
            I = np.eye(self.P.shape[0])
            self.P = np.dot((I - np.dot(K, H_vel)), self.P)
            current_distance = self.x[0, 0]

        return current_distance  # Return the newly corrected distance
