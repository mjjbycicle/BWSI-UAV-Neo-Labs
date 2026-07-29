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

    def __call__(self, current_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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