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
