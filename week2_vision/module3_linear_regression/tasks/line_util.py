import math

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
from scipy.spatial import cKDTree
import numpy as np
import cv2
import neo_lab

def downsample_points_grid(points, target_points=1500):
    x_range, y_range = np.ptp(points, axis=0)
    grid_size = np.sqrt((x_range * y_range) / target_points)
    grid_coords = np.floor(points / grid_size).astype(int)
    _, unique_indices = np.unique(grid_coords, axis=0, return_index=True)

    return points[unique_indices]


def fit_line(x, y):
    mx = np.mean(x)
    my = np.mean(y)
    centroid = np.array([mx, my])
    dx = x - mx
    dy = y - my
    sxx = np.dot(dx, dx)
    syy = np.dot(dy, dy)
    sxy = np.dot(dx, dy)
    cov_matrix = np.array([[sxx, sxy],
                           [sxy, syy]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    direction = np.array(eigenvectors[:, -1])

    return direction, centroid