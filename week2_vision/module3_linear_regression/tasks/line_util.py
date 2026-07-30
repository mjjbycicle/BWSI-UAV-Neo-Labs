import math

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
from scipy.spatial import cKDTree
import numpy as np
import cv2
import neo_lab

IMAGE_HEIGHT = 480
MIN_PX = 200

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

def fit_lines(image):
    mask = neo_lab.bright_mask(image, 210)
    mid_third = mask[IMAGE_HEIGHT / 3: IMAGE_HEIGHT * 2 / 3]
    mid_points = np.argwhere(mid_third==255)
    points = np.argwhere(mask==255)
    if len(mid_points) < 200: mid_points = points
    direction, _mean = fit_line(points[:, 1], points[:, 0])
    _direction, mean = fit_line(mid_points[:, 1], mid_points[:, 0])
    return direction, mean