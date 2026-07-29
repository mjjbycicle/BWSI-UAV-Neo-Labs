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
    direction = eigenvectors[:, -1]

    return direction, centroid


def fit_lines(_points, prev_bottom_mean=None, num_segments = 4):
    # points = downsample_points_grid(_points, target_points=1000)
    points = _points
    n_points = len(points)
    k_neighbors = min(5, n_points)
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k_neighbors)

    # 2. Construct a sparse graph matrix
    # Construct rows, cols, and data for a coordinate list (COO) format sparse matrix
    rows = np.repeat(np.arange(n_points), k_neighbors)
    cols = indices.ravel()
    data = distances.ravel()

    # Create the sparse adjacency matrix
    sparse_graph = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

    # 3. Find the sequential path using a Minimum Spanning Tree (MST)
    mst = minimum_spanning_tree(sparse_graph)

    # Calculate degree of each vertex to find the endpoints (points with degree == 1)
    mst_coo = mst.tocoo()
    degrees = np.bincount(mst_coo.row, minlength=n_points) + np.bincount(mst_coo.col, minlength=n_points)
    endpoints = np.where(degrees == 1)[0]

    # Fallback to index 0 if no clear endpoint is found due to isolated structures
    start_node = endpoints[0] if len(endpoints) > 0 else 0

    # Trace the sequential path from the start node using Depth First Search
    ordered_indices = depth_first_order(mst, directed=False, i_start=start_node)[0]
    sorted_points = points[ordered_indices]

    # 4. Calculate cumulative path lengths
    diffs = np.diff(sorted_points, axis=0)
    step_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cumulative_lengths = np.hstack(([0], np.cumsum(step_lengths)))
    total_length = cumulative_lengths[-1]

    # 5. Split into equal segments based on length thresholds
    thresholds = np.linspace(1 / num_segments, 1, num_segments, endpoint=False) * total_length
    segment_labels = np.digitize(cumulative_lengths, thresholds)

    # 6. Unpack into individual arrays
    segments = [np.array(sorted_points[segment_labels == i]) for i in range(num_segments)]

    dq1, mq1 = fit_line(segments[0][:,1], segments[0][:,0])
    dq2, mq2 = fit_line(segments[1][:,1], segments[1][:,0])
    dq3, mq3 = fit_line(segments[2][:,1], segments[2][:,0])
    dq4, mq4 = fit_line(segments[3][:,1], segments[3][:,0])

    directions = np.array([dq1, dq2, dq3, dq4])
    means = np.array([mq1, mq2, mq3, mq4])

    ###### END PUT CODE HERE #########
    ##################################
    return correct_directions(directions, means, prev_bottom_mean)

def correct_directions(directions, means, prev_bottom_mean=None): # 0 is furthest from drone, 3 is closest to drone
    if prev_bottom_mean is None:
        prev_bottom_mean = [420, 320]
    d0 = math.hypot(means[0][0] - prev_bottom_mean[0], means[0][1] - prev_bottom_mean[1])
    d3 = math.hypot(means[3][0] - prev_bottom_mean[0], means[3][1] - prev_bottom_mean[1])
    if d0 < d3: # flip line
        directions = directions[::-1]
        means = means[::-1]
    for i in range(1, 4):
        prev_mean = means[i-1]
        prev_direction = directions[i-1]
        curr_mean = means[i]
        curr_direction = directions[i]
        if is_right_of_line(curr_mean, prev_mean, prev_direction):
            if curr_direction[0] > 0 and curr_direction[1] > 0:
                directions[i] = -directions[i]
        else:
            if curr_direction[0] > 0 > curr_direction[1]:
                directions[i] = -directions[i]
    return directions, means

def is_right_of_line(point, line_point, line_direction):
    vector_to_point = point - line_point
    cross_product = np.cross(line_direction, vector_to_point)
    return cross_product > 0

def get_green_fraction(image):
    lower_green = np.array([0, 50, 0])      # Lower limit for Green
    upper_green = np.array([100, 255, 100]) # Upper limit allowing for Blue and Red noise
    mask = cv2.inRange(image, lower_green, upper_green)
    green_fraction = mask[mask > 0].size / image.size
    return green_fraction


def get_largest_component_optimized(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    if num_labels > 1:
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        lut = np.zeros(num_labels, dtype=np.uint8)
        lut[largest_label] = 255
        return lut[labels]

    return np.zeros_like(binary_mask)