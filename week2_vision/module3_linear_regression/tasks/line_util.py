import math
from typing import Any

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
from scipy.spatial import cKDTree
import numpy as np
import cv2
import neo_lab

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

MIN_PX = 200
TEST_PATH = "nadir test.jpg"
OUTPUT_PATH = "line_images/debug_output.jpeg"
V_MIN = 210
S_MAX = 20
S_MIN = 200

TOP_PERCENT = 0.5 # Which percent of the top part of the image gets weighted
TOP_WEIGHT = 4.0  # How much more weight the top part is given compared to the bottom

TEST_PATH = "line_images/line_test_image8.jpeg"
OUTPUT_PATH = "line_images/debug_output.jpeg"


def downsample_points_grid(points, target_points=1500):
    x_range, y_range = np.ptp(points, axis=0)
    grid_size = np.sqrt((x_range * y_range) / target_points)
    grid_coords = np.floor(points / grid_size).astype(int)
    _, unique_indices = np.unique(grid_coords, axis=0, return_index=True)

    return points[unique_indices]


# Computes the line of best fit and mean of all points
# GPT made it so that some points (some fraction of the top image) is weighted more
def fit_line(x, y):
    mx = np.mean(x)
    my = np.mean(y)
    dx = x - mx
    dy = y - my

    weights = np.ones(len(y))

    # Weight upper portion of image more
    top_threshold = np.max(y) * TOP_PERCENT
    weights[y < top_threshold] = TOP_WEIGHT

    # Weighted covariance
    sxx = np.sum(weights * dx * dx)
    syy = np.sum(weights * dy * dy)
    sxy = np.sum(weights * dx * dy)

    cov_matrix = np.array([[sxx, sxy],
                           [sxy, syy]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    direction = np.array(eigenvectors[:, -1])

    # Make centroid weighted too
    centroid = np.array([
        np.sum(weights * x) / np.sum(weights),
        np.sum(weights * y) / np.sum(weights)
    ])

    # centroid = np.array([mx, my])

    return direction, centroid


def fit_lines(image):
    points = get_inter_points(image)
    direction, mean = fit_line(points[:, 1], points[:, 0])
    return direction, mean


def get_inter_points(image):
    v_mask = neo_lab.bright_mask(image, V_MIN)
    s_mask = neo_lab.saturated_mask(image, S_MAX)
    points = get_points(v_mask)
    s_points = np.argwhere(s_mask==255)
    if points is None: points = np.argwhere(v_mask==255)
    # points = coord_intersection(v_points, s_points)
    return points


def get_points(mask):
    MIN_COMPONENT_AREA = 700

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    points = []

    # Skip label 0 because it's the background
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= MIN_COMPONENT_AREA:
            component_points = np.argwhere(labels == label)
            points.append(component_points)

    if len(points) > 0:
        points = np.vstack(points)
    else:
        points = None

    points = get_line_points(points)

    return points


# Made by GPT
# Looks at each individual blob of points and only returns the blob of points that it think is the main line
# Blobs are weighted more based on size, distance to center of image, and distance to the bottom
def get_line_points(points):
    if points is None:
        return None
    
    # Create a temporary mask from points for connected components
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    mask[points[:, 0], points[:, 1]] = 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    best_label = None
    best_score = -float("inf")

    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 20:
            continue

        cx, cy = centroids[i]

        # Normalize distances
        center_distance = abs(cx - IMAGE_WIDTH / 2) / (IMAGE_WIDTH / 2)
        bottom_distance = (IMAGE_HEIGHT - cy) / IMAGE_HEIGHT

        # Prefer large, centered, lower blobs
        score = (
            area
            - 100 * center_distance
            - 200 * bottom_distance
        )

        if score > best_score:
            best_score = score
            best_label = i

    if best_label is None:
        return points

    # Return only points belonging to selected component
    selected_points = points[labels[points[:, 0], points[:, 1]] == best_label]

    return selected_points


def debug(image, points):
    if points is None:
        # print("No points detected")
        return

    if len(points) < MIN_PX:
        print(f"Only {len(points)} pixels found")
    for point in points:
        cv2.circle(
            image,
            (int(point[1]), int(point[0])),
            5,
            (0, 0, 255),
            -1
        )

    # direction, centroid = fit_lines(image)
    direction, centroid = fit_line(points[:, 1], points[:, 0])
    cx, cy = int(centroid[0]), int(centroid[1])
    cv2.circle(
        image,
        (cx, cy),
        8,
        (255, 0, 0),
        -1
    )
    line_length = 200
    x1 = int(cx - direction[0] * line_length)
    y1 = int(cy - direction[1] * line_length)
    x2 = int(cx + direction[0] * line_length)
    y2 = int(cy + direction[1] * line_length)
    cv2.line(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.imwrite(OUTPUT_PATH, image)


def coord_intersection(coord1, coord2):
    if coord1 is None: return coord2
    if coord2 is None: return coord1
    void_dt = np.dtype((np.void, coord1.dtype.itemsize * coord1.shape[1]))
    v1 = np.ascontiguousarray(coord1).view(void_dt)
    v2 = np.ascontiguousarray(coord2).view(void_dt)
    _, idx, _ = np.intersect1d(v1, v2, return_indices=True)
    result = coord1[idx]
    return result


if __name__ == "__main__":
    image = cv2.imread(TEST_PATH)

    print("Starting...")
    if image is None:
        print("No image detected")
    else:
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)

        debug(image, get_inter_points(image))
