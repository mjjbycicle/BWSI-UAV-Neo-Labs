import math

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_order
from scipy.spatial import cKDTree
import numpy as np
import cv2
import neo_lab

IMAGE_HEIGHT = 480
MIN_PX = 200
TEST_PATH = "line_images/line_test_image1.jpeg"
OUTPUT_PATH = "line_images/debug_output.jpeg"
S_MIN = 230

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
    mask = neo_lab.bright_mask(image, S_MIN)
    mid_third = mask[int(IMAGE_HEIGHT / 3): int(IMAGE_HEIGHT * 2 / 3)]
    # mid_points = np.argwhere(mid_third==255)
    mid_points = get_points(mid_third)
    if mid_points is None: mid_points = np.argwhere(mid_third==255)
    points = np.argwhere(mask==255)
    if len(mid_points) < 200: mid_points = points
    direction, mean = fit_line(points[:, 1], points[:, 0])
    _direction, _mean = fit_line(mid_points[:, 1], mid_points[:, 0])
    mean[1] += IMAGE_HEIGHT / 3
    return direction, mean


def get_points(mask):

    MIN_COMPONENT_AREA = 4000

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

    return points


def debug(image, points):
    if points is None:
        print("No points detected")
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

    direction, centroid = fit_lines(image)
    #direction, centroid = fit_line(points[:, 1], points[:, 0])
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
    

if __name__ == "__main__":
    image = cv2.imread(TEST_PATH)

    print("Starting...")
    if image is None:
        print("No image detected")
    else:
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        mask = neo_lab.bright_mask(image, S_MIN)
        # points = np.argwhere(mask == 255)
        points = get_points(mask)

        debug(image, points)
