import cv2
import numpy as np
from . import filter_util as fu
# import filter_util as fu

TEST_PATH = "gate_images/aruco_test_image8.jpeg"
OUTPUT_PATH = "gate_images/debug_output.jpeg"

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)  # Type of aruco tag
ARUCO_PARAMS = cv2.aruco.DetectorParameters()  # Default params in cv2
DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# Idk how accurate the values are below
FOCAL_PX = 615.3  # Camera focal length in pixels (approx calibration)
REAL_TAG_SIZE = 0.19  # Physical corner-tag side length, meters (approx)
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
REAL_GATE_DIAMETER = 1.8
REAL_GATE_RADIUS = REAL_GATE_DIAMETER / 2
REAL_GATE_DIAGONAL = ((REAL_GATE_RADIUS ** 2) * 2) ** 0.5

HORIZONTAL_TAGS = [4, 8] #left, right ...
VERTICAL_TAGS = [7, 3] #top, bottom ...
TOP_TAGS = VERTICAL_TAGS[::2]
BOTTOM_TAGS = VERTICAL_TAGS[1::2]
LEFT_TAGS = HORIZONTAL_TAGS[::2]
RIGHT_TAGS = HORIZONTAL_TAGS[1::2]
CORNER_IDX = [TOP_TAGS, RIGHT_TAGS, BOTTOM_TAGS, LEFT_TAGS]
GATES = [[3, 4, 7, 8]]
NUM_GATES = len(GATES)


ID_TO_GATE_ID = dict()
ID_TO_CORNER_IDX = dict()
for i in range(len(GATES)):
    for j in GATES[i]:
        ID_TO_GATE_ID[j] = i
for i in range(len(CORNER_IDX)):
    for j in CORNER_IDX[i]:
        ID_TO_CORNER_IDX[j] = i

# 3D points matching the 0-3 index order: [Top, Right, Bottom, Left]
# Origin is the center of the gate
OBJ_POINTS_FULL = np.array([
    [0.0, REAL_GATE_RADIUS, 0.0],  # Index 0: Top
    [REAL_GATE_RADIUS, 0.0, 0.0],  # Index 1: Right
    [0.0, -REAL_GATE_RADIUS, 0.0],  # Index 2: Bottom
    [-REAL_GATE_RADIUS, 0.0, 0.0]  # Index 3: Left
], dtype=np.float32)

# 2. Camera Intrinsics (Example values for 640x480, replace with your RealSense data)
camera_matrix = np.array([
    [FOCAL_PX, 0.0, 320.0],  # fx, 0, cx
    [0.0, FOCAL_PX, 240.0],  # 0, fy, cy
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# RealSense distortion coefficients (usually near zero if pre-rectified, but good to include)
dist_coeffs = np.zeros((4, 1))


# Not rly needed but it makes the borders in debug output easier to see (made by the good ol' chatgpt)
def draw_marker_borders(image, corners, color=(0, 255, 0), thickness=5):
    for marker_corners in corners:
        pts = marker_corners.reshape(4, 2).astype(int)

        for i in range(4):
            cv2.line(
                image,
                tuple(pts[i]),
                tuple(pts[(i + 1) % 4]),
                color,
                thickness
            )


# Creates a red dot at where it thinks the center is
def draw_center_point(image, center):
    if center is not None:
        cv2.circle(image, (int(center[0]), int(center[1])), radius=32, color=(0, 0, 255), thickness=-1)


# Shows the output of the image detection
def debug(corners, ids, rejected, image, center):
    if ids is None:
        print("No markers detected")
        cv2.aruco.drawDetectedMarkers(image, rejected, borderColor=(0, 0, 255))
        cv2.imwrite(OUTPUT_PATH, image)
    else:
        print(f"{len(ids)} markers detected")
        cv2.aruco.drawDetectedMarkers(image, corners, ids, borderColor=(0, 255, 0))
        draw_marker_borders(image, corners, thickness=16)
        draw_center_point(image, center)
        cv2.imwrite(OUTPUT_PATH, image)

"""
# Calculates the distance each marker is from the camera in meters
def marker_distance(corners):
    distances = []

    for c in corners: # Averages the side length of each marker in pixels and calculates the distance from the camera
        points = c.reshape(-1, 2)
        side_lengths = np.array([
            np.linalg.norm(points[0] - points[1]), # Top
            np.linalg.norm(points[1] - points[2]), # Right
            np.linalg.norm(points[2] - points[3]), # Bottom
            np.linalg.norm(points[3] - points[0])  # Left
        ])
        distances.append(FOCAL_PX * REAL_TAG_SIZE / np.mean(side_lengths))

    return np.mean(distances)
"""
# This function is never used but its pretty self explanatory
def calculate_pixel_distance(tag_a, tag_b):
    """Helper for the 2-tag wide baseline math."""
    pt1 = np.array(tag_a.center_pixel)
    pt2 = np.array(tag_b.center_pixel)
    return np.linalg.norm(pt1 - pt2)


def process_two_tags_opposite(tag_a, tag_b):
    fx, fy = camera_matrix[0,0], camera_matrix[1,1]
    cx, cy = camera_matrix[0,2], camera_matrix[1,2]

    # 1. Pixel center of the gate (midpoint formula)
    u_center = (tag_a.center_pixel[0] + tag_b.center_pixel[0]) / 2.0
    v_center = (tag_a.center_pixel[1] + tag_b.center_pixel[1]) / 2.0

    # 2. Distance (Z) using wide baseline (REAL_GATE_DIAMETER)
    pixel_dist = np.linalg.norm(np.array(tag_a.center_pixel) - np.array(tag_b.center_pixel))
    z_distance = (fx * REAL_GATE_DIAMETER) / pixel_dist  # Assuming fx and fy are similar

    # 3. Un-project to find physical X and Y relative to camera
    x_relative = (u_center - cx) * z_distance / fx
    y_relative = (v_center - cy) * z_distance / fy

    corner_idx_product = (ID_TO_CORNER_IDX[tag_a.tag_id] + 1) * (ID_TO_CORNER_IDX[tag_b.tag_id] + 1)
    if corner_idx_product == 2:
        x_relative -= REAL_GATE_RADIUS / 2
        y_relative += REAL_GATE_RADIUS / 2
    elif corner_idx_product == 6:
        x_relative -= REAL_GATE_RADIUS / 2
        y_relative -= REAL_GATE_RADIUS / 2
    elif corner_idx_product == 12:
        x_relative += REAL_GATE_RADIUS / 2
        y_relative -= REAL_GATE_RADIUS / 2
    else:
        x_relative += REAL_GATE_RADIUS / 2
        y_relative += REAL_GATE_RADIUS / 2

    relative_height = - y_relative

    return relative_height, z_distance, x_relative


def process_two_tags_diagonal(tag_a, tag_b):
    fx, fy = camera_matrix[0,0], camera_matrix[1,1]
    cx, cy = camera_matrix[0,2], camera_matrix[1,2]

    # 1. Pixel center of the gate (midpoint formula)
    u_center = (tag_a.center_pixel[0] + tag_b.center_pixel[0]) / 2.0
    v_center = (tag_a.center_pixel[1] + tag_b.center_pixel[1]) / 2.0

    # 2. Distance (Z) using wide baseline (1.5m)
    pixel_dist = np.linalg.norm(np.array(tag_a.center_pixel) - np.array(tag_b.center_pixel))
    z_distance = (fx * REAL_GATE_DIAGONAL) / pixel_dist  # Assuming fx and fy are similar

    # 3. Un-project to find physical X and Y relative to camera
    x_relative = (u_center - cx) * z_distance / fx
    y_relative = (v_center - cy) * z_distance / fy

    # OpenCV Y is DOWN. Invert it so positive is UP.
    relative_height = -y_relative

    return relative_height, z_distance, x_relative


def process_single_tag(tag, corner_idx):
    fx, fy = camera_matrix[0,0], camera_matrix[1,1]
    cx, cy = camera_matrix[0,2], camera_matrix[1,2]

    # 1. Distance (Z) using single tag width (REAL_TAG_SIZE)
    z_distance = (fx * REAL_TAG_SIZE) / tag.pixel_width

    # 2. Un-project the tag's center pixel to find the tag's physical location
    u_tag, v_tag = tag.center_pixel
    x_tag = (u_tag - cx) * z_distance / fx
    y_tag = (v_tag - cy) * z_distance / fy

    # Invert Y so positive is UP
    tag_relative_height = -y_tag
    tag_lateral_offset = x_tag

    # 3. Apply the physical offset to find the Gate Center
    # corner_idx: 0=Top, 1=Right, 2=Bottom, 3=Left
    gate_relative_height = tag_relative_height
    gate_lateral_offset = tag_lateral_offset

    if corner_idx == 0:   # Top Tag: Gate is REAL_GATE_RADIUSm below it
        gate_relative_height -= REAL_GATE_RADIUS
    elif corner_idx == 2: # Bottom Tag: Gate is REAL_GATE_RADIUSm above it
        gate_relative_height += REAL_GATE_RADIUS
    elif corner_idx == 1: # Right Tag: Gate is REAL_GATE_RADIUSm to the left
        gate_lateral_offset -= REAL_GATE_RADIUS
    elif corner_idx == 3: # Left Tag: Gate is REAL_GATE_RADIUSm to the right
        gate_lateral_offset += REAL_GATE_RADIUS

    return gate_relative_height, z_distance, gate_lateral_offset


def process_frame(detected_tags, current_time, altitude, forward_velocity):
    gates_in_view = dict()
    gate_measurements = dict()
    for tag in detected_tags:
        gate_id = ID_TO_GATE_ID[tag.id]
        corner_idx = ID_TO_CORNER_IDX[tag.id]
        if gate_id not in gates_in_view:
            gates_in_view[gate_id] = {}
        gates_in_view[gate_id][corner_idx] = tag


    # 2. Process each gate we can see
    for gate_id, tags in gates_in_view.items():
        tag_count = len(tags)

        # We need matched arrays for solvePnP
        # We only append the 3D and 2D points for the corners we actually see!
        obj_points_visible = []
        img_points_visible = []

        for corner_idx in sorted(tags.keys()):
            obj_points_visible.append(OBJ_POINTS_FULL[corner_idx])
            img_points_visible.append(tags[corner_idx].center_pixel)

        obj_points_visible = np.array(obj_points_visible, dtype=np.float32)
        img_points_visible = np.array(img_points_visible, dtype=np.float32)

        # 3. The Switchboard Logic
        if tag_count == 4:
            # Best case: Full 6DoF Pose
            success, rvec, tvec = cv2.solvePnP(
                obj_points_visible, img_points_visible,
                camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
            )
            lateral_offset = tvec[0][0]
            relative_height = -tvec[1][0]
            true_distance = np.linalg.norm(tvec)
            gate_measurements[gate_id] = GateMeasurement(lateral_offset, relative_height + altitude, true_distance, forward_velocity, tag_count, current_time)

        elif tag_count == 3:
            # Good case: 3 points visible
            success, rvec, tvec = cv2.solvePnP(
                obj_points_visible, img_points_visible,
                camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_AP3P
            )
            lateral_offset = tvec[0][0]
            relative_height = -tvec[1][0]
            true_distance = np.linalg.norm(tvec)
            gate_measurements[gate_id] = GateMeasurement(lateral_offset, relative_height + altitude, true_distance, forward_velocity, tag_count, current_time)

        elif tag_count == 2:
            keys = list(tags.keys())
            if abs(keys[0] - keys[1]) == 2:
                rel_h, z_dist, lat_off = process_two_tags_opposite(tags[keys[0]], tags[keys[1]])
                gate_measurements[gate_id] = GateMeasurement(lat_off, rel_h + altitude, z_dist, forward_velocity, tag_count, current_time)
            else:
                first_key = keys[0]
                rel_h, z_dist, lat_off = process_two_tags_diagonal(tags[keys[0]], tags[keys[1]])
                gate_measurements[gate_id] = GateMeasurement(lat_off, rel_h + altitude, z_dist, forward_velocity, tag_count, current_time)

        elif tag_count == 1:
            first_key = list(tags.keys())[0]
            rel_h, z_dist, lat_off = process_single_tag(tags[first_key], first_key)
            gate_measurements[gate_id] = GateMeasurement(lat_off, rel_h + altitude, z_dist, forward_velocity, tag_count, current_time)
    return gate_measurements


def detect_gates(image, current_time, altitude, forward_velocity):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = DETECTOR.detectMarkers(gray)

    if ids is None:
        return None

    detected_tags = parse_aruco_returns(corners, ids)
    gate_measurements = process_frame(detected_tags, current_time, altitude, forward_velocity)
    return gate_measurements


class Gate:
    """A gate located from its corner ArUco tags: image center (cx, cy), inter-tag span
    (gate size proxy), mean tag pixel size (a proximity signal that works with one tag),
    and the decoded corner-tag ids."""

    def __init__(self, init_time):
        self.lateral_offset_filter = fu.GatePositionKalmanFilter(0.05, 0.0, r_base=0.1, q_base=0.05)
        self.altitude_filter = fu.GatePositionKalmanFilter(0.05, 0.0, r_base=0.1, q_base=0.0005)
        self.distance_filter = fu.DistanceSensorFusionFilter(0.05, 1.0, 0.0)
        self.prev_time = init_time

    def update(self, gate_measurement):
        dt = gate_measurement.current_time - self.prev_time
        self.prev_time = gate_measurement.current_time
        if gate_measurement.tag_count == 4:
            self.lateral_offset_filter.update(gate_measurement.lateral_offset_measurement, gate_measurement.distance_measurement, 0.005)
            self.altitude_filter.update(gate_measurement.altitude_measurement, gate_measurement.distance_measurement, 0.005)
            self.distance_filter.update(gate_measurement.distance_measurement, gate_measurement.forward_velocity, 0.005)

        elif gate_measurement.tag_count == 3:
            self.lateral_offset_filter.update(gate_measurement.lateral_offset_measurement, gate_measurement.distance_measurement, 0.03)
            self.altitude_filter.update(gate_measurement.altitude_measurement, gate_measurement.distance_measurement, 0.03)
            self.distance_filter.update(gate_measurement.distance_measurement, gate_measurement.forward_velocity, 0.03)

        elif gate_measurement.tag_count == 2:
            self.lateral_offset_filter.update(gate_measurement.lateral_offset_measurement, gate_measurement.distance_measurement, 0.02)
            self.altitude_filter.update(gate_measurement.altitude_measurement, gate_measurement.distance_measurement, 0.02)
            self.distance_filter.update(gate_measurement.distance_measurement, gate_measurement.forward_velocity, 0.02)

        elif gate_measurement.tag_count == 1:
            self.lateral_offset_filter.update(gate_measurement.lateral_offset_measurement, gate_measurement.distance_measurement, 0.08)
            self.altitude_filter.update(gate_measurement.altitude_measurement, gate_measurement.distance_measurement, 0.08)
            self.distance_filter.update(gate_measurement.distance_measurement, gate_measurement.forward_velocity, 0.08)

    def update_forward_velocity(self, forward_velocity):
        self.distance_filter.update_velocity_only(forward_velocity)

    def reset_lateral_offset(self):
        self.lateral_offset_filter.reset()

    def predict(self, current_time):
        dt = current_time - self.prev_time
        self.prev_time = current_time
        self.lateral_offset_filter.predict(dt)
        self.altitude_filter.predict(dt)

    def __str__(self):
        return f"height: {self.altitude_filter.x[0, 0]}, lateral offset: {self.lateral_offset_filter.x[0, 0]}"

class GateMeasurement:
    def __init__(self, lateral_offset_measurement, altitude_measurement, distance_measurement, forward_velocity, tag_count, current_time):
        self.lateral_offset_measurement = lateral_offset_measurement
        self.altitude_measurement = altitude_measurement
        self.distance_measurement = distance_measurement
        self.forward_velocity = forward_velocity
        self.tag_count = tag_count
        self.current_time = current_time


class Tag:
    def __init__(self, tag_id, center_pixel, pixel_width):
        self.id = tag_id
        self.center_pixel = center_pixel
        self.pixel_width = pixel_width


def parse_aruco_returns(corners, ids):
    """
    Converts OpenCV's raw ArUco outputs into clean Tag objects.
    """
    detected_tags = []
    if ids is None:
        return detected_tags
    for i in range(len(ids)):
        tag_id = int(ids[i])
        marker_corners = corners[i][0]
        center_x = np.mean(marker_corners[:, 0])
        center_y = np.mean(marker_corners[:, 1])
        center_pixel = (center_x, center_y)
        top_left = marker_corners[0]
        top_right = marker_corners[1]
        pixel_width = np.linalg.norm(top_left - top_right)
        detected_tags.append(Tag(tag_id, center_pixel, pixel_width))

    return detected_tags