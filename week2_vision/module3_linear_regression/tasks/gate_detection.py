import cv2
import numpy as np

TEST_PATH = "gate_images/aruco_test_image3.jpeg"
OUTPUT_PATH = "gate_images/debug_output.jpeg"

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)  #Type of aruco tag
ARUCO_PARAMS = cv2.aruco.DetectorParameters()                           #Default params in cv2
DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

#Idk how accurate the values are below, they were taken from a prev lab
FOCAL_PX = 320.0            # Camera focal length in pixels (approx calibration)
REAL_TAG_SIZE = 0.19        # Physical corner-tag side length, meters (approx)
MAX_DETECTION_DIST = 4.0    # Distance threshold to detect tags



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
        cv2.circle(image, (int(center[0]), int(center[1])), radius = 32, color = (0, 0, 255), thickness = -1)


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
        dist = FOCAL_PX * REAL_TAG_SIZE / np.mean(side_lengths)
        if dist < MAX_DETECTION_DIST: distances.append(dist)

    return np.mean(distances) # Averages the distances between each marker and the drone



"""
Returns the (x, y) coord of the center of detected markers (center of the gate). Returns None if a coord can't be determined
If 3 or 4 markers are detected, the center coord will be able to be found
If 2 markers are detected in a horizontal line or vertical line, the center coord will be able to be found
If 2 markers are detected diagonally, a center coord can't be determined so None will be returned
If 1 marker is detected, a center coord can't be determined so None will be returned
""" 
def center_coord(corners, ids):
    centers = np.array([c.mean(axis = 1)[0] for c in corners])

    if len(ids) == 4:
        return np.mean(centers, axis = 0)
    
    elif len(ids) == 3:
        a = np.linalg.norm(centers[0] - centers[1])
        b = np.linalg.norm(centers[0] - centers[2])
        c = np.linalg.norm(centers[1] - centers[2])

        if a > b and a > c:
            return (centers[0] + centers[1]) / 2
        elif b > a and b > c:
            return (centers[0] + centers[2]) / 2
        else:
            return (centers[1] + centers[2]) / 2
        
    elif len(ids) == 2:
        dx = abs(centers[0][0] - centers[1][0])
        dy = abs(centers[0][1] - centers[1][1])

        if dx > 2.75 * dy or dy > 2.75 * dx: # Horizontal or vertical tags
            return np.mean(centers, axis = 0)

    return None


def detect_gates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = DETECTOR.detectMarkers(gray)

    if ids is None:
        return None

    # Note: distance and center coord may be messed up if the camera detects a aruco tag from another gate
    distance = marker_distance(corners)
    center = center_coord(corners, ids) # center will be an array (x, y)

    debug(corners, ids, rejected, image, center)
    return Gate(int(corners[0]), int(corners[1]), [int(i) for i in ids.flatten()], distance)

class Gate:
    """A gate located from its corner ArUco tags: image center (cx, cy), inter-tag span
    (gate size proxy), mean tag pixel size (a proximity signal that works with one tag),
    and the decoded corner-tag ids."""

    def __init__(self, cx, cy, ids, distance):
        self.cx = cx
        self.cy = cy
        self.ids = ids
        self.distance = distance
        self.count = len(ids)


def main():
    print("Started...")
    image = cv2.imread(TEST_PATH)
    if image is not None:
        detect_gates(image)
    else:
        print("Image not found")

if __name__ == "__main__":
    main()