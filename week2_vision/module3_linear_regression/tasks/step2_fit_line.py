"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Step 2: Fit a Line (Least Squares)
Fit y = m*x + b to the colored line pixels with linear regression.
"""

import bisect
# -- Course setup: makes the shared `neo_lab` helper importable.
#    You don't need to read or change this block. --
import os as _os
import sys as _sys

import cv2.aruco
import numpy as np

import drone_core
import drone_utils
import pyrealsense2 as rs

_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab
from . import PDControl
from . import line_util as lu

# from . import cam_util as cu

# -- Constants --------------------------------------------------------------
S_MIN = 200
MIN_PIXELS = 200
ADVANCE_PITCH = 0.1  # fly forward off the spawn pad to reach the line
ADVANCE_TIME = 8.0  # seconds of forward flight before fitting
K_CURVE = 0.1
COL_CENTER = 320
MARKER_LENGTH = 38 * 7 / 1000
obj_points = np.array([
    [-MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
    [MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
    [MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0],
    [-MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0]
], dtype=np.float32)

# -- Module-level state -----------------------------------------------------
_timer = 0.0
_done = False
_lap = 0
_return_timer = 0.0
_prev_roll_err = 0.0

full_controller = PDControl.FullController(kp_yaw=0.5, kp_alt=1, max_yaw = 1, max_throttle=0.8)
roll_controller = PDControl.PDController(4.0, 0.0, 0.2)

context = rs.context()
devices = context.query_devices()
device = devices[0]
camera_matrix = None
dist_coeffs = None

mode = "None"


def reset():
    global _timer, _done
    _timer = 0.0
    _done = False


def run_line_follow(drone):
    global _timer, _done, _was_green, _lap, ADVANCE_PITCH, _prev_roll_err, _return_timer, mode
    dt = drone.get_delta_time()
    _image = drone.camera.get_downward_image_async()
    image = cv2.resize(_image, (640, 480), interpolation=cv2.INTER_LINEAR)
    _timer += dt
    _mask = neo_lab.bright_mask(image, S_MIN)
    mask = lu.get_largest_component_optimized(_mask)
    points = np.argwhere(mask == 255)
    if len(points) < MIN_PIXELS or _return_timer != 0:
        _return_timer += dt
        if _return_timer > 3 and abs(_prev_roll_err) < 160:
            _return_timer = 0.0
            return False
        else:
            if len(points) > MIN_PIXELS and _return_timer > 2:
                directions, means = lu.fit_lines(points)
                roll_err = means[0][0] - COL_CENTER
                _prev_roll_err = roll_err
            else:
                roll_err = _prev_roll_err
            normalized_roll_err = roll_err / COL_CENTER
            roll = roll_controller.calculate_position(normalized_roll_err, dt)
            drone.flight.send_pcmd(0, roll, 0, 0)
            return False
    directions, means = lu.fit_lines(points)
    angles = np.arctan(directions[:, 0] / directions[:, 1])
    angles = np.degrees(angles)
    curvature = angles[0] - angles[3]
    roll_err = means[3][0] - COL_CENTER
    target_angle = angles[3]
    target_angle -= roll_err / 30
    if abs(curvature) < 30 and abs(roll_err) < 160:
        curvature = 0
        ADVANCE_PITCH = 0.4
        roll_controller.kp = 1.5
        roll_controller.max_output = 1
        mode = "Straight"
    else:
        ADVANCE_PITCH = 0.4
        roll_controller.kp = 1.5
        if abs(roll_err) > 160:
            ADVANCE_PITCH = 0.4
            curvature = 0.0
            roll_controller.kp = 1
            mode = "Roll Correct"
        roll_controller.max_output = 1
        mode = "Curve"
    curvature_ff = -curvature * 0.005
    full_controller.set_setpoint(_alt=0.5)
    normalized_roll_err = roll_err / COL_CENTER
    roll = -roll_controller.calculate_position(normalized_roll_err, dt) + curvature_ff
    print(f"roll err norm: {normalized_roll_err}, target yaw: {target_angle}")
    roll = drone_utils.clamp(roll, -10, 10)
    output = full_controller.calculate(_alt=drone.physics.get_altitude(),
                                       _alt_vel=drone.physics.get_linear_velocity()[1], _yaw=target_angle, dt=dt)
    # print(f"roll={roll}, throttle={output[3]}, pitch={ADVANCE_PITCH}, yaw={output[2]}")
    drone.flight.send_pcmd(ADVANCE_PITCH, roll, output[2], output[3])
    _prev_roll_err = roll_err


def run_aruco_detect(drone):
    global selection, dist_coeffs, camera_matrix
    image = drone.camera.get_color_image_async()
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
                                       cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if camera_matrix is None:
        color_sensor = None
        for sensor in device.query_sensors():
            if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                color_sensor = sensor
                break
        for profile in color_sensor.get_stream_profiles():
            if profile.is_video_stream_profile():
                v_profile = profile.as_video_stream_profile()

                # Target the specific color stream type
                if v_profile.stream_type() == rs.stream.color:
                    intrinsics = v_profile.get_intrinsics()
                    dist_coeffs = intrinsics.coeffs
                    camera_matrix = np.array([
                        [intrinsics.fx, 0, intrinsics.ppx],
                        [0, intrinsics.fy, intrinsics.ppy],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    break
    if ids is not None:
        for i, corner in enumerate(corners):
            success, rvec, tvec = cv2.solvePnP(
                obj_points, corner[0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if success:
                x, y, z = tvec.flatten()
                print(f"xyz: {x, y, z}")
    print(f"id: {ids}")


def update(drone):
    global _timer, _done, _was_green, _lap, ADVANCE_PITCH, _prev_roll_err, _return_timer
    if _done:
        return True
    drone.flight.stop()  # hover in place

    # run_aruco_detect(drone)
    run_line_follow(drone)
    # full_controller.set_setpoint(_fwd=5, _rgt=0, _yaw=0, _alt=0.5)
    # output = full_controller.calculate(_fwd=drone.physics.get_position()[0], _rgt=drone.physics.get_position()[2], _alt=drone.physics.get_altitude(), _alt_vel=drone.physics.get_linear_velocity()[1], _yaw=drone.physics.get_attitude()[1])
    # drone.flight.send_pcmd(output[0], output[1], output[2], output[3])
    # print(f"alt={drone.physics.get_altitude()}")

    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(3.0)


    def start():
        _launcher.reset()
        reset()
        print("Step 2: Fit a Line (Least Squares)")


    def _update():
        if not _launcher.done:  # arm + climb to a safe height first
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()


    _drone.set_start_update(start, _update)
    _drone.go()
