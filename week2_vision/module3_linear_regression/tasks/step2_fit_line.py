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

from camera import Camera

_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab
from . import PDControl
from . import line_util as lu
from . import filter_util as fu
from . import gate_detection as gd
import threading
import time

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
_prev_bottom_mean = None

full_controller = PDControl.FullController(kp_yaw=0.05, kp_alt=1, max_yaw=1, max_throttle=0.8)
roll_controller = PDControl.PDController(6.0, 0.0, 0.5)
direction_filter = fu.VectorExponentialLowPassFilter(0.1)
mean_filter = fu.VectorExponentialLowPassFilter(0.1)

mode = "None"

_vision_thread = None
_vision_running = False
_latest_cmd = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "throttle": 0.0}
_gate_thread = None
_gate_running = False
_closest_gate = None

def reset():
    global _timer, _done
    _timer = 0.0
    _done = False


def line_control_loop(drone):
    global _timer, _done, _lap, ADVANCE_PITCH, _prev_roll_err, _return_timer, mode, _prev_bottom_mean
    global _vision_running, _latest_cmd

    # Target 20 frames per second (0.05 seconds per loop)
    target_fps = 20.0
    loop_delay = 1.0 / target_fps

    last_time = time.time()

    while _vision_running:
        loop_start_time = time.time()

        # Calculate isolated delta time for the vision controllers
        dt = loop_start_time - last_time
        last_time = loop_start_time

        _image = drone.camera.get_downward_image_async()

        if _image is not None:
            image = cv2.resize(_image, (640, 480), interpolation=cv2.INTER_LINEAR)
            _mask = neo_lab.bright_mask(image, S_MIN)
            mask = lu.get_largest_component_optimized(_mask)
            points = np.argwhere(mask == 255)

            if len(points) < MIN_PIXELS or _return_timer != 0:
                _return_timer += dt
                if _return_timer > 3 and abs(_prev_roll_err) < 160:
                    _return_timer = 0.0
                    continue
                else:
                    if len(points) > MIN_PIXELS and _return_timer > 2:
                        directions, means = lu.fit_lines(points)
                        roll_err = means[0][0] - COL_CENTER
                        _prev_roll_err = roll_err
                    else:
                        roll_err = _prev_roll_err

                    normalized_roll_err = roll_err / COL_CENTER
                    roll = roll_controller.calculate_position(normalized_roll_err, dt)

                    # UPDATE THREAD STATE INSTEAD OF SENDING
                    _latest_cmd = {"pitch": 0.0, "roll": roll, "yaw": 0.0, "throttle": 0.0}
                    continue

            _directions, _means = lu.fit_lines(points, prev_bottom_mean=_prev_bottom_mean)
            directions = direction_filter(_directions)
            means = mean_filter(_means)
            _prev_bottom_mean = means[3]

            angles = np.arctan2(directions[:, 0], directions[:, 1])
            angles = np.degrees(angles)
            curvature = angles[0] - angles[3]
            roll_err = means[3][0] - COL_CENTER
            target_angle = angles[3]

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

            curvature_ff = -curvature * 0.008
            full_controller.set_setpoint(_alt=0.5)
            normalized_roll_err = roll_err / COL_CENTER
            roll = -roll_controller.calculate_position(normalized_roll_err, dt) + curvature_ff
            roll = drone_utils.clamp(roll, -10, 10)

            output = full_controller.calculate(_alt=drone.physics.get_altitude(),
                                               _alt_vel=drone.physics.get_linear_velocity()[1],
                                               _yaw=target_angle, dt=dt)

            # UPDATE THREAD STATE INSTEAD OF SENDING
            _latest_cmd = {"pitch": ADVANCE_PITCH, "roll": roll, "yaw": output[2], "throttle": output[3]}
            _prev_roll_err = roll_err

        # Small sleep to prevent this loop from maxing out a CPU core
        # --- THE RATE LIMITER ---
        # Calculate how long the math actually took
        math_duration = time.time() - loop_start_time

        # Sleep for whatever time is remaining to hit our exact 20Hz target
        time_to_sleep = loop_delay - math_duration
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)


def gate_detect_loop(drone):
    global _timer, _done, _lap, ADVANCE_PITCH, _prev_roll_err, _return_timer, mode, _prev_bottom_mean, _closest_gate
    global _vision_running, _latest_cmd

    # Target 20 frames per second (0.05 seconds per loop)
    target_fps = 20.0
    loop_delay = 1.0 / target_fps

    last_time = time.time()

    while _vision_running:
        loop_start_time = time.time()

        # Calculate isolated delta time for the vision controllers
        dt = loop_start_time - last_time
        last_time = loop_start_time

        _image = drone.camera.get_color_image_async()

        if _image is not None:
            image = cv2.resize(_image, (640, 480), interpolation=cv2.INTER_LINEAR)
            gate = gd.detect_gates(image)
            _closest_gate = gate
            print(gate)

        # Small sleep to prevent this loop from maxing out a CPU core
        # --- THE RATE LIMITER ---
        # Calculate how long the math actually took
        math_duration = time.time() - loop_start_time

        # Sleep for whatever time is remaining to hit our exact 20Hz target
        time_to_sleep = loop_delay - math_duration
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)



def update(drone):
    global _timer, _done, _vision_thread, _vision_running, _latest_cmd, _gate_running, _gate_thread

    if _done:
        _vision_running = False  # Tell the thread loop to shut down safely
        _gate_running = False
        return True

    # Initialize the background thread on the first tick
    if _vision_thread is None:
        _vision_running = True
        # Setting daemon=True ensures the thread dies automatically if the main program crashes
        _vision_thread = threading.Thread(target=line_control_loop, args=(drone,), daemon=True)
        _vision_thread.start()

    if _gate_thread is None:
        _gate_running = True
        _gate_thread = threading.Thread(target=gate_detect_loop, args=(drone,), daemon=True)
        _gate_thread.start()

    _timer += drone.get_delta_time()

    # Fast-publish the most recent command calculated by the vision thread
    drone.flight.send_pcmd(
        _latest_cmd["pitch"],
        _latest_cmd["roll"],
        _latest_cmd["yaw"],
        _latest_cmd["throttle"]
    )

    return _done