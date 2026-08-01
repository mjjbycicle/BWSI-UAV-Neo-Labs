import threading
import time

import cv2.aruco
import numpy as np

import drone_utils
from . import PDControl
from . import filter_util as fu
from . import gate_detection as gd
from . import line_util as lu
from . import neo_lab
from . import threading_util as tu

# -- Constants --------------------------------------------------------------
S_MIN = 220
MIN_PIXELS = 200
ADVANCE_PITCH = 0.1  # fly forward off the spawn pad to reach the line
ADVANCE_TIME = 8.0  # seconds of forward flight before fitting
K_CURVE = 0.1
COL_CENTER = 320
CLOSEST_GATE_THRESHOLD = 4.0
DRIFT_FF = 0.1
SIDE_CROP = 100
VERT_CROP = 120
GATE_TIME = 20
TARGET_LF_HEIGHT = 0.9

# -- Module-level state -----------------------------------------------------
_timer = 0.0
_done = False
_height_measurement = False
_through_dist = -10
_through_time = -1

full_controller = PDControl.FullController(kp_yaw=0.13, kd_yaw=0.12, kp_alt=1, max_yaw=1.0, max_throttle=0.8)
roll_controller = PDControl.PDController(0.6, 2.5, 0.2)
direction_filter = fu.VectorExponentialLowPassFilter(0.99)
mean_filter = fu.VectorExponentialLowPassFilter(0.99)

mode = "None"
_command_lock = threading.Lock()
_active_flight_mode = "LINE_FOLLOW"

_line_follow_thread = None
_line_follow_running = False
_latest_cmd = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "throttle": 0.0}
_gate_detect_thread = None
_gate_detect_running = False
_prev_closest_gate = None
_target_height = TARGET_LF_HEIGHT
_closest_dist = 0.0
_gates = dict()
for i in range(gd.NUM_GATES):
    _gates[i] = gd.Gate(0.0, i)
_dist_to_gate_int = None


def reset():
    global _timer, _done
    _timer = 0.0
    _done = False


def line_control_loop(drone):
    global _timer, _done, ADVANCE_PITCH, mode, _target_height
    global _line_follow_running, _latest_cmd

    _return_timer = 0.0
    _prev_roll_err = 0.0
    _prev_bottom_mean = None

    # Target 20 frames per second (0.05 seconds per loop)
    target_fps = 50.0
    loop_delay = 1.0 / target_fps

    last_time = time.time()

    while _line_follow_running:
        loop_start_time = time.time()

        # Calculate isolated delta time for the vision controllers
        dt = loop_start_time - last_time
        last_time = loop_start_time

        _image = drone.camera.get_downward_image_async()

        if _image is not None:
            image = cv2.resize(_image, (640, 480), interpolation=cv2.INTER_LINEAR)
            # image[:VERT_CROP, :] = 0
            image[image.shape[0] - 2 * VERT_CROP:, :] = 0
            # image[:, SIDE_CROP] = 0
            # image[:, image.shape[1] - SIDE_CROP] = 0
            mask = neo_lab.bright_mask(image, S_MIN)
            points = np.argwhere(mask == 255)

            if len(points) < MIN_PIXELS or _return_timer != 0:
                _return_timer += dt
                if _return_timer > 3 and abs(_prev_roll_err) < 160:
                    _return_timer = 0.0
                    continue
                else:
                    if len(points) > MIN_PIXELS and _return_timer > 2:
                        direction, means = lu.fit_line(points[:, 1], points[:, 0])
                        roll_err = means[0] - COL_CENTER
                        _prev_roll_err = roll_err
                    else:
                        roll_err = _prev_roll_err

                    normalized_roll_err = roll_err / COL_CENTER
                    roll = -roll_controller.calculate_position(normalized_roll_err, dt)

                    # UPDATE THREAD STATE INSTEAD OF SENDING
                    set_flight_command("LINE_FOLLOW", 0, roll, 0, 0)
                    continue

            _direction, _mean = lu.fit_lines(image)
            direction = direction_filter(_direction)
            mean = mean_filter(_mean)
            angle = np.arctan(direction[0] / direction[1])
            angle = np.degrees(angle)
            roll_err = mean[0] - COL_CENTER
            target_angle = angle
            ADVANCE_PITCH = abs((90 - target_angle) / 90 * 0.4)
            if abs(roll_err) < 160 and target_angle < 45:
                roll_controller.kp = 0.6
                roll_controller.max_output = 0.8
                mode = "Straight"
            else:
                roll_controller.kp = 0.6
                mode = "Roll Correct"
                roll_controller.max_output = 0.8
            if time.time() - _through_time < 1:
                ADVANCE_PITCH = 0.0
            else:
                ADVANCE_PITCH = abs((90 - target_angle) / 90 * 0.5)
            full_controller.set_setpoint(_alt=_target_height)
            normalized_roll_err = roll_err / COL_CENTER
            roll = -roll_controller.calculate_position(normalized_roll_err, dt)
            output = full_controller.calculate(_alt=drone.physics.get_altitude(),
                                               _alt_vel=drone.physics.get_linear_velocity()[1],
                                               _yaw=target_angle, dt=dt)
            roll += DRIFT_FF * output[2]
            print(_through_time, _through_dist, _target_height)
            if 1.3 > _through_dist > -0.0:
                roll = 0.0
                output[2] = 0.0
            set_flight_command("LINE_FOLLOW", ADVANCE_PITCH, roll, output[2], output[3])
            _prev_roll_err = roll_err
        math_duration = time.time() - loop_start_time
        time_to_sleep = loop_delay - math_duration
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)


def gate_detect_loop(drone):
    global _timer, _done, mode, _prev_closest_gate, _target_height, _dist_to_gate_int, _closest_dist, _height_measurement
    global _line_follow_running, _latest_cmd, _target_height, _through_dist, _through_time
    target_fps = 20.0
    loop_delay = 1.0 / target_fps
    last_time = time.time()

    while _gate_detect_running:
        loop_start_time = time.time()

        _image = drone.camera.get_color_image_async()

        dt = loop_start_time - last_time
        last_time = loop_start_time

        closest_gate = None

        if _image is not None:
            image = cv2.resize(_image, (640, 480), interpolation=cv2.INTER_LINEAR)
            gate_measurements = gd.detect_gates(image, _timer, drone.physics.get_altitude(),
                                                drone.physics.get_linear_velocity()[2])
            closest_val = float("inf")
            if gate_measurements is not None:
                for (gate_id, gate_measurement) in gate_measurements.items():
                    if gate_measurement is not None:
                        _gates[gate_id].update(gate_measurement)
                    else:
                        _gates[gate_id].predict()

                    if _gates[gate_id].distance_filter.x[0, 0] < closest_val:
                        closest_val = _gates[gate_id].distance_filter.x[0, 0]
                        closest_gate = _gates[gate_id]
            if closest_val <= CLOSEST_GATE_THRESHOLD:
                _target_height = closest_gate.altitude_filter.x[0, 0]
                _closest_dist = closest_val
                _height_measurement = True
            if closest_val > CLOSEST_GATE_THRESHOLD:
                if _height_measurement:
                    _through_dist = _closest_dist
                    _through_time = time.time()
                elif _through_dist < -0.5 and _through_time != -1:
                    _target_height = TARGET_LF_HEIGHT
                    _through_time = -1
                elif _through_time != -0.25:
                    _through_dist -= abs(drone.physics.get_linear_velocity()[2] * dt)
                _height_measurement = False
        math_duration = time.time() - loop_start_time
        time_to_sleep = loop_delay - math_duration
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)


def update_gate_distances(drone):
    for (gate_id, gate) in _gates.items():
        if gate is not None:
            gate.update_forward_velocity(drone.physics.get_linear_velocity()[2])


def update(drone):
    global _timer, _done, _line_follow_thread, _line_follow_running, _latest_cmd
    global _gate_detect_running, _gate_detect_thread
    global _active_flight_mode

    update_gate_distances(drone)

    if _done:
        _line_follow_running = False
        _gate_detect_running = False
        return True
    if _line_follow_thread is None or _gate_detect_thread is None:
        # 1. Start Line Follower (Active)
        _line_follow_running = True
        _line_follow_thread = tu.PausableThread(target=line_control_loop, args=(drone,), daemon=True)
        _line_follow_thread.start()

        # 2. Start Gate Detector (Active)
        _gate_detect_running = True
        _gate_detect_thread = tu.PausableThread(target=gate_detect_loop, args=(drone,), daemon=True)
        _gate_detect_thread.start()

    _timer += drone.get_delta_time()

    current_cmd = _latest_cmd

    # Fast-publish the most recent command calculated by the vision thread
    drone.flight.send_pcmd(
        current_cmd["pitch"],
        current_cmd["roll"],
        current_cmd["yaw"],
        current_cmd["throttle"]
    )

    return _done


def set_flight_command(caller_mode, pitch, roll, yaw, throttle):
    global _latest_cmd

    with _command_lock:
        # Only accept commands from the thread that is officially driving
        if caller_mode == _active_flight_mode:
            _latest_cmd["pitch"] = pitch
            _latest_cmd["roll"] = roll
            _latest_cmd["yaw"] = yaw
            _latest_cmd["throttle"] = throttle
