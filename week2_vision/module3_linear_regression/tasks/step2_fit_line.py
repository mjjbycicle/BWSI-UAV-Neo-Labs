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
S_MIN = 200
MIN_PIXELS = 200
ADVANCE_PITCH = 0.1  # fly forward off the spawn pad to reach the line
ADVANCE_TIME = 8.0  # seconds of forward flight before fitting
K_CURVE = 0.1
COL_CENTER = 320

# -- Module-level state -----------------------------------------------------
_timer = 0.0
_done = False

full_controller = PDControl.FullController(kp_yaw=0.08, kp_alt=1, max_yaw=0.7, max_throttle=0.8)
roll_controller = PDControl.PDController(1.0, 0.0, 0.4)
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
_closest_gate = None
_target_height = 0.5
_gate_fly_through_thread = None
_gate_fly_through_running = False
_gate_fly_through = fu.BooleanDebouncer(delay_seconds=0.1)
_gate_fly_through_timer = 0.0
_gates = dict()
for i in range(gd.NUM_GATES):
    _gates[i] = gd.Gate(0.0)


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
    target_fps = 20.0
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
                    roll = -roll_controller.calculate_position(normalized_roll_err, dt)

                    # UPDATE THREAD STATE INSTEAD OF SENDING
                    set_flight_command("LINE_FOLLOW", 0, roll, 0, 0)
                    continue

            directions, means = lu.fit_lines(points, prev_bottom_mean=_prev_bottom_mean)
            # directions = direction_filter(_directions)
            # means = mean_filter(_means)
            _prev_bottom_mean = means[3]

            angles = np.arctan2(directions[:, 0], directions[:, 1])
            angles = np.degrees(angles)
            curvature = 0 #angles[0] - angles[3]
            roll_err = means[3][0] - COL_CENTER
            target_angle = angles[0]

            if abs(curvature) < 30 and abs(roll_err) < 160:
                curvature = 0
                ADVANCE_PITCH = 0.1
                roll_controller.kp = 1.5
                roll_controller.max_output = 1
                mode = "Straight"
            else:
                ADVANCE_PITCH = 0.1
                roll_controller.kp = 1.5
                if abs(roll_err) > 160:
                    ADVANCE_PITCH = 0.1
                    curvature = 0.0
                    roll_controller.kp = 1
                    mode = "Roll Correct"
                else:
                    mode = "Curve"
                roll_controller.max_output = 1

            curvature_ff = -curvature * 0.008
            full_controller.set_setpoint(_alt=1.5)
            normalized_roll_err = roll_err / COL_CENTER
            print(f"roll err: {normalized_roll_err}")
            # target_angle -= normalized_roll_err * 30
            roll = -roll_controller.calculate_position(normalized_roll_err, dt) + curvature_ff
            roll = drone_utils.clamp(roll, -1, 1)
            output = full_controller.calculate(_alt=drone.physics.get_altitude(),
                                               _alt_vel=drone.physics.get_linear_velocity()[1],
                                               _yaw=target_angle, dt=dt)
            # print(f"roll: {roll}, yaw: {output[2]}")
            # UPDATE THREAD STATE INSTEAD OF SENDING
            set_flight_command("LINE_FOLLOW", ADVANCE_PITCH, roll, output[2], output[3])
            _prev_roll_err = roll_err

        # Small sleep to prevent this loop from maxing out a CPU core
        # --- THE RATE LIMITER ---
        # Calculate how long the math actually took
        math_duration = time.time() - loop_start_time

        # Sleep for whatever time is remaining to hit our exact 20Hz target
        time_to_sleep = loop_delay - math_duration
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)


def gate_fly_through_loop(drone):
    global _gate_fly_through, _gate_fly_through_timer, _line_follow_running, _latest_cmd

    # Target 20 frames per second (0.05 seconds per loop)
    target_fps = 60.0
    loop_delay = 1.0 / target_fps

    last_time = time.time()

    while _gate_fly_through_running:
        loop_start_time = time.time()

        # Calculate isolated delta time for the vision controllers
        dt = loop_start_time - last_time
        last_time = loop_start_time

        # UPDATE THREAD STATE INSTEAD OF SENDING
        set_flight_command("GATE_FLY_THROUGH", 0.5, 0.0, 0.0, 0.0)

        # Small sleep to prevent this loop from maxing out a CPU core
        # --- THE RATE LIMITER ---
        # Calculate how long the math actually took
        math_duration = time.time() - loop_start_time
        time_to_sleep = loop_delay - math_duration
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)


def gate_detect_loop(drone):
    global _timer, _done, mode, _closest_gate, _target_height
    global _line_follow_running, _latest_cmd

    # Target 20 frames per second (0.05 seconds per loop)
    target_fps = 20.0
    loop_delay = 1.0 / target_fps

    last_time = time.time()

    while _gate_detect_running:
        loop_start_time = time.time()

        # Calculate isolated delta time for the vision controllers
        dt = loop_start_time - last_time
        last_time = loop_start_time

        _image = drone.camera.get_color_image_async()

        if _image is not None:
            image = cv2.resize(_image, (640, 480), interpolation=cv2.INTER_LINEAR)
            gate_measurements = gd.detect_gates(image, _timer, drone.physics.get_altitude(),
                                                drone.physics.get_linear_velocity()[2])
            if gate_measurements is not None:
                for (gate_id, gate_measurement) in gate_measurements.items():
                    if gate_measurement is not None:
                        _gates[gate_id].update(gate_measurement)
                    else:
                        _gates[gate_id].predict()

        # Small sleep to prevent this loop from maxing out a CPU core
        # --- THE RATE LIMITER ---
        # Calculate how long the math actually took
        math_duration = time.time() - loop_start_time

        # Sleep for whatever time is remaining to hit our exact 20Hz target
        time_to_sleep = loop_delay - math_duration
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)


def update_gate_distances(drone):
    for (gate_id, gate) in _gates.items():
        if gate is not None:
            gate.update_forward_velocity(drone.physics.get_linear_velocity()[2])


def update(drone):
    global _timer, _done, _line_follow_thread, _line_follow_running, _latest_cmd
    global _gate_detect_running, _gate_detect_thread, _gate_fly_through_thread, _gate_fly_through, _gate_fly_through_running
    global _active_flight_mode

    update_gate_distances(drone)

    if _done:
        _line_follow_running = False  # Tell the thread loop to shut down safely
        _gate_detect_running = False
        _gate_fly_through_running = False
        return True

    # Initialize the background thread on the first tick
    if _line_follow_thread is None or _gate_detect_thread is None or _gate_fly_through_thread is None:
        # 1. Start Line Follower (Active)
        _line_follow_running = True
        _line_follow_thread = tu.PausableThread(target=line_control_loop, args=(drone,), daemon=True)
        _line_follow_thread.start()

        # 2. Start Gate Detector (Active)
        _gate_detect_running = True
        _gate_detect_thread = tu.PausableThread(target=gate_detect_loop, args=(drone,), daemon=True)
        _gate_detect_thread.start()

        # 3. Start Gate Fly-Through (Paused!)
        _gate_fly_through_running = True
        _gate_fly_through_thread = tu.PausableThread(target=gate_fly_through_loop, args=(drone,), daemon=True,
                                                     paused=True)
        _gate_fly_through_thread.start()

    if _gate_fly_through.value:
        # Instantly switch authority
        with _command_lock:
            _active_flight_mode = "GATE_FLY_THROUGH"
        _line_follow_thread.pause()
        _gate_fly_through_thread.resume()

    else:
        # Instantly switch authority back
        with _command_lock:
            _active_flight_mode = "LINE_FOLLOW"
        _line_follow_thread.resume()
        _gate_fly_through_thread.pause()

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
