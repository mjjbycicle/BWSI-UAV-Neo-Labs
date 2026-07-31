"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 2/3 Lab — Step 1: Detect the Line Pixels
Find the colored line pixels in the downward camera.
"""

import drone_core
import drone_utils as uav_utils
import cv2
import numpy as np

# -- Course setup: makes the shared `neo_lab` helper importable.
#    You don't need to read or change this block. --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab
from . import PDControl
from . import line_util as lu

# -- Constants --------------------------------------------------------------
S_MIN         = 100
TARGET_HEIGHT = 1.0
ADVANCE_TIME  = 60       # seconds of forward flight before reporting

# -- Module-level state -----------------------------------------------------
_timer = 0.0
_done  = False
_hold = 0.0

full_controller = PDControl.FullController(kp_yaw=0.01, kp_alt=1, max_yaw=0.7, max_throttle=0.8)

def reset():
    global _timer, _done
    _timer = 0.0
    _done  = False


def update(drone):
    global _timer, _done, _hold
    if _done:
        return True
    ##################################
    #### START PUT CODE HERE #########

    # The drone spawns on a colored landing pad, so first fly forward (ADVANCE_PITCH) for
    # ADVANCE_TIME to reach the line, then hover. The line is vivid against a grey floor, so
    # threshold by saturation: neo_lab.saturated_mask(image, S_MIN) gives a mask of the line
    # pixels. Count them, print the count, and set _done. See the README (Key terms).

    # full_controller.set_setpoint(_alt=TARGET_HEIGHT)
    # drone.flight.send_pcmd(0, 0, 0, output)
    # print(f"alt: {drone.physics.get_altitude()}, output: {output}")
    # image = drone.camera.get_downward_image_async()
    # direction, mean = lu.fit_lines(image)
    # angle = np.degrees(np.arctan(direction[0] / direction[1]))
    # print(angle)
    # output = full_controller.calculate(_yaw = angle, dt = drone.get_delta_time())[2]
    drone.flight.send_pcmd_with_alt_hold(0, 0, 0, TARGET_HEIGHT)

    _timer += drone.get_delta_time()

    if abs(drone.physics.get_altitude() - TARGET_HEIGHT) < 0.05:
        _hold += drone.get_delta_time()
        if _hold > 3:
            _done = True
            return True
    else:
        _hold = 0.0

    ###### END PUT CODE HERE #########
    ##################################
    return _done


if __name__ == "__main__":
    _drone = drone_core.create_drone()

    def start():
        reset()
        print("Step 1: Detect the Line Pixels")

    def _update():
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go()
