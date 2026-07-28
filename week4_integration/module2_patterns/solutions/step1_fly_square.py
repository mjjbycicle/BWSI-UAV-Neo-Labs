"""
MIT BWSI Autonomous Drone Racing Course - UAV Neo
GNU General Public License v3.0

Week 4 · Module 2 — Step 1: Fly a Square  (SOLUTION)
"""

import drone_core
import drone_utils as uav_utils

# -- Course setup: makes the shared `neo_lab` helper importable.
#    You don't need to read or change this block. --
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.realpath(__file__))
while _os.path.basename(_d) != "labs" and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
if _d not in _sys.path:
    _sys.path.insert(0, _d)
import neo_lab

# -- Constants --------------------------------------------------------------
SIDE = 3.0
WAYPOINTS = [(0.0, SIDE), (SIDE, SIDE), (SIDE, 0.0), (0.0, 0.0)]
TARGET_HEIGHT = 3.0
KP_POS = 0.18
KD_POS = 0.5
ALT_KP = 0.12
ROLL_LIMIT = 0.25
PITCH_LIMIT = 0.25
THROTTLE_LIMIT = 0.5
WP_TOL = 0.6

# -- Module-level state -----------------------------------------------------
_x = 0.0
_z = 0.0
_wp = 0
_done = False

def reset():
    global _x, _z, _wp, _done
    _x = 0.0
    _z = 0.0
    _wp = 0
    _done = False


def update(drone):
    global _x, _z, _wp, _done
    if _done:
        return True
    dt = drone.get_delta_time()
    vx, vy, vz = drone.physics.get_linear_velocity()
    _x += vx * dt
    _z += vz * dt
    if _wp >= len(WAYPOINTS):
        drone.flight.stop()
        print("[Step 1] Square complete")
        _done = True
        return True
    target_right, target_fwd = WAYPOINTS[_wp]
    roll = uav_utils.clamp(KP_POS * (target_right - _x) - KD_POS * vx,
                           -ROLL_LIMIT, ROLL_LIMIT)
    pitch = uav_utils.clamp(KP_POS * (target_fwd - _z) - KD_POS * vz,
                            -PITCH_LIMIT, PITCH_LIMIT)
    throttle = uav_utils.clamp(ALT_KP * (TARGET_HEIGHT - neo_lab.height(drone)),
                               -THROTTLE_LIMIT, THROTTLE_LIMIT)
    drone.flight.send_pcmd(pitch, roll, 0, throttle)
    if abs(target_right - _x) < WP_TOL and abs(target_fwd - _z) < WP_TOL:
        print(f"[Step 1] reached corner {_wp}: ({target_right:.1f}, {target_fwd:.1f})")
        _wp += 1
    return False


if __name__ == "__main__":
    _drone = drone_core.create_drone()
    _launcher = neo_lab.Launcher(3.0)

    def start():
        _launcher.reset()
        reset()
        print("Step 1: Fly a Square")

    def _update():
        if not _launcher.done:
            _launcher.update(_drone)
            return
        if update(_drone):
            _drone.flight.land()

    _drone.set_start_update(start, _update)
    _drone.go()
