import cameratransform as ct
import numpy as np


def get_correct_center(roll, pitch, yaw, alt, focal_px, center_x, center_y):
    cam = ct.Camera(
        ct.RectilinearProjection(
            ct.CameraProjection(
                focallength_px=focal_px,  # Focal length in pixels
                center_x_px=center_x,  # Image center X (e.g., 1920/2)
                center_y_px=center_y  # Image center Y (e.g., 1080/2)
            )
        ),
        # 2. Define extrinsic (height and tilt angles)
        orientation=ct.SpatialOrientation(
            elevation_m=alt,  # Altitude (height) in meters
            tilt_deg=pitch,  # Camera pitch angle in degrees
            roll_deg=roll,  # Camera roll angle in degrees
            heading_deg=yaw  # Yaw angle
        )
    )

    # 3. Project the point directly below the drone
    # World origin [0, 0, 0] is the ground point directly below the camera
    pixel_x, pixel_y = cam.spaceFromImage([0, 0, 0])
    return pixel_x, pixel_y
