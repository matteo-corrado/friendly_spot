"""Geometry helpers for aiming PTZ.

Two modes:
- Bearing-only: map pixel X to yaw via HFOV and combine with camera yaw
    (prefer yaw from transforms; fallback to config).
- Transform-based: unproject pixel to ray (pinhole), transform ray to BODY/PTZ
    frame using transforms_snapshot, compute pan/tilt.

References:
- frame_helpers.get_a_tform_b
- image_response.shot.frame_name_image_sensor
- image_response.source.pinhole (if available)
"""
import math
from typing import Dict, Tuple

import numpy as np
from bosdyn.client import frame_helpers


def bbox_center(bbox_xywh):
    """Return (cx, cy) center of a bbox given as (x,y,w,h)."""
    x, y, w, h = bbox_xywh
    return (x + w / 2.0, y + h / 2.0)


def pixel_to_yaw_offset(cx: float, img_w: int, hfov_deg: float) -> float:
    """Map pixel X to relative yaw using a simple symmetric HFOV model.

    Inputs: cx (float), img_w (int), hfov_deg (float)
    Output: yaw offset in degrees
    """
    norm_x = (cx - (img_w / 2.0)) / (img_w / 2.0)  # -1..+1
    return norm_x * (hfov_deg / 2.0)


def camera_yaw_from_transforms(snapshot, frame_name_image_sensor: str) -> float:
    """Compute camera yaw (deg) relative to BODY using transforms.

    Inputs:
    - snapshot: transforms_snapshot from an ImageResponse
    - frame_name_image_sensor: name of the image sensor frame

    Output: yaw angle in degrees in [-180, 180] where 0 aligns with BODY +X.
    """
    vision_T_cam = frame_helpers.get_a_tform_b(snapshot, frame_helpers.VISION_FRAME_NAME,
                                               frame_name_image_sensor)
    vision_T_body = frame_helpers.get_a_tform_b(snapshot, frame_helpers.VISION_FRAME_NAME,
                                                frame_helpers.BODY_FRAME_NAME)
    body_T_cam = vision_T_body.inverse() * vision_T_cam
    # Extract yaw from body_T_cam rotation: yaw = atan2(R[1,0], R[0,0]) for x-forward,y-left
    R = body_T_cam.rotation.to_matrix()
    yaw_rad = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(yaw_rad)


def bearing_from_detection_bearing_only(source_name: str, bbox_xywh, img_w: int,
                                        hfov_deg: float, cam_yaw_deg: float) -> float:
    """Compute robot-frame bearing (deg) by combining camera yaw and pixel-derived offset.

    Inputs:
    - source_name: image source (unused; for logging)
    - bbox_xywh: detection bbox
    - img_w: image width
    - hfov_deg: horizontal FOV estimate for the camera
    - cam_yaw_deg: camera yaw relative to BODY (from transforms)

    Output: approximate robot-frame bearing in degrees.
    """
    cx, _ = bbox_center(bbox_xywh)
    yaw_off = pixel_to_yaw_offset(cx, img_w, hfov_deg)
    return cam_yaw_deg + yaw_off


def ray_to_ptz_angles(d_ptz: np.ndarray) -> Tuple[float, float]:
    """Given a unit direction vector in PTZ frame (x forward, y left, z up), return (pan, tilt) radians."""
    pan = math.atan2(d_ptz[1], d_ptz[0])
    hyp = math.hypot(d_ptz[0], d_ptz[1])
    tilt = math.atan2(-d_ptz[2], hyp)
    return pan, tilt


def pixel_to_ray_pinhole(resp, u: float, v: float) -> np.ndarray:
    """Compute a unit direction ray in the camera frame from pixel (u,v) using pinhole intrinsics.

    Inputs:
    - resp: ImageResponse with source.pinhole.intrinsics populated
    - u, v: pixel coordinates (float)

    Output: unit 3D direction (np.ndarray shape (3,)) in the camera frame.
    Raises ValueError if pinhole intrinsics are missing.
    """
    if not resp.source.pinhole:
        raise ValueError("Pinhole intrinsics not available for this source.")
    intr = resp.source.pinhole.intrinsics
    fx = intr.focal_length.x
    fy = intr.focal_length.y
    cx = intr.principal_point.x
    cy = intr.principal_point.y

    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0
    d = np.array([x, y, z], dtype=float)
    n = np.linalg.norm(d)
    if n == 0:
        return d
    return d / n


def transform_direction(snapshot, from_frame: str, to_frame: str, d_from: np.ndarray) -> np.ndarray:
    """Rotate a direction vector from one frame to another using a transforms snapshot.

    Inputs:
    - snapshot: transforms snapshot
    - from_frame: source frame name (e.g., image sensor)
    - to_frame: destination frame name (e.g., BODY)
    - d_from: unit direction vector in from_frame

    Output: unit direction vector in to_frame
    """
    a_T_from = frame_helpers.get_a_tform_b(snapshot, frame_helpers.VISION_FRAME_NAME, from_frame)
    a_T_to = frame_helpers.get_a_tform_b(snapshot, frame_helpers.VISION_FRAME_NAME, to_frame)
    to_T_from = a_T_to.inverse() * a_T_from
    R = to_T_from.rotation.to_matrix()
    d_to = R.dot(d_from)
    n = np.linalg.norm(d_to)
    if n == 0:
        return d_to
    return d_to / n
