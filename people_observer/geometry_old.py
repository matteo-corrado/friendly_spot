"""Geometry helpers for aiming PTZ.

Two modes:
- Bearing-only: map pixel to bearing using camera intrinsics (Kannala-Brandt or pinhole)
    and frame transforms. Fallback to simple geometric projection if intrinsics unavailable.
- Transform-based: unproject pixel to ray using SDK intrinsics, transform to BODY/PTZ
    frame using transforms_snapshot, compute pan/tilt.

References:
- cameras.pixel_to_camera_ray: uses OpenCV fisheye undistortion for Kannala-Brandt
- frame_helpers.get_a_tform_b: transform rays between coordinate frames
- math_helpers: SDK angle normalization, quaternion operations, SE3Pose transforms
- ImageSource.kannala_brandt / ImageSource.pinhole: intrinsics from SDK
"""
import logging
from typing import Optional, Tuple

import numpy as np
from bosdyn.client import frame_helpers, math_helpers
from bosdyn.api import image_pb2, geometry_pb2

from . import cameras
from .config import CAM_YAW_DEG

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def pixel_to_bearing_simple(
    x: float, 
    y: float, 
    img_width: int, 
    img_height: int, 
    hfov_deg: float, 
    cam_yaw_deg: float
) -> float:
    """Convert pixel to robot-frame bearing using simple geometric projection.
    
    This is a fallback approximation when intrinsics are unavailable.
    For cameras with intrinsics, use pixel_to_bearing_with_intrinsics() instead.
    
    Args:
        x, y: Pixel coordinates
        img_width, img_height: Image dimensions
        hfov_deg: Horizontal FOV in degrees
        cam_yaw_deg: Camera yaw offset relative to robot body frame
        
    Returns:
        Bearing in degrees relative to robot body frame front (0 deg = forward)
    """
    # Normalize to [-0.5, 0.5] centered
    norm_x = (x / img_width) - 0.5
    
    # Map to camera-frame angle
    hfov_rad = np.deg2rad(hfov_deg)
    cam_angle_rad = norm_x * hfov_rad
    cam_angle_deg = np.rad2deg(cam_angle_rad)
    
    # Add camera yaw offset to get robot-frame bearing
    robot_bearing_deg = cam_yaw_deg + cam_angle_deg
    
    # Log the calculation (simple projection mode)
    logger.info(f"Simple projection: pixel=({x:.0f},{y:.0f}) cam_angle={cam_angle_deg:.2f}° cam_yaw={cam_yaw_deg:.2f}° -> bearing={robot_bearing_deg:.2f}°")
    
    # Normalize to [-180, 180] using SDK angle normalization
    robot_bearing_deg = np.rad2deg(math_helpers.recenter_angle_mod(np.deg2rad(robot_bearing_deg), 0.0))
    
    return robot_bearing_deg


def pixel_to_bearing_with_intrinsics(
    x: float,
    y: float,
    intrinsics: dict,
    cam_yaw_deg: float,
    robot,
    image_response
) -> float:
    """Convert pixel to robot-frame bearing using SDK intrinsics and transforms.
    
    Uses OpenCV fisheye undistortion for Kannala-Brandt cameras, or pinhole
    projection for rectilinear cameras. Then transforms ray to robot body frame.
    Falls back to simple projection if transforms fail.
    
    Args:
        x, y: Pixel coordinates
        intrinsics: Dict from cameras.get_camera_intrinsics()
        cam_yaw_deg: Camera yaw offset (used as fallback if transforms fail)
        robot: Authenticated robot instance
        image_response: ImageResponse containing frame tree snapshot
        
    Returns:
        Bearing in degrees relative to robot body frame front (0 deg = forward)
    """
    if intrinsics is None or intrinsics['model'] is None:
        logger.warning("No intrinsics available, falling back to simple projection")
        hfov = cameras.calculate_hfov_from_intrinsics(intrinsics) if intrinsics else 133.0
        return pixel_to_bearing_simple(
            x, y, 
            intrinsics['width'] if intrinsics else 640,
            intrinsics['height'] if intrinsics else 480,
            hfov, cam_yaw_deg
        )
    
    # Convert pixel to 3D ray in camera frame using appropriate model
    try:
        x_cam, y_cam, z_cam = cameras.pixel_to_camera_ray(x, y, intrinsics)
    except Exception as e:
        logger.error(f"Failed to undistort pixel: {e}")
        hfov = cameras.calculate_hfov_from_intrinsics(intrinsics)
        return pixel_to_bearing_simple(
            x, y, intrinsics['width'], intrinsics['height'], hfov, cam_yaw_deg
        )
    
    # Transform ray from camera frame to robot body frame
    try:
        shot = image_response.shot
        frame_tree = shot.transforms_snapshot
        camera_frame = shot.frame_name_image_sensor
        body_frame = frame_helpers.BODY_FRAME_NAME
        
        # Get transform: body_tform_camera (returns math_helpers.SE3Pose)
        body_tform_camera = frame_helpers.get_a_tform_b(
            frame_tree,
            body_frame,
            camera_frame
        )
        
        # Transform ray using SDK Vec3 and SE3Pose methods
        ray_camera_vec = math_helpers.Vec3(x_cam, y_cam, z_cam)
        ray_body_vec = body_tform_camera.rot.transform_vec3(ray_camera_vec.to_proto())
        
        # Log transform from robot body frame to human (ray direction)
        logger.info(f"Robot->Human transform: ray_body=[{ray_body_vec.x:.3f}, {ray_body_vec.y:.3f}, {ray_body_vec.z:.3f}]")
        
        # Convert to bearing angle (atan2 in horizontal plane)
        bearing_rad = np.arctan2(ray_body_vec.y, ray_body_vec.x)
        
        # Normalize to [-180, 180] using SDK angle normalization
        bearing_rad = math_helpers.recenter_angle_mod(bearing_rad, 0.0)
        bearing_deg = np.rad2deg(bearing_rad)
        
        logger.debug(f"Pixel ({x:.0f},{y:.0f}) -> bearing {bearing_deg:.1f} deg (via transforms)")
        return bearing_deg
        
    except Exception as e:
        logger.warning(f"Transform failed, using yaw offset fallback: {e}")
        
        # Fallback: compute angle in camera frame, add yaw offset
        cam_angle_rad = np.arctan2(x_cam, z_cam)  # Horizontal angle in camera frame
        cam_angle_deg = np.rad2deg(cam_angle_rad)
        robot_bearing_deg_rad = np.deg2rad(cam_yaw_deg + cam_angle_deg)
        robot_bearing_deg_rad = math_helpers.recenter_angle_mod(robot_bearing_deg_rad, 0.0)
        robot_bearing_deg = np.rad2deg(robot_bearing_deg_rad)
        
        return robot_bearing_deg


def get_camera_yaw_fallback(source_name: str) -> float:
    """Return hardcoded camera yaw offset from config (fallback if transforms unavailable)."""
    return CAM_YAW_DEG.get(source_name, 0.0)


# Legacy functions kept for compatibility
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
    vision_tform_cam = frame_helpers.get_a_tform_b(
        snapshot, 
        frame_helpers.VISION_FRAME_NAME,
        frame_name_image_sensor
    )
    vision_tform_body = frame_helpers.get_a_tform_b(
        snapshot, 
        frame_helpers.VISION_FRAME_NAME,
        frame_helpers.BODY_FRAME_NAME
    )
    body_tform_cam = vision_tform_body.inverse() * vision_tform_cam
    
    # Extract yaw from quaternion using SDK helper
    yaw_rad, pitch_rad, roll_rad = math_helpers.quat_to_eulerZYX(body_tform_cam.rot)
    yaw_rad = math_helpers.recenter_angle_mod(yaw_rad, 0.0)
    return np.rad2deg(yaw_rad)


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
    pan = np.arctan2(d_ptz[1], d_ptz[0])
    hyp = np.hypot(d_ptz[0], d_ptz[1])
    tilt = np.arctan2(-d_ptz[2], hyp)
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
    - d_from: unit direction vector in from_frame (numpy array)

    Output: unit direction vector in to_frame (numpy array)
    """
    vision_tform_from = frame_helpers.get_a_tform_b(
        snapshot, 
        frame_helpers.VISION_FRAME_NAME, 
        from_frame
    )
    vision_tform_to = frame_helpers.get_a_tform_b(
        snapshot, 
        frame_helpers.VISION_FRAME_NAME, 
        to_frame
    )
    to_tform_from = vision_tform_to.inverse() * vision_tform_from
    
    # Use SDK Vec3 and Quat transform methods
    d_from_vec = math_helpers.Vec3(d_from[0], d_from[1], d_from[2])
    d_to_vec = to_tform_from.rot * d_from_vec  # Quat * Vec3 applies rotation
    
    # Convert back to numpy and normalize
    d_to = d_to_vec.to_numpy()
    norm = np.linalg.norm(d_to)
    if norm > 0:
        d_to = d_to / norm
    return d_to
