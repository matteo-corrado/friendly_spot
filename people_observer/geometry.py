"""Geometry transformations for pixel-to-PTZ control.

This module provides two approaches for computing PTZ angles from pixel detections:

1. **Transform Mode (Recommended)**: 
   - Unprojects pixel to 3D ray using camera intrinsics (Kannala-Brandt fisheye or pinhole)
   - Transforms ray through SDK frame tree (camera → body → PTZ)
   - Accurate distortion correction and dynamic frame transforms
   
2. **Bearing Mode (Fallback)**:
   - Simple geometric projection using HFOV and camera yaw offset
   - Fast but approximate, ignores lens distortion
   - Used when intrinsics unavailable or explicitly requested

Key Functions:
    pixel_to_ptz_angles_transform: Full 3D transform pipeline (primary method)
    pixel_to_ptz_angles_simple: Fallback bearing-only approximation
    
Frame Conventions:
    - Body frame: origin at hip center, +X forward, +Y left, +Z up
    - PTZ pan: [0, 360°] where 0=forward, +90=right (clockwise from above)
    - PTZ tilt: [-30, 100°] where negative=down, positive=up
    
References:
    - cameras.pixel_to_camera_ray: Undistorts pixel using cv2.fisheye or pinhole model
    - frame_helpers.get_a_tform_b: SDK transform lookup
    - math_helpers: SDK Vec3, Quat, SE3Pose operations
"""
import logging
from typing import Optional, Tuple

import numpy as np
from bosdyn.client import frame_helpers, math_helpers
from bosdyn.api import image_pb2

from . import cameras
from .config import CAM_YAW_DEG

logger = logging.getLogger(__name__)


def pixel_to_ptz_angles_transform(
    pixel_x: float,
    pixel_y: float,
    intrinsics: dict,
    image_response,
    robot
) -> Tuple[float, float]:
    """Convert pixel coordinates to PTZ pan/tilt angles using full transform pipeline.
    
    This is the primary method for accurate PTZ control. It:
    1. Unprojects pixel to 3D ray in camera frame using intrinsics
    2. Transforms ray from camera frame → robot body frame
    3. Computes pan (horizontal bearing) and tilt (vertical angle)
    4. Converts from body frame convention to PTZ convention
    
    Args:
        pixel_x, pixel_y: Detection center in pixel coordinates
        intrinsics: Camera intrinsics dict from cameras.get_camera_intrinsics()
        image_response: ImageResponse containing frame tree snapshot
        robot: Robot instance (for logging/debugging)
        
    Returns:
        (pan_deg, tilt_deg) tuple in PTZ coordinates:
            pan: [0, 360] degrees, 0=forward, clockwise from above
            tilt: [-30, 100] degrees, negative=down, positive=up
            
    Raises:
        ValueError: If intrinsics are None or invalid
        RuntimeError: If frame transforms fail
    """
    if intrinsics is None or intrinsics.get('model') is None:
        raise ValueError(f"Valid camera intrinsics required for transform mode")
    
    # Log intrinsics info
    model = intrinsics['model']
    logger.info(f"Using {model} camera model for pixel ({pixel_x:.0f}, {pixel_y:.0f})")
    
    # Step 1: Unproject pixel to 3D ray in camera frame
    try:
        ray_cam_x, ray_cam_y, ray_cam_z = cameras.pixel_to_camera_ray(
            pixel_x, pixel_y, intrinsics
        )
        logger.debug(f"Unprojected ray in camera frame: [{ray_cam_x:.3f}, {ray_cam_y:.3f}, {ray_cam_z:.3f}]")
    except Exception as e:
        raise RuntimeError(f"Failed to unproject pixel using {model} model: {e}")
    
    # Step 2: Transform ray from camera frame to body frame
    try:
        shot = image_response.shot
        frame_tree = shot.transforms_snapshot
        camera_frame = shot.frame_name_image_sensor
        body_frame = frame_helpers.BODY_FRAME_NAME
        
        # Get transform: body_tform_camera
        body_tform_camera = frame_helpers.get_a_tform_b(
            frame_tree,
            body_frame,
            camera_frame
        )
        
        # Apply rotation transform (position doesn't affect direction)
        ray_cam_vec = math_helpers.Vec3(ray_cam_x, ray_cam_y, ray_cam_z)
        ray_body_vec = body_tform_camera.rot.transform_vec3(ray_cam_vec.to_proto())
        
        logger.info(f"Robot -> Human ray (body frame): [{ray_body_vec.x:.3f}, {ray_body_vec.y:.3f}, {ray_body_vec.z:.3f}]")
        
    except Exception as e:
        raise RuntimeError(f"Frame transform failed: {e}")
    
    # Step 3: Compute bearing (pan) and elevation (tilt) from body-frame ray
    # Body frame: +X=forward, +Y=left, +Z=up
    # Pan: horizontal angle from forward direction
    bearing_rad = np.arctan2(ray_body_vec.y, ray_body_vec.x)
    bearing_rad = math_helpers.recenter_angle_mod(bearing_rad, 0.0)  # [-π, π]
    bearing_deg = np.rad2deg(bearing_rad)  # [-180, 180]
    
    # Tilt: vertical angle from horizontal plane
    horizontal_dist = np.sqrt(ray_body_vec.x**2 + ray_body_vec.y**2)
    tilt_rad = np.arctan2(ray_body_vec.z, horizontal_dist)
    tilt_deg = np.rad2deg(tilt_rad)
    
    logger.info(f"Body frame angles: bearing={bearing_deg:.2f}°, tilt={tilt_deg:.2f}°")
    
    # Step 4: Convert from body frame to PTZ coordinates
    # Body bearing: 0=forward, +90=left, -90=right (math convention)
    # PTZ pan: 0=forward, +90=right, +180=back, +270=left (compass convention)
    # Conversion: PTZ_pan = -bearing (negate to flip left/right direction)
    ptz_pan_deg = -bearing_deg
    
    # Normalize to [0, 360]
    if ptz_pan_deg < 0:
        ptz_pan_deg += 360
    
    # PTZ tilt: negative=down, positive=up (same convention as body frame Z)
    ptz_tilt_deg = tilt_deg
    
    logger.info(f"PTZ angles: pan={ptz_pan_deg:.2f}°, tilt={ptz_tilt_deg:.2f}°")
    
    return ptz_pan_deg, ptz_tilt_deg


def pixel_to_ptz_angles_simple(
    pixel_x: float,
    pixel_y: float,
    img_width: int,
    img_height: int,
    hfov_deg: float,
    camera_yaw_deg: float,
    default_tilt_deg: float = -5.0
) -> Tuple[float, float]:
    """Convert pixel to PTZ angles using simple geometric projection (fallback).
    
    This is a fast approximation that:
    1. Maps pixel X to camera-relative angle using HFOV
    2. Adds hardcoded camera yaw to get body-frame bearing
    3. Uses default tilt (doesn't compute from pixel Y)
    
    Use when:
    - Camera intrinsics unavailable
    - Transform mode fails
    - Bearing mode explicitly requested
    
    Args:
        pixel_x, pixel_y: Detection center coordinates
        img_width, img_height: Image dimensions
        hfov_deg: Horizontal field of view in degrees
        camera_yaw_deg: Camera yaw offset from body +X axis
        default_tilt_deg: Fixed tilt angle (doesn't vary with pixel_y)
        
    Returns:
        (pan_deg, tilt_deg) tuple in PTZ coordinates [0,360] and [-30,100]
    """
    logger.info(f"Using simple projection (fallback mode) for pixel ({pixel_x:.0f}, {pixel_y:.0f})")
    
    # Normalize pixel X to [-0.5, 0.5] centered
    norm_x = (pixel_x / img_width) - 0.5
    
    # Map to camera-frame angle using HFOV
    hfov_rad = np.deg2rad(hfov_deg)
    cam_angle_deg = np.rad2deg(norm_x * hfov_rad)
    
    # Add camera yaw to get body-frame bearing
    body_bearing_deg = camera_yaw_deg + cam_angle_deg
    
    # Normalize to [-180, 180]
    body_bearing_deg = np.rad2deg(
        math_helpers.recenter_angle_mod(np.deg2rad(body_bearing_deg), 0.0)
    )
    
    logger.info(f"Simple projection: cam_angle={cam_angle_deg:.2f}° + cam_yaw={camera_yaw_deg:.2f}° = bearing={body_bearing_deg:.2f}°")
    
    # Convert to PTZ pan [0, 360]
    ptz_pan_deg = -body_bearing_deg
    if ptz_pan_deg < 0:
        ptz_pan_deg += 360
    
    ptz_tilt_deg = default_tilt_deg
    
    logger.info(f"PTZ angles (simple): pan={ptz_pan_deg:.2f}°, tilt={ptz_tilt_deg:.2f}°")
    
    return ptz_pan_deg, ptz_tilt_deg


def get_camera_yaw_fallback(source_name: str) -> float:
    """Get hardcoded camera yaw offset from config.
    
    Used as fallback when frame transforms unavailable.
    
    Args:
        source_name: Camera source name (e.g., "frontleft_fisheye_image")
        
    Returns:
        Yaw offset in degrees relative to body +X axis
    """
    return CAM_YAW_DEG.get(source_name, 0.0)
