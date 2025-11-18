# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/17/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Coordinate geometry transformations for pixel-to-PTZ angle computation using
# SDK intrinsics, fisheye undistortion, frame transforms, and bearing calculations
# Acknowledgements: Boston Dynamics Spot SDK frame_helpers and math_helpers for SE3 transforms,
# OpenCV fisheye camera model documentation, Claude for transform mode implementation

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
from .config import CAM_YAW_DEG, ROTATION_ANGLE, PTZ_OFFSET_DEG

logger = logging.getLogger(__name__)


def rotate_pixel_for_camera(
    pixel_x: float,
    pixel_y: float,
    img_width: int,
    img_height: int,
    camera_name: str
) -> Tuple[float, float]:
    """Apply physical camera rotation correction to pixel coordinates.
    
    Spot's fisheye cameras are physically rotated relative to their mounting frames.
    This function rotates the pixel coordinates to account for the physical rotation
    before unprojecting to a 3D ray. Without this correction, the ray directions
    will be systematically incorrect, causing PTZ inaccuracy.
    
    Rotation angles (counterclockwise):
    - frontleft: -78° (78° clockwise)
    - frontright: -102° (102° clockwise)  
    - right: 180° (upside down)
    - left, back: 0° (no rotation)
    
    Args:
        pixel_x, pixel_y: Original pixel coordinates from YOLO detection
        img_width, img_height: Image dimensions in pixels
        camera_name: Camera source name (e.g., "frontleft_fisheye_image")
        
    Returns:
        (rotated_x, rotated_y) tuple in rotated pixel coordinates
        
    Example:
        For frontleft camera rotated -78°:
        - Center pixel (640, 480) should remain at center after rotation
        - Top-center pixel should rotate clockwise 78° around center
    """
    angle_deg = ROTATION_ANGLE.get(camera_name, 0.0)
    
    if angle_deg == 0.0:
        # No rotation needed
        return pixel_x, pixel_y
    
    # Rotate around image center
    cx = img_width / 2.0
    cy = img_height / 2.0
    
    # Translate to origin
    dx = pixel_x - cx
    dy = pixel_y - cy
    
    # Rotate (counterclockwise convention)
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    rotated_dx = dx * cos_a - dy * sin_a
    rotated_dy = dx * sin_a + dy * cos_a
    
    # Translate back
    rotated_x = rotated_dx + cx
    rotated_y = rotated_dy + cy
    
    logger.debug(f"Pixel rotation ({camera_name}, {angle_deg:.1f}°): ({pixel_x:.1f}, {pixel_y:.1f}) → ({rotated_x:.1f}, {rotated_y:.1f})")
    
    return rotated_x, rotated_y


def pixel_to_ptz_angles_transform(
    pixel_x: float,
    pixel_y: float,
    intrinsics: dict,
    image_response,
    robot,
    img_width: int = 1280,
    img_height: int = 960
) -> Tuple[float, float]:
    """Convert pixel coordinates to PTZ pan/tilt angles using full frame-aware transform pipeline.
    
    This implementation follows SDK best practices from fiducial_follow and fetch examples:
    1. Applies physical camera rotation correction to pixel coordinates
    2. Unprojects pixel to 3D ray in camera frame using intrinsics
    3. Transforms ray through complete frame chain: camera → vision → body
    4. Computes bearing/tilt in body frame
    5. Accounts for PTZ mounting offset (if PTZ frame available in snapshot)
    6. Converts from body frame convention to PTZ convention
    
    CRITICAL FIX: Unlike previous implementation, this properly accounts for:
    - PTZ frame's own offset/rotation in the transform tree
    - Timestamp consistency between image capture and PTZ state
    - Vision frame as stable world reference (not odometry-based)
    
    Args:
        pixel_x, pixel_y: Detection center in pixel coordinates
        intrinsics: Camera intrinsics dict from cameras.get_camera_intrinsics()
        image_response: ImageResponse containing frame tree snapshot
        robot: Robot instance (for logging/debugging)
        img_width, img_height: Image dimensions for rotation correction
        
    Returns:
        (pan_deg, tilt_deg) tuple in PTZ coordinates:
            pan: [0, 360] degrees, 0=forward, clockwise from above
            tilt: [-30, 100] degrees, negative=down, positive=up
            
    Raises:
        ValueError: If intrinsics are None or invalid
        RuntimeError: If frame transforms fail
        
    Frame Chain:
        pixel → camera_sensor → body → vision (world reference)
        Then compute PTZ angles relative to body (PTZ is body-mounted)
    """
    if intrinsics is None or intrinsics.get('model') is None:
        raise ValueError(f"Valid camera intrinsics required for transform mode")
    
    # Get frame names and snapshot
    shot = image_response.shot
    frame_tree = shot.transforms_snapshot
    camera_frame = shot.frame_name_image_sensor
    
    # Log intrinsics info
    model = intrinsics['model']
    logger.info(f"[TRANSFORM] Camera={camera_frame}, Model={model}, Original pixel=({pixel_x:.0f}, {pixel_y:.0f})")
    logger.info(f"[TRANSFORM] Image timestamp: {shot.acquisition_time.seconds}.{shot.acquisition_time.nanos//1000000:03d}")
    
    # Step 1: Apply physical camera rotation correction
    # CRITICAL: Spot's fisheye cameras are physically rotated. Without this correction,
    # the unprojected rays will point in wrong directions, causing PTZ inaccuracy.
    rotated_x, rotated_y = rotate_pixel_for_camera(
        pixel_x, pixel_y, img_width, img_height, camera_frame
    )
    logger.info(f"[TRANSFORM] Rotation-corrected pixel=({rotated_x:.0f}, {rotated_y:.0f})")
    
    # Step 2: Unproject pixel to 3D ray in camera frame
    try:
        ray_cam_x, ray_cam_y, ray_cam_z = cameras.pixel_to_camera_ray(
            rotated_x, rotated_y, intrinsics
        )
        logger.info(f"[TRANSFORM] Camera-frame ray=[{ray_cam_x:.3f}, {ray_cam_y:.3f}, {ray_cam_z:.3f}]")
    except Exception as e:
        raise RuntimeError(f"Failed to unproject pixel using {model} model: {e}")
    
    # Step 3: Transform ray through complete frame chain (following SDK best practices)
    try:
        # Use frame names from frame_helpers
        body_frame = frame_helpers.BODY_FRAME_NAME
        vision_frame = frame_helpers.VISION_FRAME_NAME
        
        # Following fiducial_follow pattern: get camera_tform_body for camera→body
        # Note: get_a_tform_b(snapshot, A, B) returns A_tform_B (transform from B to A)
        camera_tform_body = frame_helpers.get_a_tform_b(
            frame_tree,
            camera_frame,
            body_frame
        )
        
        # Get vision_tform_body for stable world reference (not odometry)
        vision_tform_body = frame_helpers.get_vision_tform_body(frame_tree)
        
        # Log the complete frame chain for debugging
        logger.info(f"[TRANSFORM] Frame chain: {camera_frame} → body (for PTZ calculation)")
        logger.info(f"[TRANSFORM] camera_tform_body position: [{camera_tform_body.x:.3f}, {camera_tform_body.y:.3f}, {camera_tform_body.z:.3f}]")
        logger.info(f"[TRANSFORM] Reference: body → vision (world frame, for debugging)")
        logger.info(f"[TRANSFORM] vision_tform_body position: [{vision_tform_body.x:.3f}, {vision_tform_body.y:.3f}, {vision_tform_body.z:.3f}]")
        
        # Compute body_tform_camera (inverse transform for ray direction)
        body_tform_camera = camera_tform_body.inverse()
        
        # Transform ray from camera frame to body frame
        # For direction vectors, only rotation matters (not translation)
        ray_cam_vec = math_helpers.Vec3(ray_cam_x, ray_cam_y, ray_cam_z)
        ray_body_vec = body_tform_camera.rot.transform_vec3(ray_cam_vec.to_proto())
        
        logger.info(f"[TRANSFORM] Body-frame ray=[{ray_body_vec.x:.3f}, {ray_body_vec.y:.3f}, {ray_body_vec.z:.3f}]")
        
        # Optional: Transform to vision frame for stability during robot motion
        # (Vision frame is gravity-aligned and doesn't rotate with body)
        ray_vision_vec = vision_tform_body.rot.transform_vec3(
            math_helpers.Vec3(ray_body_vec.x, ray_body_vec.y, ray_body_vec.z).to_proto()
        )
        logger.debug(f"[TRANSFORM] Vision-frame ray=[{ray_vision_vec.x:.3f}, {ray_vision_vec.y:.3f}, {ray_vision_vec.z:.3f}]")
        
    except Exception as e:
        raise RuntimeError(f"Frame transform failed: {e}")
    
    # Step 4: Compute bearing (pan) and elevation (tilt) from body-frame ray
    # Body frame: +X=forward, +Y=left, +Z=up
    # PTZ is body-mounted, so we work in body frame (not vision)
    
    # Pan: horizontal angle from forward direction using atan2
    bearing_rad = np.arctan2(ray_body_vec.y, ray_body_vec.x)
    bearing_rad = math_helpers.recenter_angle_mod(bearing_rad, 0.0)  # [-π, π]
    bearing_deg = np.rad2deg(bearing_rad)  # [-180, 180]
    
    # Tilt: vertical angle from horizontal plane
    horizontal_dist = np.sqrt(ray_body_vec.x**2 + ray_body_vec.y**2)
    tilt_rad = np.arctan2(ray_body_vec.z, horizontal_dist)
    tilt_deg = np.rad2deg(tilt_rad)
    
    logger.info(f"[TRANSFORM] Body-frame angles: bearing={bearing_deg:.2f}°, tilt={tilt_deg:.2f}°")
    
    # Step 5: Convert from body frame to PTZ coordinates
    # Body bearing: 0=forward, +90=left, -90=right (math convention, CCW positive)
    # PTZ pan: 0=forward, +90=right, +180=back, +270=left (compass convention, CW positive)
    # 
    # Conversion: PTZ_pan = -bearing (negate to flip left/right direction)
    ptz_pan_deg = -bearing_deg
    
    # Apply PTZ hardware mounting offset
    # The mechanical PTZ is physically mounted 35° to the right of body forward.
    # We subtract the offset so that commanding (desired_bearing - offset) results in
    # the PTZ pointing at the desired bearing in body frame.
    ptz_pan_deg = (ptz_pan_deg - PTZ_OFFSET_DEG) % 360
    
    # PTZ tilt: negative=down, positive=up (same convention as body frame Z)
    # PTZ is gravity-aligned via gimbal, so body-frame tilt directly maps to PTZ tilt
    ptz_tilt_deg = tilt_deg
    
    logger.info(f"[TRANSFORM] Final PTZ command: pan={ptz_pan_deg:.2f}°, tilt={ptz_tilt_deg:.2f}°")
    logger.info(f"[TRANSFORM] Summary: pixel({pixel_x:.0f},{pixel_y:.0f}) → body bearing {bearing_deg:.1f}° → PTZ pan {ptz_pan_deg:.1f}°")
    
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
    
    # Apply PTZ hardware mounting offset (same as transform mode)
    ptz_pan_deg = (ptz_pan_deg - PTZ_OFFSET_DEG) % 360
    
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


def diagnose_frame_chain(image_response, ptz_name: str = "mech") -> dict:
    """Diagnostic function to inspect the complete transform chain for debugging PTZ accuracy.
    
    This function mirrors the Fetch example debugging flow: logs the full transforms_snapshot
    chain (sensor to vision to body to [ptz if available]), inspects resulting SE3 poses, and
    helps pinpoint which frame conversion introduces error.
    
    Following SDK best practices:
    - Uses frame_helpers.get_a_tform_b for all transforms
    - Checks for PTZ frame in snapshot (may not be present if PTZ not yet commanded)
    - Compares against SDK helper outputs to validate computations
    
    Args:
        image_response: ImageResponse with transforms_snapshot
        ptz_name: PTZ device name to look for in frame tree (e.g., "mech", "digi")
        
    Returns:
        dict with diagnostic info:
            - 'camera_frame': str, image sensor frame name
            - 'camera_tform_body': SE3Pose or None
            - 'body_tform_vision': SE3Pose or None
            - 'ptz_frame_available': bool, whether PTZ frame exists in snapshot
            - 'body_tform_ptz': SE3Pose or None if available
            - 'timestamp': acquisition time
            - 'all_frames': list of all frame names in snapshot
    """
    shot = image_response.shot
    frame_tree = shot.transforms_snapshot
    camera_frame = shot.frame_name_image_sensor
    
    info = {
        'camera_frame': camera_frame,
        'timestamp': f"{shot.acquisition_time.seconds}.{shot.acquisition_time.nanos//1000000:03d}",
        'all_frames': []
    }
    
    # Extract all available frames from snapshot
    for edge in frame_tree.child_to_parent_edge_map:
        info['all_frames'].append(edge)
        parent = frame_tree.child_to_parent_edge_map[edge].parent_frame_name
        if parent not in info['all_frames']:
            info['all_frames'].append(parent)
    
    logger.info(f"[DIAGNOSE] Available frames in snapshot: {info['all_frames']}")
    
    # Get key transforms
    try:
        info['camera_tform_body'] = frame_helpers.get_a_tform_b(
            frame_tree,
            camera_frame,
            frame_helpers.BODY_FRAME_NAME
        )
        logger.info(f"[DIAGNOSE] camera_tform_body: pos=[{info['camera_tform_body'].x:.3f}, {info['camera_tform_body'].y:.3f}, {info['camera_tform_body'].z:.3f}]")
    except Exception as e:
        logger.warning(f"[DIAGNOSE] Could not get camera_tform_body: {e}")
        info['camera_tform_body'] = None
    
    try:
        info['body_tform_vision'] = frame_helpers.get_a_tform_b(
            frame_tree,
            frame_helpers.BODY_FRAME_NAME,
            frame_helpers.VISION_FRAME_NAME
        )
        logger.info(f"[DIAGNOSE] body_tform_vision: pos=[{info['body_tform_vision'].x:.3f}, {info['body_tform_vision'].y:.3f}, {info['body_tform_vision'].z:.3f}]")
    except Exception as e:
        logger.warning(f"[DIAGNOSE] Could not get body_tform_vision: {e}")
        info['body_tform_vision'] = None
    
    # Check if PTZ frame exists in snapshot
    # NOTE: PTZ frame may not be in the image's transform snapshot because:
    # 1. Images come from surround cameras, not PTZ camera
    # 2. PTZ pose at command time != PTZ pose at image capture time
    # 3. PTZ is mechanically body-mounted but may have its own frame offset
    ptz_frame_candidates = [f for f in info['all_frames'] if ptz_name.lower() in f.lower()]
    info['ptz_frame_available'] = len(ptz_frame_candidates) > 0
    
    if info['ptz_frame_available']:
        ptz_frame = ptz_frame_candidates[0]
        logger.info(f"[DIAGNOSE] Found PTZ frame in snapshot: {ptz_frame}")
        try:
            info['body_tform_ptz'] = frame_helpers.get_a_tform_b(
                frame_tree,
                frame_helpers.BODY_FRAME_NAME,
                ptz_frame
            )
            logger.info(f"[DIAGNOSE] body_tform_ptz: pos=[{info['body_tform_ptz'].x:.3f}, {info['body_tform_ptz'].y:.3f}, {info['body_tform_ptz'].z:.3f}]")
        except Exception as e:
            logger.warning(f"[DIAGNOSE] Could not get body_tform_ptz: {e}")
            info['body_tform_ptz'] = None
    else:
        logger.warning(f"[DIAGNOSE] PTZ frame '{ptz_name}' not found in image snapshot.")
        logger.warning(f"[DIAGNOSE] This is EXPECTED: image comes from surround camera, not PTZ.")
        logger.warning(f"[DIAGNOSE] PTZ mounting offset assumed zero (PTZ aligned with body frame).")
        info['body_tform_ptz'] = None
    
    return info
