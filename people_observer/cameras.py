"""Camera utilities for surround capture with depth support.

Prefer batch calls to ImageClient.get_image so responses in a cycle share a
similar transforms_snapshot. Functions return both decoded images and raw
ImageResponses so downstream code can access intrinsics and frame transforms.

Functions:
- fetch_image_sources(image_client) -> dict
    Query ImageSource metadata including intrinsics and camera models (once at startup).
- get_camera_intrinsics(source_name, image_sources) -> dict | None
    Extract intrinsics (fx, fy, cx, cy), model type, and distortion coefficients.
- pixel_to_camera_ray(x, y, intrinsics) -> (x_norm, y_norm, z_norm)
    Convert pixel to normalized 3D ray using Kannala-Brandt or pinhole model.
- calculate_hfov_from_intrinsics(intrinsics) -> float
    Calculate horizontal FOV in degrees from focal length.
- ensure_available_sources(image_client, desired) -> list[str]
    Validate desired source names against the robot's advertised image sources.
- decode_image(resp) -> np.ndarray | None
    JPEG/RAW to BGR image decode for visualization/inference.
- get_frames(image_client, sources, include_depth=False) -> (frames, responses, depth_frames)
    Batch-fetch visual and optionally depth images using SDK build_image_request pattern.
    Returns decoded images, all responses, and depth images (in meters with NaN for invalid).
"""
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import logging
import math

import numpy as np
import cv2
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

from .config import FALLBACK_HFOV_DEG

logger = logging.getLogger(__name__)

# Cache for image source metadata (populated once at startup)
_image_sources_cache: Optional[Dict[str, image_pb2.ImageSource]] = None


def fetch_image_sources(image_client: ImageClient) -> Dict[str, image_pb2.ImageSource]:
    """Query robot for ImageSource metadata including intrinsics and camera models.
    
    Call once at startup. Caches results for fast repeated access.
    
    Returns:
        dict mapping source name to ImageSource proto
    """
    global _image_sources_cache
    
    if _image_sources_cache is None:
        sources = image_client.list_image_sources()
        _image_sources_cache = {src.name: src for src in sources}
        logger.info(f"Fetched {len(_image_sources_cache)} image sources from robot")
        
        # Log camera models and intrinsics availability with clear status
        for name, src in _image_sources_cache.items():
            if src.HasField('pinhole'):
                fx = src.pinhole.intrinsics.focal_length.x
                fy = src.pinhole.intrinsics.focal_length.y
                logger.info(f"{name}: PINHOLE model (fx={fx:.1f}, fy={fy:.1f})")
            elif src.HasField('kannala_brandt'):
                fx = src.kannala_brandt.intrinsics.focal_length.x
                fy = src.kannala_brandt.intrinsics.focal_length.y
                k1 = src.kannala_brandt.k1
                k2 = src.kannala_brandt.k2
                k3 = src.kannala_brandt.k3
                k4 = src.kannala_brandt.k4
                logger.info(f"{name}: KANNALA-BRANDT fisheye (fx={fx:.1f}, fy={fy:.1f}, k1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}, k4={k4:.4f})")
            else:
                logger.warning(f"{name}: NO CAMERA MODEL FOUND - will use fallback projection")
    
    return _image_sources_cache


def get_camera_intrinsics(source_name: str, image_sources: Dict[str, image_pb2.ImageSource]) -> Optional[dict]:
    """Extract camera intrinsics and model type from ImageSource.
    
    Args:
        source_name: Camera source name
        image_sources: Dict from fetch_image_sources()
        
    Returns:
        dict with keys:
            - 'model': 'pinhole', 'kannala_brandt', or None
            - 'fx', 'fy': focal lengths in pixels
            - 'cx', 'cy': principal point in pixels
            - 'distortion': [k1, k2, k3, k4] for fisheye, [] for pinhole
            - 'width', 'height': image dimensions
        Returns None if source not found or no camera model
    """
    if source_name not in image_sources:
        logger.warning(f"Unknown source {source_name}")
        return None
    
    src = image_sources[source_name]
    result = {
        'width': src.cols,
        'height': src.rows,
        'model': None,
        'fx': None,
        'fy': None,
        'cx': None,
        'cy': None,
        'distortion': []
    }
    
    # Extract pinhole model
    if src.HasField('pinhole'):
        result['model'] = 'pinhole'
        result['fx'] = src.pinhole.intrinsics.focal_length.x
        result['fy'] = src.pinhole.intrinsics.focal_length.y
        result['cx'] = src.pinhole.intrinsics.principal_point.x
        result['cy'] = src.pinhole.intrinsics.principal_point.y
        result['distortion'] = []  # Pinhole has no distortion in SDK
        
    # Extract Kannala-Brandt fisheye model
    elif src.HasField('kannala_brandt'):
        result['model'] = 'kannala_brandt'
        result['fx'] = src.kannala_brandt.intrinsics.focal_length.x
        result['fy'] = src.kannala_brandt.intrinsics.focal_length.y
        result['cx'] = src.kannala_brandt.intrinsics.principal_point.x
        result['cy'] = src.kannala_brandt.intrinsics.principal_point.y
        result['distortion'] = [
            src.kannala_brandt.k1,
            src.kannala_brandt.k2,
            src.kannala_brandt.k3,
            src.kannala_brandt.k4
        ]
    else:
        logger.warning(f"{source_name}: no camera model in ImageSource")
        return None
    
    return result


def pixel_to_camera_ray(x: float, y: float, intrinsics: dict) -> Tuple[float, float, float]:
    """Convert pixel coordinates to normalized 3D ray in camera frame.
    
    Uses OpenCV fisheye undistortion for Kannala-Brandt model, or simple
    pinhole projection for rectilinear cameras.
    
    Args:
        x, y: Pixel coordinates
        intrinsics: Dict from get_camera_intrinsics()
        
    Returns:
        (x_norm, y_norm, z_norm): Unit vector in camera frame (z forward)
    """
    if intrinsics is None or intrinsics['model'] is None:
        raise ValueError("No valid camera model in intrinsics")
    
    # Build camera matrix for OpenCV
    K = np.array([
        [intrinsics['fx'], 0, intrinsics['cx']],
        [0, intrinsics['fy'], intrinsics['cy']],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Pixel as 2D point
    pixel = np.array([[x, y]], dtype=np.float32).reshape(1, 1, 2)
    
    if intrinsics['model'] == 'kannala_brandt':
        # Use OpenCV fisheye undistortion
        D = np.array(intrinsics['distortion'], dtype=np.float32)
        
        # Undistort to normalized image plane
        normalized = cv2.fisheye.undistortPoints(pixel, K, D)
        x_norm = float(normalized[0, 0, 0])
        y_norm = float(normalized[0, 0, 1])
        z_norm = 1.0
        
    elif intrinsics['model'] == 'pinhole':
        # Simple pinhole projection (no distortion)
        x_norm = (x - intrinsics['cx']) / intrinsics['fx']
        y_norm = (y - intrinsics['cy']) / intrinsics['fy']
        z_norm = 1.0
    else:
        raise ValueError(f"Unsupported camera model: {intrinsics['model']}")
    
    # Normalize to unit vector
    length = math.sqrt(x_norm**2 + y_norm**2 + z_norm**2)
    return (x_norm / length, y_norm / length, z_norm / length)


def calculate_hfov_from_intrinsics(intrinsics: Optional[dict]) -> float:
    """Calculate horizontal FOV in degrees from intrinsics.
    
    Args:
        intrinsics: Dict from get_camera_intrinsics()
        
    Returns:
        Horizontal FOV in degrees (or fallback value if intrinsics unavailable)
    """
    if intrinsics is None or intrinsics['fx'] is None:
        return FALLBACK_HFOV_DEG
    
    fx = intrinsics['fx']
    width = intrinsics['width']
    
    if fx > 0 and width > 0:
        hfov_rad = 2 * math.atan(width / (2 * fx))
        hfov_deg = math.degrees(hfov_rad)
        return hfov_deg
    
    return FALLBACK_HFOV_DEG


def ensure_available_sources(image_client: ImageClient, desired: List[str]) -> List[str]:
    """Return a filtered list of desired image source names that exist on the robot.

    Inputs:
    - image_client: Spot ImageClient
    - desired: list of preferred source names

    Output: list of usable source names (order preserved from desired).
    Raises RuntimeError if none are available.
    """
    available = {s.name for s in image_client.list_image_sources()}
    usable = [s for s in desired if s in available]
    if not usable:
        raise RuntimeError(f"No desired surround cameras found. Available: {sorted(available)}")
    return usable


def decode_image(resp) -> np.ndarray:
    """Decode an ImageResponse to a BGR ndarray.

    Supports JPEG and RAW (grayscale) images. Returns a 3-channel BGR image.
    """
    fmt = resp.shot.image.format
    data = resp.shot.image.data
    if fmt == image_pb2.Image.FORMAT_JPEG:
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif fmt == image_pb2.Image.FORMAT_RAW:
        rows, cols = resp.shot.image.rows, resp.shot.image.cols
        img = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = None
    return img


def _decode_depth_image(resp) -> Optional[np.ndarray]:
    """Decode depth ImageResponse to float32 ndarray in meters.
    
    Uses SDK pattern: depth_scale from ImageSource converts uint16 pixels to meters.
    Invalid pixels (0 and 65535) are marked as NaN.
    
    Args:
        resp: ImageResponse with depth data
        
    Returns:
        np.ndarray of shape (rows, cols) with depth in meters, or None if not a depth image
    """
    # Check pixel format
    if resp.shot.image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        return None
    
    rows = resp.shot.image.rows
    cols = resp.shot.image.cols
    data = resp.shot.image.data
    
    # Decode uint16 depth data (SDK pattern)
    depth_u16 = np.frombuffer(data, dtype=np.uint16).reshape(rows, cols)
    
    # Get scale factor from ImageSource (depth_scale converts pixel value to meters)
    depth_scale = resp.source.depth_scale if resp.source.depth_scale > 0 else 1.0
    
    # Convert to float32 meters: depth_m = pixel_value / depth_scale
    depth_m = depth_u16.astype(np.float32) / depth_scale
    
    # Mask out invalid depth values (0 and 65535 are invalid per SDK)
    depth_m[(depth_u16 == 0) | (depth_u16 == 65535)] = np.nan
    
    return depth_m


def get_frames(image_client: ImageClient, sources: List[str], include_depth: bool = False):
    """Fetch a batch of images for the given sources using SDK build_image_request pattern.

    Inputs:
    - image_client: Spot ImageClient
    - sources: list of visual image source names
    - include_depth: if True, also fetch corresponding depth_in_visual_frame images

    Returns:
    - frames: OrderedDict mapping source name -> BGR image (np.ndarray)
    - responses: list of ALL ImageResponse objects (visual + depth)
    - depth_frames: OrderedDict mapping visual source name -> depth in meters (np.ndarray)
    """
    frames: "OrderedDict[str, np.ndarray]" = OrderedDict()
    depth_frames: "OrderedDict[str, np.ndarray]" = OrderedDict()
    
    # Build ImageRequest list using SDK pattern
    image_requests = []
    for src in sources:
        # Visual image request (JPEG format by default)
        image_requests.append(build_image_request(src))
        
        # Add depth request if requested
        if include_depth:
            # Construct depth source name: strip _fisheye_image/_image suffix, add _depth_in_visual_frame
            base_name = src.replace('_fisheye_image', '').replace('_image', '').replace('_depth', '')
            depth_source = f"{base_name}_depth_in_visual_frame"
            # Request depth in RAW format with DEPTH_U16 pixel format
            image_requests.append(build_image_request(
                depth_source,
                pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
                image_format=image_pb2.Image.FORMAT_RAW
            ))
    
    # Fetch all images in one batch call
    responses = image_client.get_image(image_requests)
    
    # Separate visual and depth responses
    for r in responses:
        source_name = r.source.name
        
        # Check if this is a depth source
        if '_depth_in_visual_frame' in source_name:
            depth_img = _decode_depth_image(r)
            if depth_img is not None:
                # Map depth back to the visual source name for easy lookup
                visual_name = source_name.replace('_depth_in_visual_frame', '_fisheye_image')
                # Handle cases where visual source might not have _fisheye_image suffix
                if visual_name not in sources:
                    visual_name = source_name.replace('_depth_in_visual_frame', '_image')
                depth_frames[visual_name] = depth_img
        else:
            # Visual image
            img = decode_image(r)
            if img is not None:
                frames[source_name] = img
    
    return frames, responses, depth_frames


def get_depth_at_pixel(depth_img: np.ndarray, pixel_x: int, pixel_y: int) -> Optional[float]:
    """Extract depth value at a specific pixel from decoded depth image.
    
    Utility function for single-pixel depth queries. Useful for computing 3D points
    with SDK's pixel_to_camera_space() function.
    
    Args:
        depth_img: Decoded depth image in meters (with NaN for invalid pixels)
        pixel_x: X pixel coordinate
        pixel_y: Y pixel coordinate
        
    Returns:
        Depth in meters at the pixel, or None if invalid or out of bounds
    """
    if depth_img is None:
        return None
    
    img_h, img_w = depth_img.shape
    if not (0 <= pixel_x < img_w and 0 <= pixel_y < img_h):
        return None
    
    depth_m = depth_img[pixel_y, pixel_x]
    
    # Check for invalid depth (NaN)
    if np.isnan(depth_m):
        return None
    
    return float(depth_m)
