"""Camera utilities for surround capture.

Prefer batch calls to ImageClient.get_image_from_sources so responses in a cycle
share a similar transforms_snapshot. Functions here do minimal decode and return
both images and the raw ImageResponses so downstream code can access intrinsics
and frame transforms.

Functions
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
    JPEG/RAW to BGR image decode for visualization/inference. Returns None if
    format is unsupported.
- get_frames(image_client, sources) -> (OrderedDict[name->image], list[resp])
    Batch-fetch frames. Returns decoded images and their corresponding
    ImageResponses in the same order for metadata access.
"""
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import logging
import math

import numpy as np
import cv2
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient

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
                logger.info(f"  ✓ {name}: PINHOLE model (fx={fx:.1f}, fy={fy:.1f})")
            elif src.HasField('kannala_brandt'):
                fx = src.kannala_brandt.intrinsics.focal_length.x
                fy = src.kannala_brandt.intrinsics.focal_length.y
                k1 = src.kannala_brandt.k1
                k2 = src.kannala_brandt.k2
                k3 = src.kannala_brandt.k3
                k4 = src.kannala_brandt.k4
                logger.info(f"  ✓ {name}: KANNALA-BRANDT fisheye (fx={fx:.1f}, fy={fy:.1f}, k1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}, k4={k4:.4f})")
            else:
                logger.warning(f"  ✗ {name}: NO CAMERA MODEL FOUND - will use fallback projection")
    
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


def get_frames(image_client: ImageClient, sources: List[str]):
    """Fetch a batch of images for the given sources.

    Inputs:
    - image_client: Spot ImageClient
    - sources: list of image source names

    Returns:
    - frames: OrderedDict mapping source name -> BGR image (np.ndarray)
    - responses: list of ImageResponse objects in the same order as request
    """
    frames: "OrderedDict[str, np.ndarray]" = OrderedDict()
    responses = image_client.get_image_from_sources(sources)
    for r in responses:
        img = decode_image(r)
        if img is not None:
            frames[r.source.name] = img
    return frames, responses
