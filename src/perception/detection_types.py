# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Unified detection types and data structures for people tracking with depth validation,
# distance estimation heuristics, and shared interfaces between tracking and perception modules
# Acknowledgements: NumPy for array operations, Claude for dataclass design and depth validation logic

"""Unified detection types for people tracking and perception pipeline.

Shared data structures between people_observer (tracking) and friendly_spot (perception).
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import time


@dataclass
class PersonDetection:
    """Unified person detection with segmentation, depth, and tracking info.
    
    This class bridges people_observer (person detection/tracking in surround cameras)
    and friendly_spot perception pipeline (pose/face/emotion/gesture analysis).
    
    Attributes:
        bbox_xywh: Bounding box as (x, y, width, height) in pixels
        mask: Segmentation mask as boolean numpy array (H, W), or None if unavailable
        confidence: Detection confidence [0.0, 1.0]
        distance_m: Distance to person in meters, or None if unavailable
        depth_source: How distance was calculated: 'mask+depth', 'bbox+depth', 'heuristic', or 'unknown'
        source_camera: Which camera detected person (e.g., 'frontleft_fisheye_image', 'ptz')
        tracked_by_ptz: Whether PTZ camera is currently tracking this person
        tracking_quality: PTZ tracking quality [0.0, 1.0], 1.0 = centered and stable
        detection_time: Timestamp when first detected (seconds since epoch)
        last_seen_time: Timestamp of last update (seconds since epoch)
        frame: Optional BGR frame containing the person (for visualization/debugging)
        depth_frame: Optional depth image in meters (H, W) with NaN for invalid pixels
    """
    # Detection info
    bbox_xywh: Tuple[int, int, int, int]
    mask: Optional[np.ndarray] = None
    confidence: float = 0.0
    
    # Depth/distance
    distance_m: Optional[float] = None
    depth_source: str = "unknown"  # 'mask+depth', 'bbox+depth', 'heuristic', 'unknown'
    depth_validated: bool = False  # True if depth passes sanity check vs heuristic
    
    # Tracking info
    source_camera: str = "unknown"
    tracked_by_ptz: bool = False
    tracking_quality: float = 0.0
    
    # Timestamps
    detection_time: float = field(default_factory=time.time)
    last_seen_time: float = field(default_factory=time.time)
    
    # Optional frames (for visualization or further processing)
    frame: Optional[np.ndarray] = None
    depth_frame: Optional[np.ndarray] = None
    
    def update_last_seen(self):
        """Update last_seen_time to current time."""
        self.last_seen_time = time.time()
    
    def age_seconds(self) -> float:
        """Return age of detection in seconds."""
        return time.time() - self.detection_time
    
    def time_since_seen(self) -> float:
        """Return seconds since last update."""
        return time.time() - self.last_seen_time
    
    def bbox_area(self) -> int:
        """Return bounding box area in pixels."""
        return self.bbox_xywh[2] * self.bbox_xywh[3]
    
    def bbox_center(self) -> Tuple[float, float]:
        """Return bounding box center (cx, cy) in pixels."""
        x, y, w, h = self.bbox_xywh
        return (x + w / 2, y + h / 2)
    
    def has_mask(self) -> bool:
        """Check if segmentation mask is available."""
        return self.mask is not None
    
    def has_depth(self) -> bool:
        """Check if depth measurement is available."""
        return self.distance_m is not None and self.distance_m > 0
    
    def is_stale(self, timeout_sec: float = 1.0) -> bool:
        """Check if detection is stale (not updated recently)."""
        return self.time_since_seen() > timeout_sec


def validate_depth_against_heuristic(
    depth_distance: Optional[float],
    bbox_xywh: Tuple[int, int, int, int],
    frame_height: int,
    assumed_person_height_m: float = 1.7,
    tolerance_factor: float = 2.0
) -> bool:
    """Validate depth sensor distance against heuristic bbox-based estimate.
    
    Performs sanity check: is depth-based distance within reasonable range
    of what bbox size would suggest? Catches cases where depth sensor gives
    wildly incorrect values (e.g., reflections, multipath, transparent surfaces).
    
    Args:
        depth_distance: Distance from depth sensor in meters
        bbox_xywh: Bounding box (x, y, width, height) in pixels
        frame_height: Frame height in pixels (for heuristic calculation)
        assumed_person_height_m: Assumed height of person in meters
        tolerance_factor: How much deviation to allow (2.0 = allow 2x difference)
    
    Returns:
        True if depth distance is reasonable, False if suspiciously different
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if depth_distance is None or depth_distance <= 0:
        logger.debug(f"Depth validation failed: invalid depth value ({depth_distance})")
        return False
    
    # Calculate heuristic distance from bbox size
    _, _, w, h = bbox_xywh
    pixel_height = h
    if pixel_height <= 5:
        logger.debug(f"Depth validation failed: bbox too small (height={pixel_height}px)")
        return False  # Bbox too small, can't validate
    
    # Approximate focal length in pixels (heuristic)
    focal_px = frame_height * 1.2
    heuristic_distance = (assumed_person_height_m * focal_px) / pixel_height
    
    # Check if depth distance is within tolerance factor
    ratio = depth_distance / heuristic_distance
    
    # Allow tolerance_factor difference in either direction
    is_valid = (1.0 / tolerance_factor) <= ratio <= tolerance_factor
    
    logger.debug(f"Depth validation: depth={depth_distance:.2f}m, heuristic={heuristic_distance:.2f}m, ratio={ratio:.2f}, valid={is_valid} (tolerance={tolerance_factor}x)")
    
    return is_valid


def estimate_distance_from_bbox(
    bbox_xywh: Tuple[int, int, int, int],
    frame_height: int,
    assumed_person_height_m: float = 1.7
) -> float:
    """Fallback heuristic distance estimate from bounding box size.
    
    Uses pinhole camera model: distance = (object_height * focal_length) / pixel_height
    
    Args:
        bbox_xywh: Bounding box (x, y, width, height) in pixels
        frame_height: Frame height in pixels
        assumed_person_height_m: Assumed height of person in meters
    
    Returns:
        Estimated distance in meters
    """
    import logging
    logger = logging.getLogger(__name__)
    
    _, _, w, h = bbox_xywh
    pixel_height = h
    logger.debug(f"Estimating distance from bbox: bbox=({bbox_xywh}), frame_height={frame_height}, pixel_height={pixel_height}")
    
    if pixel_height <= 5:
        logger.debug(f"Bbox too small (height={pixel_height}px), using default distance 3.0m")
        return 3.0  # Default fallback distance
    
    # Approximate focal length (heuristic)
    focal_px = frame_height * 1.2
    distance_m = (assumed_person_height_m * focal_px) / pixel_height
    raw_distance = distance_m
    
    # Clamp to reasonable range (0.5m to 10m)
    distance_m = max(0.5, min(distance_m, 10.0))
    
    if distance_m != raw_distance:
        logger.debug(f"Distance clamped: {raw_distance:.2f}m -> {distance_m:.2f}m (range: 0.5-10.0m)")
    else:
        logger.debug(f"Heuristic distance estimate: {distance_m:.2f}m")
    
    return distance_m
