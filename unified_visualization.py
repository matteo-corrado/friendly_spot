"""Unified visualization for friendly_spot and people_observer.

Provides consistent visualization across:
- people_observer: Multi-camera person detection with depth masks
- friendly_spot: Perception pipeline results (pose, face, emotion, gesture)

All visualization functions support saving to disk and live display.
"""
import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Visualization constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_SCALE_SMALL = 0.4
FONT_THICKNESS = 2
LINE_THICKNESS = 2
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
MASK_ALPHA = 0.4


def create_depth_colormap(depth_img: np.ndarray, min_dist: float = 0.5, 
                          max_dist: float = 5.0) -> np.ndarray:
    """Convert depth image to color-coded visualization.
    
    Args:
        depth_img: Depth in meters (float32, NaN for invalid)
        min_dist: Minimum distance for color scale
        max_dist: Maximum distance for color scale
    
    Returns:
        BGR image with depth as heatmap (blue=close, red=far)
    """
    depth_norm = np.copy(depth_img)
    valid_mask = ~np.isnan(depth_norm)
    
    depth_norm[valid_mask] = np.clip(depth_norm[valid_mask], min_dist, max_dist)
    depth_norm[valid_mask] = (depth_norm[valid_mask] - min_dist) / (max_dist - min_dist)
    
    depth_u8 = np.zeros_like(depth_norm, dtype=np.uint8)
    depth_u8[valid_mask] = (depth_norm[valid_mask] * 255).astype(np.uint8)
    
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    colored[~valid_mask] = [0, 0, 0]
    
    return colored


def draw_detection_with_mask(image: np.ndarray, bbox: Tuple[int, int, int, int],
                             confidence: float, distance: Optional[float] = None,
                             mask: Optional[np.ndarray] = None,
                             depth_img: Optional[np.ndarray] = None,
                             label: str = "Person") -> np.ndarray:
    """Draw detection with optional segmentation mask and depth overlay.
    
    Args:
        image: BGR image to annotate
        bbox: (x, y, w, h) bounding box
        confidence: Detection confidence
        distance: Distance in meters (if available)
        mask: Segmentation mask (bool array, same size as image)
        depth_img: Depth image for color-coded mask
        label: Detection label
    
    Returns:
        Annotated image
    """
    img = image.copy()
    h, w = img.shape[:2]
    x, y, bbox_w, bbox_h = bbox
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w, int(x + bbox_w)), min(h, int(y + bbox_h))
    
    # Draw segmentation mask if available
    if mask is not None:
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), 
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        
        overlay = img.copy()
        if depth_img is not None and depth_img.shape[:2] == (h, w):
            depth_colored = create_depth_colormap(depth_img)
            overlay[mask] = depth_colored[mask]
        else:
            overlay[mask] = COLOR_YELLOW  # Cyan for no depth
        
        img = cv2.addWeighted(overlay, MASK_ALPHA, img, 1 - MASK_ALPHA, 0)
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_GREEN, LINE_THICKNESS)
    
    # Draw label with confidence and distance
    if distance is not None:
        text = f"{label} {confidence:.2f} | {distance:.2f}m"
    else:
        text = f"{label} {confidence:.2f}"
    
    text_size = cv2.getTextSize(text, FONT, FONT_SCALE_SMALL, 1)[0]
    label_y = y1 - 5 if y1 > text_size[1] + 10 else y2 + text_size[1] + 5
    
    cv2.rectangle(img, (x1, label_y - text_size[1] - 4), 
                 (x1 + text_size[0] + 4, label_y + 2), COLOR_BLACK, -1)
    cv2.putText(img, text, (x1 + 2, label_y), FONT, FONT_SCALE_SMALL, 
               COLOR_WHITE, 1)
    
    return img


def draw_perception_results(image: np.ndarray, perception_data,
                           distance: Optional[float] = None) -> np.ndarray:
    """Draw perception pipeline results on image.
    
    Args:
        image: BGR image
        perception_data: PerceptionData with pose, face, emotion, gesture
        distance: Distance to person in meters
    
    Returns:
        Annotated image
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # Draw pose landmarks if available
    if hasattr(perception_data, 'pose_result') and perception_data.pose_result:
        landmarks = np.array(perception_data.pose_result.get('keypoints', []))
        if len(landmarks) > 0:
            for kp in landmarks:
                x, y = int(kp[0]), int(kp[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(img, (x, y), 3, COLOR_BLUE, -1)
    
    # Draw info panel
    info_lines = []
    if hasattr(perception_data, 'pose_label'):
        info_lines.append(f"Pose: {perception_data.pose_label}")
    if hasattr(perception_data, 'face_label'):
        info_lines.append(f"Face: {perception_data.face_label}")
    if hasattr(perception_data, 'emotion_label'):
        info_lines.append(f"Emotion: {perception_data.emotion_label}")
    if hasattr(perception_data, 'gesture_label'):
        info_lines.append(f"Gesture: {perception_data.gesture_label}")
    if distance is not None:
        info_lines.append(f"Distance: {distance:.2f}m")
    
    # Draw info panel in top-left
    y_offset = 30
    for line in info_lines:
        text_size = cv2.getTextSize(line, FONT, FONT_SCALE_SMALL, 1)[0]
        cv2.rectangle(img, (10, y_offset - text_size[1] - 2), 
                     (10 + text_size[0] + 4, y_offset + 2), COLOR_BLACK, -1)
        cv2.putText(img, line, (12, y_offset), FONT, FONT_SCALE_SMALL, 
                   COLOR_WHITE, 1)
        y_offset += text_size[1] + 8
    
    return img


def save_frame(image: np.ndarray, output_dir: str, prefix: str = "frame",
              iteration: Optional[int] = None) -> str:
    """Save annotated frame to disk.
    
    Args:
        image: BGR image to save
        output_dir: Directory to save to
        prefix: Filename prefix
        iteration: Iteration number (optional)
    
    Returns:
        Path to saved file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    if iteration is not None:
        filename = f"{timestamp}_{prefix}_iter{iteration:04d}.jpg"
    else:
        filename = f"{timestamp}_{prefix}.jpg"
    
    filepath = Path(output_dir) / filename
    success = cv2.imwrite(str(filepath), image)
    
    if success:
        logger.debug(f"Saved frame: {filepath}")
        return str(filepath)
    else:
        logger.error(f"Failed to save frame: {filepath}")
        return None


def show_frame(image: np.ndarray, window_name: str = "Friendly Spot",
              wait_key: int = 1) -> int:
    """Display frame in OpenCV window.
    
    Args:
        image: BGR image to display
        window_name: Window title
        wait_key: Milliseconds to wait for key press
    
    Returns:
        Key code pressed (or -1 if timeout)
    """
    cv2.imshow(window_name, image)
    return cv2.waitKey(wait_key)


def visualize_pipeline_frame(image: np.ndarray, 
                             perception_data=None,
                             person_detection=None,
                             show: bool = True,
                             save_dir: Optional[str] = None,
                             iteration: Optional[int] = None) -> Optional[int]:
    """Unified visualization for perception pipeline.
    
    Args:
        image: Input frame
        perception_data: Optional PerceptionData results
        person_detection: Optional PersonDetection with bbox/mask/depth
        show: Display in window
        save_dir: Directory to save frame (None = don't save)
        iteration: Iteration number for filename
    
    Returns:
        Key code if showing window, None otherwise
    """
    annotated = image.copy()
    
    # Draw person detection with mask if available
    if person_detection is not None:
        annotated = draw_detection_with_mask(
            annotated,
            bbox=person_detection.bbox,
            confidence=person_detection.confidence,
            distance=person_detection.distance_m,
            mask=person_detection.mask,
            depth_img=person_detection.depth_frame,
            label="Person"
        )
    
    # Draw perception results
    if perception_data is not None:
        distance = perception_data.distance_m if hasattr(perception_data, 'distance_m') else None
        annotated = draw_perception_results(annotated, perception_data, distance)
    
    # Save if requested
    if save_dir:
        save_frame(annotated, save_dir, prefix="pipeline", iteration=iteration)
    
    # Show if requested
    if show:
        return show_frame(annotated, window_name="Friendly Spot Pipeline")
    
    return None


def close_all_windows():
    """Close all OpenCV windows."""
    cv2.destroyAllWindows()
