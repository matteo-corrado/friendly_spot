# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Unified visualization overlay providing consistent rendering across modules with depth masks,
# detection bounding boxes, pose landmarks, and perception result annotations
# Acknowledgements: OpenCV for drawing functions, Claude for unified visualization design

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


def draw_emotion_bars(image: np.ndarray, emotion_scores: dict, 
                      x: int, y: int, width: int = 150, height: int = 100) -> None:
    """Draw emotion scores as horizontal bars.
    
    Args:
        image: Image to draw on (modified in-place)
        emotion_scores: Dict mapping emotion names to confidence scores (0-100)
        x, y: Top-left position for bar chart
        width: Width of bars
        height: Total height of chart
    """
    if not emotion_scores:
        return
    
    # Sort emotions by score
    emotions = sorted(emotion_scores.items(), key=lambda kv: kv[1], reverse=True)
    num_emotions = len(emotions)
    bar_height = height // num_emotions
    
    # Background
    cv2.rectangle(image, (x, y), (x + width, y + height), COLOR_BLACK, -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), COLOR_WHITE, 1)
    
    # Draw bars
    for i, (emotion, score) in enumerate(emotions):
        bar_y = y + i * bar_height
        bar_w = int((score / 100.0) * (width - 60))
        
        # Bar
        cv2.rectangle(image, (x + 5, bar_y + 3), 
                     (x + 5 + bar_w, bar_y + bar_height - 3), COLOR_BLUE, -1)
        
        # Label
        label = f"{emotion[:6]}: {score:.0f}%"
        cv2.putText(image, label, (x + 8, bar_y + bar_height - 8), 
                   FONT, 0.3, COLOR_WHITE, 1)


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
                           distance: Optional[float] = None,
                           desired_behavior: Optional[str] = None,
                           comfort_score: Optional[float] = None) -> np.ndarray:
    """Draw perception pipeline results on image.
    
    Args:
        image: BGR image
        perception_data: PerceptionInput with pose, face, emotion, gesture
        distance: Distance to person in meters
        desired_behavior: Desired robot behavior/action to display
        comfort_score: Calculated comfort score (0-1)
    
    Returns:
        Annotated image
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # Draw face bbox if available
    if hasattr(perception_data, 'face_bbox') and perception_data.face_bbox is not None:
        x, y, fw, fh = perception_data.face_bbox
        # Draw green rectangle around face
        cv2.rectangle(img, (x, y), (x + fw, y + fh), COLOR_GREEN, 2)
        
        # Draw emotion label above face bbox
        if hasattr(perception_data, 'emotion_label') and perception_data.emotion_label:
            emotion_text = perception_data.emotion_label.upper()
            text_size = cv2.getTextSize(emotion_text, FONT, FONT_SCALE_SMALL, 2)[0]
            text_x = x
            text_y = y - 10
            # Background for text
            cv2.rectangle(img, (text_x, text_y - text_size[1] - 4),
                         (text_x + text_size[0] + 4, text_y + 2), COLOR_BLACK, -1)
            cv2.putText(img, emotion_text, (text_x + 2, text_y), FONT, 
                       FONT_SCALE_SMALL, COLOR_GREEN, 2)
        
        # Draw emotion score bars below face if available
        if hasattr(perception_data, 'emotion_scores') and perception_data.emotion_scores:
            bar_x = x
            bar_y = y + fh + 10
            # Ensure bars don't go off screen
            if bar_y + 100 < h:
                draw_emotion_bars(img, perception_data.emotion_scores, bar_x, bar_y)
    
    # Draw pose landmarks if available
    if hasattr(perception_data, 'pose_landmarks') and perception_data.pose_landmarks is not None:
        landmarks = perception_data.pose_landmarks
        if len(landmarks) > 0:
            # Draw keypoints
            for kp in landmarks:
                if len(kp) >= 2:  # Has x, y
                    x_kp = int(kp[0] * w) if kp[0] <= 1.0 else int(kp[0])
                    y_kp = int(kp[1] * h) if kp[1] <= 1.0 else int(kp[1])
                    if 0 <= x_kp < w and 0 <= y_kp < h:
                        cv2.circle(img, (x_kp, y_kp), 4, COLOR_BLUE, -1)
                        cv2.circle(img, (x_kp, y_kp), 5, COLOR_WHITE, 1)
    
    # Draw gesture indicator if not "none"
    if hasattr(perception_data, 'gesture_label') and perception_data.gesture_label and perception_data.gesture_label != 'none':
        gesture_text = f"{perception_data.gesture_label.upper()}"
        text_size = cv2.getTextSize(gesture_text, FONT, FONT_SCALE, 2)[0]
        # Position in top-right corner
        text_x = w - text_size[0] - 20
        text_y = 40
        cv2.rectangle(img, (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5), COLOR_YELLOW, -1)
        cv2.putText(img, gesture_text, (text_x, text_y), FONT, FONT_SCALE, 
                   COLOR_BLACK, 2)
    
    # Draw info panel in top-left
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
    if comfort_score is not None:
        info_lines.append(f"Comfort: {comfort_score:.2f}")
    if desired_behavior is not None:
        info_lines.append(f"Action: {desired_behavior}")
    
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
                             desired_behavior=None,
                             comfort_score=None,
                             show: bool = True,
                             save_dir: Optional[str] = None,
                             iteration: Optional[int] = None) -> Optional[int]:
    """Unified visualization for perception pipeline.
    
    Args:
        image: Input frame (PTZ camera image)
        perception_data: Optional PerceptionData results
        person_detection: Optional PersonDetection with bbox/mask/depth (from surround cameras)
        desired_behavior: Desired robot behavior/action to display
        comfort_score: Calculated comfort score (0-1)
        show: Display in window
        save_dir: Directory to save frame (None = don't save)
        iteration: Iteration number for filename
    
    Returns:
        Key code if showing window, None otherwise
    
    Note: PersonDetection bbox/mask from surround cameras are NOT drawn on PTZ image,
          since they were detected in a different camera frame. Only perception results
          (face bbox, pose, emotion, etc.) from PTZ image are drawn.
    """
    annotated = image.copy()
    
    # DO NOT draw person detection bbox/mask from surround cameras on PTZ image
    # The detection came from a different camera frame and coordinates don't align
    
    # Draw perception results (face bbox, landmarks, pose, emotion, gesture from PTZ)
    if perception_data is not None:
        distance = perception_data.distance_m if hasattr(perception_data, 'distance_m') else None
        annotated = draw_perception_results(annotated, perception_data, distance, desired_behavior, comfort_score)
    
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
