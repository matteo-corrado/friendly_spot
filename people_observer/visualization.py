"""Visualization helpers for displaying detections on camera images.

Functions:
- draw_detections(image, detections, camera_name) -> image
    Draw bounding boxes and labels on image
- show_detections_grid(frames_dict, detections_dict, window_name="Detections")
    Display multiple camera views in a grid with detections
"""
import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

from .detection import Detection

logger = logging.getLogger(__name__)

# Visualization constants
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_SCALE_STATS = 0.7
CAMERA_LABEL_PADDING = 10
CONFIDENCE_LABEL_PADDING = 4
DETECTION_COUNT_PADDING = 10
GRID_COLS = 3
DEFAULT_TARGET_WIDTH = 1920
GRID_ASPECT_RATIO = 3.0 / 4.0  # 4:3 for fisheye cameras
STATS_PANEL_HEIGHT = 40
STATS_FONT_THICKNESS = 2
WAIT_KEY_MS = 1  # Minimal wait for key press

# Colors for visualization (BGR format)
COLOR_PERSON = (0, 255, 0)  # Green for person detections
COLOR_TEXT_BG = (0, 0, 0)    # Black background for text
COLOR_TEXT = (255, 255, 255)  # White text


def draw_detections(image: np.ndarray, detections: List[Detection], 
                   camera_name: str = "") -> np.ndarray:
    """Draw bounding boxes and labels on image for all detections.
    
    Args:
        image: BGR image (will be copied, original not modified)
        detections: List of Detection objects
        camera_name: Name to display in corner
        
    Returns:
        Annotated image with bounding boxes drawn
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # Draw camera name in top-left corner
    if camera_name:
        label = camera_name.replace("_fisheye_image", "").replace("_", " ").upper()
        text_size = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)[0]
        cv2.rectangle(img, (5, 5), (CAMERA_LABEL_PADDING + 5 + text_size[0], CAMERA_LABEL_PADDING + 15 + text_size[1]), 
                     COLOR_TEXT_BG, -1)
        cv2.putText(img, label, (CAMERA_LABEL_PADDING, 20 + text_size[1]), FONT, FONT_SCALE, 
                   COLOR_TEXT, THICKNESS)
    
    # Draw each detection
    for det in detections:
        x, y, bbox_w, bbox_h = det.bbox_xywh
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + bbox_w), int(y + bbox_h)
        
        # Clamp to image bounds
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PERSON, THICKNESS)
        
        # Draw confidence label
        conf_label = f"Person {det.conf:.2f}"
        text_size = cv2.getTextSize(conf_label, FONT, FONT_SCALE, 1)[0]
        
        # Position label above box if possible, otherwise below
        if y1 - text_size[1] - 10 > 0:
            text_y = y1 - 5
            rect_y1 = y1 - text_size[1] - 10
            rect_y2 = y1 - 2
        else:
            text_y = y2 + text_size[1] + 5
            rect_y1 = y2 + 2
            rect_y2 = y2 + text_size[1] + 10
        
        # Draw text background
        cv2.rectangle(img, (x1, rect_y1), (x1 + text_size[0] + 2 * CONFIDENCE_LABEL_PADDING, rect_y2),
                     COLOR_TEXT_BG, -1)
        # Draw text
        cv2.putText(img, conf_label, (x1 + CONFIDENCE_LABEL_PADDING, text_y), FONT, FONT_SCALE,
                   COLOR_TEXT, 1)
    
    # Draw detection count in top-right
    count_label = f"Detections: {len(detections)}"
    text_size = cv2.getTextSize(count_label, FONT, FONT_SCALE, THICKNESS)[0]
    cv2.rectangle(img, (w - text_size[0] - DETECTION_COUNT_PADDING - 5, 5), 
                 (w - 5, 25 + text_size[1]), COLOR_TEXT_BG, -1)
    cv2.putText(img, count_label, (w - text_size[0] - DETECTION_COUNT_PADDING, 20 + text_size[1]), 
               FONT, FONT_SCALE, COLOR_TEXT, THICKNESS)
    
    return img


def create_grid_layout(images: List[np.ndarray], cols: int = GRID_COLS, 
                       target_width: int = DEFAULT_TARGET_WIDTH) -> np.ndarray:
    """Arrange multiple images in a grid layout.
    
    Args:
        images: List of BGR images (can be different sizes)
        cols: Number of columns in grid
        target_width: Target width for entire grid
        
    Returns:
        Single image containing grid of input images
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Calculate grid dimensions
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    # Calculate cell size (make all images same size)
    cell_width = target_width // cols
    # Maintain aspect ratio for fisheye cameras
    cell_height = int(cell_width * GRID_ASPECT_RATIO)
    
    # Resize all images to cell size
    resized = []
    for img in images:
        if img.shape[0] > 0 and img.shape[1] > 0:
            resized.append(cv2.resize(img, (cell_width, cell_height)))
        else:
            # Create blank if invalid
            resized.append(np.zeros((cell_height, cell_width, 3), dtype=np.uint8))
    
    # Pad with black images if needed to fill grid
    while len(resized) < rows * cols:
        resized.append(np.zeros((cell_height, cell_width, 3), dtype=np.uint8))
    
    # Arrange in grid
    grid_rows = []
    for r in range(rows):
        row_images = resized[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_images))
    
    return np.vstack(grid_rows)


def show_detections_grid(frames_dict: Dict[str, np.ndarray],
                        detections_dict: Dict[str, List[Detection]],
                        window_name: str = "People Observer - Detections",
                        wait_key: int = WAIT_KEY_MS,
                        target_width: int = DEFAULT_TARGET_WIDTH) -> int:
    """Display all camera frames with detections in a grid layout.
    
    Args:
        frames_dict: Dict of camera_name -> BGR image
        detections_dict: Dict of camera_name -> list of detections
        window_name: OpenCV window name
        wait_key: Milliseconds to wait for key press (1 = minimal wait)
        target_width: Target width for grid display
        
    Returns:
        Key code pressed (or -1 if timeout)
    """
    if not frames_dict:
        logger.warning("No frames to display")
        return -1
    
    # Draw detections on each frame
    annotated = []
    for name in sorted(frames_dict.keys()):
        img = frames_dict[name]
        dets = detections_dict.get(name, [])
        annotated.append(draw_detections(img, dets, name))
    
    # Create grid layout (5 cameras -> GRID_COLS columns)
    grid = create_grid_layout(annotated, cols=GRID_COLS, target_width=target_width)
    
    # Add overall stats at bottom
    total_dets = sum(len(d) for d in detections_dict.values())
    stats = f"Total detections: {total_dets} across {len(frames_dict)} cameras | Press 'q' to quit"
    stats_panel = np.zeros((STATS_PANEL_HEIGHT, grid.shape[1], 3), dtype=np.uint8)
    text_size = cv2.getTextSize(stats, FONT, FONT_SCALE_STATS, STATS_FONT_THICKNESS)[0]
    text_x = (grid.shape[1] - text_size[0]) // 2
    cv2.putText(stats_panel, stats, (text_x, 28), FONT, FONT_SCALE_STATS, COLOR_TEXT, STATS_FONT_THICKNESS)
    
    # Combine grid and stats
    display = np.vstack([grid, stats_panel])
    
    # Show window
    cv2.imshow(window_name, display)
    return cv2.waitKey(wait_key)


def save_annotated_frames(frames_dict: Dict[str, np.ndarray],
                         detections_dict: Dict[str, List[Detection]],
                         output_dir: str,
                         iteration: int):
    """Save annotated frames to disk.
    
    Args:
        frames_dict: Dict of camera_name -> BGR image
        detections_dict: Dict of camera_name -> list of detections
        output_dir: Directory to save images
        iteration: Iteration number for filename
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, img in frames_dict.items():
        dets = detections_dict.get(name, [])
        annotated = draw_detections(img, dets, name)
        filename = f"{output_dir}/iter{iteration:04d}_{name}.jpg"
        cv2.imwrite(filename, annotated)
    
    logger.info(f"Saved {len(frames_dict)} annotated frames to {output_dir}")
