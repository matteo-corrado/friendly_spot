# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/18/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Visualization helpers for displaying detections with bounding boxes, depth annotations,
# image rotation correction, grid layouts, and frame saving with RGB and colorized depth views
# Acknowledgements: OpenCV documentation for drawing functions and image rotation,
# Boston Dynamics SDK get_depth_plus_visual_image.py for depth colorization patterns,
# Claude for rotation coordinate transforms and grid layout design

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

from ..perception.yolo_detector import Detection
import sys
from pathlib import Path

# Import unified_visualization from people_observer package
parent_parent_dir = Path(__file__).parent.parent.parent
people_observer_path = parent_parent_dir / "people_observer"
if str(people_observer_path) not in sys.path:
    sys.path.insert(0, str(people_observer_path))
from unified_visualization import create_depth_colormap, draw_detection_with_mask

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


# Note: create_depth_colormap is now imported from unified_visualization
# Keeping local implementation for backwards compatibility
def create_depth_colormap_legacy(depth_img: np.ndarray, min_dist: float = 0.5, max_dist: float = 5.0) -> np.ndarray:
    """Convert depth image to color-coded visualization.
    
    Args:
        depth_img: Depth in meters (float32, NaN for invalid)
        min_dist: Minimum distance for color scale (meters)
        max_dist: Maximum distance for color scale (meters)
    
    Returns:
        BGR image with depth visualized as heatmap (blue=close, red=far)
    """
    # Normalize depth to 0-1 range
    depth_norm = np.copy(depth_img)
    valid_mask = ~np.isnan(depth_norm)
    
    # Clamp to range
    depth_norm[valid_mask] = np.clip(depth_norm[valid_mask], min_dist, max_dist)
    depth_norm[valid_mask] = (depth_norm[valid_mask] - min_dist) / (max_dist - min_dist)
    
    # Convert to 8-bit
    depth_u8 = np.zeros_like(depth_norm, dtype=np.uint8)
    depth_u8[valid_mask] = (depth_norm[valid_mask] * 255).astype(np.uint8)
    
    # Apply COLORMAP_JET (blue=close, red=far)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    
    # Set invalid pixels to black
    colored[~valid_mask] = [0, 0, 0]
    
    return colored


def draw_detections(image: np.ndarray, detections: List[Detection], 
                   camera_name: str = "", depth_img: Optional[np.ndarray] = None) -> np.ndarray:
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
        
        # Draw segmentation mask overlay if available
        if det.mask is not None:
            mask = det.mask
            # Resize mask to image size if needed
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Create overlay
            overlay = img.copy()
            
            # If depth is available, use depth colormap
            if depth_img is not None and depth_img.shape[:2] == (h, w):
                # Create depth colormap
                depth_colored = create_depth_colormap(depth_img)
                # Apply mask to depth colormap
                overlay[mask] = depth_colored[mask]
            else:
                # No depth, use solid color
                overlay[mask] = COLOR_MASK_OVERLAY
            
            # Blend with original image
            img = cv2.addWeighted(overlay, MASK_ALPHA, img, 1 - MASK_ALPHA, 0)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PERSON, THICKNESS)
        
        # Draw confidence label with distance if available
        if depth_img is not None and det.mask is not None:
            # Extract depth from mask
            from .tracker import estimate_detection_depth_m
            dist = estimate_detection_depth_m(depth_img, det, use_mask=True)
            if dist is not None:
                conf_label = f"Person {det.conf:.2f} | {dist:.2f}m"
            else:
                conf_label = f"Person {det.conf:.2f}"
        else:
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
                        depth_dict: Optional[Dict[str, np.ndarray]] = None,
                        window_name: str = "People Observer - Detections",
                        wait_key: int = WAIT_KEY_MS,
                        target_width: int = DEFAULT_TARGET_WIDTH) -> int:
    """Display all camera frames with detections in a grid layout.
    
    Args:
        frames_dict: Dict of camera_name -> BGR image
        detections_dict: Dict of camera_name -> list of detections
        depth_dict: Optional dict of camera_name -> depth image (float32, meters)
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
        depth = depth_dict.get(name) if depth_dict else None
        annotated.append(draw_detections(img, dets, name, depth))
    
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
                         iteration: int,
                         depth_dict: Optional[Dict[str, np.ndarray]] = None):
    """Save annotated frames to disk.
    
    Args:
        frames_dict: Dict of camera_name -> BGR image
        detections_dict: Dict of camera_name -> list of detections
        output_dir: Directory to save images
        iteration: Iteration number for filename
        depth_dict: Optional dict of camera_name -> depth image (float32, meters)
    """
    import os
    
    try:
        abs_output_dir = os.path.abspath(output_dir)
        logger.info(f"save_annotated_frames called: output_dir={abs_output_dir}, iteration={iteration}")
        logger.info(f"Frames to save: {list(frames_dict.keys())}")
        
        # Create directory if doesn't exist
        os.makedirs(abs_output_dir, exist_ok=True)
        
        # Add timestamp to make filenames unique
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
        
        saved_count = 0
        failed_files = []
        
        for name, img in frames_dict.items():
            if img is None or img.size == 0:
                logger.warning(f"Skipping {name}: empty or None image")
                continue
            
            dets = detections_dict.get(name, [])
            depth = depth_dict.get(name) if depth_dict else None
            annotated = draw_detections(img, dets, name, depth)
            
            # Filename format: timestamp_iter####_cameraname.jpg (always unique)
            filename = f"{timestamp}_iter{iteration:04d}_{name}.jpg"
            filepath = os.path.join(abs_output_dir, filename)
            
            success = cv2.imwrite(filepath, annotated)
            if success and os.path.exists(filepath):
                saved_count += 1
                file_size = os.path.getsize(filepath)
                logger.debug(f"  [OK] {filename} ({file_size:,} bytes)")
            else:
                failed_files.append(filename)
                logger.error(f"  Failed to write {filename}")
        
        if saved_count == len(frames_dict):
            logger.info(f"Saved all {saved_count} frames to {abs_output_dir}")
        else:
            logger.warning(f"Saved {saved_count}/{len(frames_dict)} frames (failed: {failed_files})")
    except Exception as e:
        logger.error(f"Error saving annotated frames: {e}", exc_info=True)
