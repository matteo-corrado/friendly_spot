"""Visualization helpers for displaying detections on camera images with depth support.

Functions:
- draw_detections(image, detections, camera_name) -> image
    Draw bounding boxes and labels on image
- draw_depth_image(depth_img, detections, camera_name, distances) -> image
    Colorize depth image with detections and distance annotations
- draw_detections_with_depth(image, depth_img, detections, camera_name, distances) -> image
    Draw detections with depth values overlaid on RGB image
- show_detections_grid(frames_dict, detections_dict, window_name="Detections")
    Display multiple camera views in a grid with detections
- show_detections_with_depth_grid(frames_dict, depth_frames_dict, detections_dict, distances_dict)
    Display RGB and depth views side-by-side with distance annotations
"""
import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

from .detection import Detection
from .config import ROTATION_ANGLE

logger = logging.getLogger(__name__)


def rotate_image_for_display(image: np.ndarray, camera_name: str) -> np.ndarray:
    """Rotate image to correct orientation for display based on camera's physical rotation.
    
    Spot's fisheye cameras are physically rotated relative to their mounting frames:
    - frontleft: -78° → rotate image 90° clockwise to approximate upright
    - frontright: -102° → rotate image 90° clockwise to approximate upright  
    - right: 180° → rotate image 180°
    - left, back: 0° → no rotation needed
    
    Args:
        image: BGR image from camera
        camera_name: Camera source name (e.g., "frontleft_fisheye_image")
        
    Returns:
        Rotated image for proper display orientation
    """
    rotation_angle = ROTATION_ANGLE.get(camera_name, 0.0)
    
    # Map rotation angles to OpenCV rotation codes
    # For display purposes, we use 90° increments (nearest upright orientation)
    if abs(rotation_angle - 180.0) < 45:  # right camera (180°)
        return cv2.rotate(image, cv2.ROTATE_180)
    elif abs(rotation_angle + 78.0) < 45:  # frontleft (-78°) → rotate 90° CW
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif abs(rotation_angle + 102.0) < 45:  # frontright (-102°) → rotate 90° CW
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:  # left (0°), back (0°) → no rotation
        return image

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
COLOR_DEPTH_MARKER = (0, 255, 255)  # Cyan for depth center markers
COLOR_DISTANCE_LABEL = (255, 255, 0)  # Yellow for distance text

# Depth visualization settings
DEPTH_COLORMAP = cv2.COLORMAP_JET  # Colormap for depth visualization
DEPTH_MIN_PERCENTILE = 5  # Filter out closest 5% for better contrast
DEPTH_MAX_PERCENTILE = 95  # Filter out furthest 5% for better contrast


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
    # Rotate image to correct orientation first
    img = rotate_image_for_display(image.copy(), camera_name)
    h, w = img.shape[:2]
    orig_h, orig_w = image.shape[:2]
    
    # Determine rotation applied for bbox coordinate adjustment
    rotation_angle = ROTATION_ANGLE.get(camera_name, 0.0)
    is_90cw = abs(rotation_angle + 78.0) < 45 or abs(rotation_angle + 102.0) < 45
    is_180 = abs(rotation_angle - 180.0) < 45
    
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
        
        # Transform bbox coordinates based on rotation
        if is_90cw:  # 90° clockwise: (x,y) → (y, orig_w-x-w)
            x1_rot = int(y)
            y1_rot = int(orig_w - x - bbox_w)
            x2_rot = int(y + bbox_h)
            y2_rot = int(orig_w - x)
            x1, y1, x2, y2 = x1_rot, y1_rot, x2_rot, y2_rot
        elif is_180:  # 180°: (x,y) → (orig_w-x-w, orig_h-y-h)
            x1 = int(orig_w - x - bbox_w)
            y1 = int(orig_h - y - bbox_h)
            x2 = int(orig_w - x)
            y2 = int(orig_h - y)
        else:  # No rotation
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


def draw_depth_image(depth_img: Optional[np.ndarray], 
                    detections: List[Detection],
                    camera_name: str = "",
                    distances: Optional[List[Optional[float]]] = None) -> np.ndarray:
    """Convert depth image to colorized visualization with detection annotations.
    
    Uses SDK pattern: min/max scaling for 8-bit conversion before colormap application.
    
    Args:
        depth_img: Depth image in meters (np.ndarray with NaN for invalid pixels)
        detections: List of Detection objects
        camera_name: Name to display in corner
        distances: Optional list of distances (meters) indexed same as detections, None values for no depth
        
    Returns:
        BGR colorized depth visualization with annotations
    """
    if depth_img is None:
        # Return black image if no depth
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Get valid (non-NaN) depth values for scaling
    valid_depth = depth_img[~np.isnan(depth_img)]
    
    if len(valid_depth) == 0:
        # All invalid - return black
        return np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
    
    # SDK pattern: use min/max scaling (from get_depth_plus_visual_image.py)
    # This provides better visualization than percentile-based scaling
    min_val = np.min(valid_depth)
    max_val = np.max(valid_depth)
    depth_range = max_val - min_val
    
    if depth_range == 0:
        depth_range = 1.0  # Avoid division by zero
    
    # Normalize to 0-255 range using min/max scaling
    depth_normalized = np.copy(depth_img)
    depth_normalized = (255.0 / depth_range * (depth_normalized - min_val)).astype(np.uint8)
    
    # Set invalid pixels to 0 (black)
    depth_normalized[np.isnan(depth_img)] = 0
    
    # Convert to RGB for colormap
    depth_normalized_rgb = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)
    
    # Apply colormap (SDK uses COLORMAP_JET)
    depth_colorized = cv2.applyColorMap(depth_normalized_rgb, DEPTH_COLORMAP)
    
    # Rotate to correct orientation
    depth_colorized = rotate_image_for_display(depth_colorized, camera_name)
    
    # Draw camera name
    h, w = depth_colorized.shape[:2]
    orig_h, orig_w = depth_img.shape[:2]
    
    # Determine rotation applied for bbox coordinate adjustment
    rotation_angle = ROTATION_ANGLE.get(camera_name, 0.0)
    is_90cw = abs(rotation_angle + 78.0) < 45 or abs(rotation_angle + 102.0) < 45
    is_180 = abs(rotation_angle - 180.0) < 45
    if camera_name:
        label = camera_name.replace("_fisheye_image", "").replace("_", " ").upper() + " (DEPTH)"
        text_size = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)[0]
        cv2.rectangle(depth_colorized, (5, 5), (CAMERA_LABEL_PADDING + 5 + text_size[0], CAMERA_LABEL_PADDING + 15 + text_size[1]), 
                     COLOR_TEXT_BG, -1)
        cv2.putText(depth_colorized, label, (CAMERA_LABEL_PADDING, 20 + text_size[1]), FONT, FONT_SCALE, 
                   COLOR_TEXT, THICKNESS)
    
    # Draw detections with distance annotations
    for idx, det in enumerate(detections):
        x, y, bbox_w, bbox_h = det.bbox_xywh
        
        # Transform bbox coordinates based on rotation
        if is_90cw:  # 90° clockwise: (x,y) → (y, orig_w-x-w)
            x1_rot = int(y)
            y1_rot = int(orig_w - x - bbox_w)
            x2_rot = int(y + bbox_h)
            y2_rot = int(orig_w - x)
            x1, y1, x2, y2 = x1_rot, y1_rot, x2_rot, y2_rot
        elif is_180:  # 180°: (x,y) → (orig_w-x-w, orig_h-y-h)
            x1 = int(orig_w - x - bbox_w)
            y1 = int(orig_h - y - bbox_h)
            x2 = int(orig_w - x)
            y2 = int(orig_h - y)
        else:  # No rotation
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + bbox_w), int(y + bbox_h)
        
        # Clamp to image bounds
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        
        # Draw bounding box
        cv2.rectangle(depth_colorized, (x1, y1), (x2, y2), COLOR_PERSON, THICKNESS)
        
        # Mark center of detection
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(depth_colorized, (center_x, center_y), 5, COLOR_DEPTH_MARKER, -1)
        cv2.circle(depth_colorized, (center_x, center_y), 6, COLOR_PERSON, 2)
        
        # Draw distance if available
        if distances and idx < len(distances) and distances[idx] is not None:
            dist = distances[idx]
            dist_label = f"{dist:.2f}m"
            
            # Large distance label at center
            text_size = cv2.getTextSize(dist_label, FONT, FONT_SCALE * 1.5, 2)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y - 10
            
            # Draw text background
            cv2.rectangle(depth_colorized, 
                         (text_x - 4, text_y - text_size[1] - 4),
                         (text_x + text_size[0] + 4, text_y + 4),
                         COLOR_TEXT_BG, -1)
            # Draw distance text
            cv2.putText(depth_colorized, dist_label, (text_x, text_y), 
                       FONT, FONT_SCALE * 1.5, COLOR_DISTANCE_LABEL, 2)
    
    # Add depth range info in top-right
    range_label = f"Range: {min_val:.1f}-{max_val:.1f}m"
    text_size = cv2.getTextSize(range_label, FONT, FONT_SCALE, THICKNESS)[0]
    cv2.rectangle(depth_colorized, (w - text_size[0] - DETECTION_COUNT_PADDING - 5, 5), 
                 (w - 5, 25 + text_size[1]), COLOR_TEXT_BG, -1)
    cv2.putText(depth_colorized, range_label, (w - text_size[0] - DETECTION_COUNT_PADDING, 20 + text_size[1]), 
               FONT, FONT_SCALE, COLOR_TEXT, THICKNESS)
    
    return depth_colorized


def draw_detections_with_depth(image: np.ndarray,
                               depth_img: Optional[np.ndarray],
                               detections: List[Detection],
                               camera_name: str = "",
                               distances: Optional[List[Optional[float]]] = None) -> np.ndarray:
    """Draw detections with depth values overlaid on RGB image.
    
    Args:
        image: BGR image
        depth_img: Depth image in meters (np.ndarray with NaN for invalid pixels)
        detections: List of Detection objects
        camera_name: Name to display in corner
        distances: Optional list of distances (meters) indexed same as detections, None values for no depth
        
    Returns:
        Annotated RGB image with depth information
    """
    img = draw_detections(image, detections, camera_name)
    h, w = img.shape[:2]
    orig_h, orig_w = image.shape[:2]
    
    # Determine rotation for bbox coordinate adjustment
    rotation_angle = ROTATION_ANGLE.get(camera_name, 0.0)
    is_90cw = abs(rotation_angle + 78.0) < 45 or abs(rotation_angle + 102.0) < 45
    is_180 = abs(rotation_angle - 180.0) < 45
    
    # Overlay distance information on detections
    for idx, det in enumerate(detections):
        if distances and idx < len(distances) and distances[idx] is not None:
            x, y, bbox_w, bbox_h = det.bbox_xywh
            
            # Transform bbox coordinates based on rotation (need to work with rotated coords)
            if is_90cw:
                # Image was rotated 90° CW, detections drawn with transformed coords
                # Use the same transformation
                center_x = int(y + bbox_h / 2)
                center_y = int(orig_w - x - bbox_w / 2)
            elif is_180:
                center_x = int(orig_w - x - bbox_w / 2)
                center_y = int(orig_h - y - bbox_h / 2)
            else:
                center_x = int(x + bbox_w / 2)
                center_y = int(y + bbox_h / 2)
            
            dist = distances[idx]
            dist_label = f"{dist:.2f}m"
            
            # Draw distance label at center
            text_size = cv2.getTextSize(dist_label, FONT, FONT_SCALE * 1.2, 2)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y
            
            # Draw text background
            cv2.rectangle(img,
                         (text_x - 4, text_y - text_size[1] - 4),
                         (text_x + text_size[0] + 4, text_y + 4),
                         COLOR_TEXT_BG, -1)
            # Draw distance text in yellow
            cv2.putText(img, dist_label, (text_x, text_y),
                       FONT, FONT_SCALE * 1.2, COLOR_DISTANCE_LABEL, 2)
            
            # Draw center marker
            cv2.circle(img, (center_x, center_y), 4, COLOR_DEPTH_MARKER, -1)
            cv2.circle(img, (center_x, center_y), 5, COLOR_PERSON, 1)
    
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


def show_detections_with_depth_grid(frames_dict: Dict[str, np.ndarray],
                                    depth_frames_dict: Dict[str, np.ndarray],
                                    detections_dict: Dict[str, List[Detection]],
                                    distances_dict: Optional[Dict[str, List[Optional[float]]]] = None,
                                    window_name: str = "People Observer - RGB + Depth",
                                    wait_key: int = WAIT_KEY_MS,
                                    target_width: int = DEFAULT_TARGET_WIDTH) -> int:
    """Display RGB and depth views side-by-side with distance annotations.
    
    Args:
        frames_dict: Dict of camera_name -> BGR image
        depth_frames_dict: Dict of camera_name -> depth image (meters, NaN for invalid)
        detections_dict: Dict of camera_name -> list of detections
        distances_dict: Optional dict of camera_name -> list of distances (indexed same as detections)
        window_name: OpenCV window name
        wait_key: Milliseconds to wait for key press
        target_width: Target width for grid display
        
    Returns:
        Key code pressed (or -1 if timeout)
    """
    if not frames_dict:
        logger.warning("No frames to display")
        return -1
    
    # Create side-by-side pairs for each camera
    annotated_pairs = []
    
    for name in sorted(frames_dict.keys()):
        rgb_img = frames_dict[name]
        depth_img = depth_frames_dict.get(name)
        dets = detections_dict.get(name, [])
        distances = distances_dict.get(name, []) if distances_dict else []
        
        # Draw RGB with depth annotations
        rgb_annotated = draw_detections_with_depth(rgb_img, depth_img, dets, name, distances)
        
        # Draw colorized depth
        depth_annotated = draw_depth_image(depth_img, dets, name, distances)
        
        # Ensure both images are same size
        if rgb_annotated.shape[:2] != depth_annotated.shape[:2]:
            depth_annotated = cv2.resize(depth_annotated, 
                                        (rgb_annotated.shape[1], rgb_annotated.shape[0]))
        
        # Concatenate horizontally (RGB left, depth right)
        pair = np.hstack([rgb_annotated, depth_annotated])
        annotated_pairs.append(pair)
    
    # Create grid layout - each row shows one camera's RGB+Depth pair
    # Use cols=1 so each pair gets its own row
    grid = create_grid_layout(annotated_pairs, cols=2, target_width=target_width)
    
    # Add overall stats at bottom
    total_dets = sum(len(d) for d in detections_dict.values())
    cameras_with_depth = sum(1 for name in frames_dict.keys() if name in depth_frames_dict and depth_frames_dict[name] is not None)
    stats = f"Total detections: {total_dets} | Cameras: {len(frames_dict)} ({cameras_with_depth} with depth) | Press 'q' to quit"
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
                         depth_frames_dict: Optional[Dict[str, np.ndarray]] = None,
                         distances_dict: Optional[Dict[str, List[Optional[float]]]] = None):
    """Save annotated RGB and depth frames to disk.
    
    Args:
        frames_dict: Dict of camera_name -> BGR image
        detections_dict: Dict of camera_name -> list of detections
        output_dir: Directory to save images
        iteration: Iteration number for filename
        depth_frames_dict: Optional dict of camera_name -> depth image (meters, NaN for invalid)
        distances_dict: Optional dict of camera_name -> list of distances (indexed same as detections)
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
            
            # Save annotated RGB image
            if depth_frames_dict and distances_dict:
                # Draw with depth annotations if available
                depth_img = depth_frames_dict.get(name)
                distances = distances_dict.get(name, [])
                annotated_rgb = draw_detections_with_depth(img, depth_img, dets, name, distances)
            else:
                # Draw without depth annotations
                annotated_rgb = draw_detections(img, dets, name)
            
            # Filename format: timestamp_iter####_cameraname_rgb.jpg
            rgb_filename = f"{timestamp}_iter{iteration:04d}_{name}_rgb.jpg"
            rgb_filepath = os.path.join(abs_output_dir, rgb_filename)
            
            success = cv2.imwrite(rgb_filepath, annotated_rgb)
            if success and os.path.exists(rgb_filepath):
                saved_count += 1
                file_size = os.path.getsize(rgb_filepath)
                logger.debug(f"{rgb_filename} ({file_size:,} bytes)")
            else:
                failed_files.append(rgb_filename)
                logger.error(f"Failed to write {rgb_filename}")
            
            # Save colorized depth image if available
            if depth_frames_dict and name in depth_frames_dict:
                depth_img = depth_frames_dict[name]
                distances = distances_dict.get(name, []) if distances_dict else []
                depth_colorized = draw_depth_image(depth_img, dets, name, distances)
                
                # Filename format: timestamp_iter####_cameraname_depth.jpg
                depth_filename = f"{timestamp}_iter{iteration:04d}_{name}_depth.jpg"
                depth_filepath = os.path.join(abs_output_dir, depth_filename)
                
                success = cv2.imwrite(depth_filepath, depth_colorized)
                if success and os.path.exists(depth_filepath):
                    saved_count += 1
                    file_size = os.path.getsize(depth_filepath)
                    logger.debug(f"{depth_filename} ({file_size:,} bytes)")
                else:
                    failed_files.append(depth_filename)
                    logger.error(f"Failed to write {depth_filename}")
        
        total_expected = len(frames_dict) + (len(depth_frames_dict) if depth_frames_dict else 0)
        if saved_count == total_expected:
            logger.info(f"Saved all {saved_count} images (RGB + depth) to {abs_output_dir}")
        else:
            logger.warning(f"Saved {saved_count}/{total_expected} images (failed: {failed_files})")
    except Exception as e:
        logger.error(f"Error saving annotated frames: {e}", exc_info=True)
