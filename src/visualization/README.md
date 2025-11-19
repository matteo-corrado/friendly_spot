# Visualization Module

Overlay rendering and visualization helpers for detection and tracking display.

## Overview

The visualization module provides real-time overlay rendering for detections, tracking boxes, depth information, and PTZ indicators. Supports multi-camera grid layouts and customizable styling for debugging and demonstration.

## Components

### Overlay System
- **`overlay.py`**: Main visualization coordinator
  - `UnifiedVisualization`: Manages multi-camera grid display
  - Renders detections, tracking boxes, confidence scores
  - Color-coded by detection class or tracking status
  - Configurable grid layouts (1x1, 2x2, 3x2, etc.)

### Helper Functions
- **`helpers.py`**: Low-level drawing utilities
  - `draw_detections()`: Render bounding boxes with labels
  - `draw_depth_info()`: Overlay depth values on detections
  - `draw_ptz_indicator()`: Show current PTZ angles
  - `show_detections_grid()`: Multi-camera grid view

## Usage

### Basic Detection Overlay

```python
from src.visualization import UnifiedVisualization
from src.perception import YoloDetector
import cv2

# Initialize
viz = UnifiedVisualization()
detector = YoloDetector()

while True:
    # Get frame and detections
    frame = camera.get_frame()
    detections = detector.predict_batch([frame])[0]
    
    # Render overlay
    viz_frame = viz.render(frame, detections)
    
    # Display
    cv2.imshow("Detections", viz_frame)
    if cv2.waitKey(1) == ord('q'):
        break
```

### Multi-Camera Grid

```python
from src.visualization.helpers import show_detections_grid

# Multiple cameras with detections
frames = [cam1.get_frame(), cam2.get_frame(), cam3.get_frame()]
detections = [dets1, dets2, dets3]
sources = ["front-left", "front-right", "left"]

# Display 2x2 grid (3 cameras + empty slot)
show_detections_grid(
    frames=frames,
    detections=detections,
    source_names=sources,
    grid_cols=2
)
```

### Depth Visualization

```python
from src.visualization.helpers import draw_depth_info

# Render frame with depth overlay
frame_bgr = camera.get_frame()
depth_map = depth_camera.get_frame()

for det in detections:
    depth_m = estimate_depth(depth_map, det)
    frame_bgr = draw_depth_info(
        frame_bgr,
        det.bbox_xywh,
        depth_m,
        color=(0, 255, 0)
    )
```

### PTZ Angle Indicator

```python
from src.visualization.helpers import draw_ptz_indicator

# Show current PTZ angles on frame
frame = ptz_camera.get_frame()
frame = draw_ptz_indicator(
    frame,
    pan_deg=45.0,
    tilt_deg=-20.0,
    zoom=2.0
)
```

### Custom Color Schemes

```python
# Color by detection confidence
def confidence_color(conf: float):
    """Green (high conf) → Yellow → Red (low conf)"""
    if conf > 0.8:
        return (0, 255, 0)  # Green
    elif conf > 0.5:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 0, 255)  # Red

# Render with custom colors
viz = UnifiedVisualization(color_fn=confidence_color)
```

## Overlay Elements

### Bounding Boxes
- **Rectangle**: Colored box around detection
- **Label**: Class name + confidence ("person 0.87")
- **ID**: Tracking ID if available

### Depth Information
- **Distance text**: Meters from camera ("2.3m")
- **Color coding**: Green (near) → Yellow → Red (far)
- **Mask overlay**: Semi-transparent segmentation mask

### PTZ Indicators
- **Crosshair**: Center of PTZ view
- **Angle text**: Pan/tilt/zoom values
- **Direction arrow**: Shows pan direction

### Status Info
- **FPS**: Frame rate in top-left
- **Detection count**: "3 detections" in top-right
- **Camera name**: Source identifier

## Configuration

### Grid Layout

Control multi-camera arrangement:

```python
# 2x2 grid (4 cameras max)
show_detections_grid(frames, detections, sources, grid_cols=2)

# 3x2 grid (6 cameras)
show_detections_grid(frames, detections, sources, grid_cols=3)

# Horizontal strip (1 row, N cameras)
show_detections_grid(frames, detections, sources, grid_cols=len(frames))
```

### Box Thickness

Adjust bounding box line width:

```python
# In helpers.py
BOX_THICKNESS = 3  # Pixels (default: 2)
```

### Label Position

Control label placement:

```python
# Options: "top", "bottom", "inside", "outside"
LABEL_POSITION = "top"  # Above bounding box (default)
```

### Font Settings

Customize text rendering:

```python
# In helpers.py
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White
BG_COLOR = (0, 0, 0)  # Black background
```

## Color Schemes

### Default (Detection Class)

```python
CLASS_COLORS = {
    "person": (0, 255, 0),      # Green
    "dog": (255, 0, 0),         # Blue
    "cat": (0, 128, 255),       # Orange
    # ... more classes
}
```

### Tracking Status

```python
TRACKING_COLORS = {
    "active": (0, 255, 0),      # Green (currently tracked)
    "lost": (0, 0, 255),        # Red (tracking lost)
    "tentative": (0, 255, 255), # Yellow (new detection)
}
```

### Depth Gradient

```python
def depth_to_color(depth_m: float) -> tuple:
    """
    Map depth to color gradient:
    - 0-2m: Green (close)
    - 2-5m: Yellow (medium)
    - 5+m: Red (far)
    """
    if depth_m < 2.0:
        return (0, 255, 0)
    elif depth_m < 5.0:
        ratio = (depth_m - 2.0) / 3.0
        green = int(255 * (1 - ratio))
        return (0, green, 255)
    else:
        return (0, 0, 255)
```

## Performance Optimization

### Reduce Overlay Complexity

Disable expensive visualizations:

```python
viz = UnifiedVisualization(
    show_masks=False,        # Skip segmentation mask overlay (slow)
    show_depth=False,        # Skip depth text
    show_confidence=False    # Skip confidence scores
)
```

### Downscale Visualization

Render at lower resolution:

```python
# Render at half resolution for faster display
frame_small = cv2.resize(frame, (640, 480))
viz_frame = viz.render(frame_small, detections)
cv2.imshow("Viz", viz_frame)
```

### Limit Frame Rate

Throttle visualization FPS:

```python
import time

DISPLAY_FPS = 15  # Display at 15 FPS even if processing faster
frame_interval = 1.0 / DISPLAY_FPS

while True:
    start = time.time()
    
    # Process and visualize
    frame = camera.get_frame()
    viz_frame = viz.render(frame, detections)
    cv2.imshow("Viz", viz_frame)
    
    # Throttle
    elapsed = time.time() - start
    if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)
```

### Use Hardware Acceleration

Enable GPU rendering (if available):

```python
# OpenCV CUDA module (requires opencv-contrib-python with CUDA)
import cv2.cuda as cuda

gpu_frame = cuda.GpuMat()
gpu_frame.upload(frame)
# ... GPU operations
frame = gpu_frame.download()
```

## Troubleshooting

### Bounding Boxes Not Visible
- **Wrong color space**: Ensure frame is BGR (not RGB)
- **Off-screen coords**: Check bbox values are within image bounds
- **Zero thickness**: Increase `BOX_THICKNESS` parameter

### Text Unreadable
- **Too small**: Increase `FONT_SCALE`
- **Low contrast**: Add background rectangle behind text
- **Wrong encoding**: Use ASCII labels (avoid Unicode unless font supports)

### Slow Rendering
- **Too many detections**: Limit overlay to top N detections
- **Large images**: Downscale before visualization
- **Complex masks**: Disable mask overlay or reduce alpha blending

### Grid Layout Broken
- **Mismatched counts**: Ensure `len(frames) == len(detections) == len(source_names)`
- **Invalid grid_cols**: Use divisor of total camera count for clean grid
- **Resolution mismatch**: Resize all frames to same size before grid

## Advanced Usage

### Custom Visualization Elements

Add new overlay types:

```python
def draw_trajectory(frame, past_positions, color=(0, 255, 255)):
    """Draw motion trajectory line."""
    if len(past_positions) < 2:
        return frame
    
    points = np.array(past_positions, dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=False, 
                  color=color, thickness=2)
    return frame
```

### Conditional Overlays

Show different info based on state:

```python
viz_frame = frame.copy()

if mode == "tracking":
    viz_frame = draw_tracking_overlay(viz_frame, tracker_state)
elif mode == "debug":
    viz_frame = draw_debug_info(viz_frame, detections, fps)
elif mode == "demo":
    viz_frame = draw_minimal_overlay(viz_frame)  # Clean for demos
```

### Recording Visualizations

Save annotated video:

```python
import cv2

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1920, 1080))

while True:
    viz_frame = viz.render(frame, detections)
    out.write(viz_frame)
    cv2.imshow("Recording", viz_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

out.release()
```

## Dependencies

- `opencv-python` >= 4.8.0: Drawing primitives and display
- `numpy` >= 1.24.0: Array operations

## References

- [OpenCV Drawing Functions](https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html)
- [OpenCV Text Rendering](https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576)
- [Color Spaces in OpenCV](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html)
