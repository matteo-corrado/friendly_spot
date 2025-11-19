# Perception Module

Person detection and tracking using YOLO and PTZ camera control.

## Overview

The perception module handles real-time person detection from Spot's cameras using YOLOv8, tracking logic to maintain focus on the target person, and PTZ (pan-tilt-zoom) camera control to keep tracked persons in view.

## Components

### Core Detection
- **`yolo_detector.py`**: YOLO wrapper with batch prediction and person-class filtering
  - `Detection`: Dataclass for detection results (bbox, confidence, segmentation mask)
  - `YoloDetector`: Model loader and inference engine
  - GPU acceleration with automatic fallback to CPU
  
- **`detection_types.py`**: Common data structures for detection results
  - Used by behavior and visualization modules

### Camera Interface
- **`cameras.py`**: Camera frame acquisition from Spot SDK ImageClient
  - Fetch frames from surround cameras (fisheye) or depth sensors
  - Handle image rotation and format conversion (JPEG → BGR)
  - Intrinsics retrieval for coordinate transforms

### Geometry & Transforms
- **`geometry.py`**: Coordinate frame mathematics
  - Pixel to 3D ray projection using camera intrinsics (Kannala-Brandt or pinhole)
  - Frame transforms (image → body → vision)
  - Bearing calculations for spatial reasoning

### Tracking Logic
- **`tracker.py`**: Main detection loop orchestrator
  - Fetch frames from multiple cameras
  - Run YOLO detection in batch mode
  - Select target person (nearest by depth or largest by area)
  - Compute PTZ pan/tilt angles and command camera
  - Integration point for visualization

### Configuration
- **`config.py`**: Centralized configuration for perception pipeline
  - Camera source definitions (surround cameras, PTZ, depth)
  - YOLO model parameters (confidence, IOU, image size)
  - Detection thresholds and coordinate frame settings
  - Runtime config dataclass for mode selection

## Usage

### Basic Detection Pipeline

```python
from src.perception import YoloDetector
from src.perception.cameras import fetch_camera_frames

# Initialize detector
detector = YoloDetector(model_path="yolov8n.pt", conf=0.4, device="cuda")

# Fetch frames
frames = fetch_camera_frames(image_client, sources=["frontleft_fisheye_image"])

# Run detection
detections = detector.predict_batch([f['image'] for f in frames])

# Process results
for frame, dets in zip(frames, detections):
    for det in dets:
        print(f"Person detected: bbox={det.bbox_xywh}, conf={det.conf:.2f}")
```

### PTZ Tracking Loop

```python
from src.perception.tracker import run_loop
from src.perception.config import RuntimeConfig

# Configure tracking mode
config = RuntimeConfig(
    mode="transform",  # Use SDK intrinsics for precise transforms
    enable_depth=True,  # Prioritize nearest person
    target_ptz_distance_m=2.5  # Desired tracking distance
)

# Run tracking loop (blocking)
run_loop(robot, image_client, ptz_client, config)
```

### Custom Detection Selection

```python
from src.perception.tracker import estimate_detection_depth_m

# Get depth for each detection
depths = []
for det in detections:
    depth = estimate_detection_depth_m(depth_image, det, use_mask=True)
    depths.append(depth)

# Select nearest person
nearest_idx = depths.index(min(d for d in depths if d is not None))
target = detections[nearest_idx]
```

## Coordinate Frames

The perception module works with multiple coordinate frames:

- **Image Frame**: Pixel coordinates (origin at top-left, y-down, x-right)
- **Camera Frame**: 3D optical frame (z forward, x right, y down)
- **Body Frame**: Robot body frame (x forward, y left, z up)
- **Vision Frame**: Non-moving odometry frame (for stable world references)

All transforms use Boston Dynamics SDK `frame_helpers` and camera intrinsics from `ImageResponse.source.pinhole` or `ImageResponse.source.fisheye`.

## Depth Estimation

Depth prioritization requires depth sensor data:

1. **Mask-based** (most precise): Use YOLO segmentation mask to extract median depth only for person pixels
2. **Bbox-based** (fallback): Sample central 40% of bounding box to avoid edges
3. **Area-based** (no depth): Select largest detection by pixel area

Configure via `RuntimeConfig.enable_depth` and ensure depth sources are available in `config.SURROUND_SOURCES`.

## Model Requirements

### YOLO Models
Download YOLOv8 weights to workspace root:
- Detection: `yolov8n.pt` (11 MB, fast)
- Segmentation: `yolov8n-seg.pt` (12 MB, required for mask-based depth)

```bash
# Download detection model
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o yolov8n.pt

# Download segmentation model (recommended)
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt -o yolov8n-seg.pt
```

### GPU Acceleration
For real-time performance on high-resolution cameras:
- CUDA-capable GPU (tested on RTX 3060)
- PyTorch with CUDA support: `torch>=2.0.0+cu118`
- Automatic fallback to CPU if GPU unavailable

## Configuration

Edit `src/perception/config.py` for camera sources, YOLO parameters, and tracking behavior:

```python
# Camera sources for multi-camera detection
SURROUND_SOURCES = [
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    # Add more cameras as needed
]

# YOLO detection parameters
MIN_CONFIDENCE = 0.4  # Lower = more detections (may include false positives)
YOLO_IOU_THRESHOLD = 0.5  # Higher = stricter NMS overlap rejection

# PTZ tracking
TARGET_PTZ_DISTANCE_M = 2.5  # Desired distance to tracked person
PTZ_MOVEMENT_THRESHOLD_DEG = 5.0  # Minimum angle change before commanding PTZ
```

## Extending Detection

### Add New YOLO Classes

Currently filters for `person` class only. To detect other objects:

```python
# In yolo_detector.py
TRACKED_CLASSES = [0, 16, 24]  # person, dog, backpack

# Modify predict_batch() to accept class list:
results = model.predict(
    images,
    classes=TRACKED_CLASSES,  # Filter detections
    ...
)
```

### Implement Custom Tracker

Replace simple area/depth selection with Kalman filter or DeepSORT:

```python
from src.perception.tracker import run_loop

def custom_target_selector(detections, depth_map, prev_target):
    """
    Args:
        detections: List[Detection] across all cameras
        depth_map: Optional depth image
        prev_target: Previous target detection for continuity
    
    Returns:
        Selected Detection or None
    """
    # Implement tracking logic (e.g., associate by IOU, predict position)
    pass

# Modify run_loop() to use custom selector
```

## Troubleshooting

### No Detections
- **Low confidence threshold**: Increase `MIN_CONFIDENCE` in config.py
- **Wrong camera**: Verify `SURROUND_SOURCES` cameras exist on your Spot (use `list_image_sources` utility)
- **Model not loaded**: Check YOLO model file exists at `DEFAULT_YOLO_MODEL` path

### Poor PTZ Tracking
- **Intrinsics unavailable**: Falls back to HFOV projection (less accurate). Ensure camera intrinsics are populated in `ImageResponse`.
- **Frame transform errors**: Check robot time sync (`robot.time_sync.wait_for_sync()`)
- **Depth unreliable**: Verify depth sensors are enabled and depth images have valid pixels (not all NaN)

### Slow Inference
- **Large image size**: Reduce `YOLO_IMG_SIZE` (must be multiple of 32, e.g., 640 → 416)
- **CPU inference**: Enable GPU with `YOLO_DEVICE = "cuda"` and install PyTorch CUDA
- **Too many cameras**: Reduce `SURROUND_SOURCES` to 1-2 cameras for faster batch processing

## Dependencies

- `ultralytics` >= 8.0.0: YOLO inference
- `torch` >= 2.0.0: PyTorch backend
- `opencv-python` >= 4.8.0: Image processing
- `numpy` >= 1.24.0: Array operations
- `bosdyn-client` == 5.0.1.2: Spot SDK

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Spot SDK Image Service](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/image)
- [Spot SDK Frame Helpers](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/frame_helpers)
- Kannala-Brandt fisheye model: [OpenCV Fisheye Camera Model](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)
