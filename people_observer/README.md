# people_observer

Real-time person detection and tracking with PTZ following. Uses GPU-accelerated YOLO on Spot's surround fisheye cameras to detect people, maps detections to robot-frame bearings using SDK intrinsics and transforms, and aims the Spot CAM PTZ at tracked targets.

## Architecture Overview

### System Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              USER EXECUTION                                             │
│  python -m people_observer.app <ROBOT_IP> [--mode transform|bearing]                    │
│                                [--visualize] [--save-images DIR]                        │
│                                [--dry-run] [--once] [--exit-on-detection]               │
└────────────────────────────────┬────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              app.py (Entry Point)                           │
│  - Parse CLI arguments                                                      │
│  - Create RuntimeConfig from config.py                                      │
│  - Connect to robot (io_robot.connect)                                      │
│  - Initialize clients (ImageClient, PtzClient, CompositorClient)            │
│  - Configure stream settings                                                │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           config.py (Configuration)                                     │
│  Constants:                                   Dataclasses:                              │
│  - SURROUND_SOURCES (5 cameras)               - RuntimeConfig                           │
│  - DEFAULT_YOLO_MODEL (yolov8n.pt)            - YOLOConfig                              │
│  - YOLO_DEVICE (cuda/cpu)                     - PTZConfig                               │
│  - MIN_CONFIDENCE (0.30)                      - ConnectionConfig                        │
│  - LOOP_HZ (7)                                                                          │
│  - PTZ_NAME (mech), COMPOSITOR_SCREEN         observer_mode: str (transform/bearing)    │
│  - DEFAULT_ZOOM (1.0)                         Intrinsics fetched at runtime from SDK    │
│  - TRANSFORM_MODE (transform - default)                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          tracker.run_loop() (Main Loop)                      │
│                                                                               │
│  ┌───────────────────────── 7 Hz Loop ─────────────────────────────────────────┐ 
│  │                                                                             │
│  │  1. Fetch Frames (cameras.py)                                               │
│  │     ├─> ImageClient.get_image_from_sources()                                │
│  │     ├─> Decode JPEG/RAW to BGR numpy arrays                                 │
│  │     └─> Return: dict[camera_name -> image], ImageResponses                  │        │
│  │         (ImageResponse contains frame_tree for transforms)                  │        │
│  │                                                                             │        │
│  │  2. Run Detection (detection.py)                                            │        │
│  │     ├─> YoloDetector.predict_batch()                                        │        │
│  │     │   ├─> GPU inference (CUDA) with FP16 if available                     │        │
│  │     │   ├─> CPU fallback if GPU unavailable                                 │        │
│  │     │   ├─> Filter: person class only (class_id=0)                          │        │
│  │     │   └─> Apply confidence threshold (0.30)                               │        │
│  │     └─> Return: list[list[Detection]] per camera                            │        │
│  │         Detection: {source, bbox_xywh, conf}                                │        │
│  │                                                                             │        │
│  │  3. Select Target Person (tracker.py)                                       │        │
│  │     ├─> Rank by depth if available (estimate_detection_depth_m)             │        │
│  │     ├─> Fallback: rank by largest bbox area (pick_largest)                  │        │
│  │     ├─> Filter by MIN_AREA_PX (600)                                         │        │
│  │     └─> Return: best (camera_name, Detection, ImageResponse)                │        │
│  │                                                                             │        │
│  │  4. Compute PTZ Angles (geometry.py)                                        │        │
│  │     │                                                                       │        │
│  │     ├─> Mode: "transform" (DEFAULT - accurate)                              │        │
│  │     │   ├─> Get camera intrinsics from ImageSource                          │        │
│  │     │   │   └─> Kannala-Brandt (k1-k4) for fisheye                          │        │
│  │     │   │   └─> Pinhole (fx, fy, cx, cy) for PTZ/hand                       │        │
│  │     │   ├─> pixel_to_camera_ray() using cv2.fisheye.undistortPoints         │        │
│  │     │   ├─> Transform ray: camera frame → body frame (SE3Pose)              │        │
│  │     │   ├─> Body bearing = atan2(ray.y, ray.x) [-180, 180]                  │        │
│  │     │   ├─> Body tilt = atan2(ray.z, hypot(x,y))                            │        │
│  │     │   ├─> Convert body → PTZ coordinates                                  │        │
│  │     │   │   └─> PTZ pan = -bearing (flip left/right)                        │        │
│  │     │   │   └─> Normalize pan to [0, 360]                                   │        │
│  │     │   │   └─> PTZ tilt = body tilt                                        │        │
│  │     │   └─> Return: (pan_deg, tilt_deg)                                     │        │
│  │     │                                                                       │        │
│  │     └─> Mode: "bearing" (fallback, no intrinsics)                           │        │
│  │         ├─> pixel_to_ptz_angles_simple()                                    │        │
│  │         ├─> Calculate HFOV from intrinsics or use fallback (133°)           │        │
│  │         ├─> pixel_offset = HFOV × (pixel_x/width - 0.5)                     │        │
│  │         ├─> camera_yaw from CAM_YAW_DEG config                              │        │
│  │         └─> pan_deg = camera_yaw + pixel_offset                             │        │
│  │                                                                             │        │
│  │  5. Command PTZ (ptz_control.py)                                            │        │
│  │     ├─> Query current PTZ position (get_ptz_position)                       │        │
│  │     ├─> Validate angles: pan [0,360], tilt [-30,100], zoom [1.0,30.0]       │        │
│  │     ├─> Clamp to valid ranges if needed                                     │        │
│  │     ├─> set_ptz(pan_deg, tilt_deg, zoom=1.0)                                │        │
│  │     │   ├─> PtzDescription(name="mech")                                     │        │
│  │     │   ├─> PtzClient.set_ptz_position(desc, pan, tilt, zoom)               │        │
│  │     │   └─> Log success/failure with detailed error info                    │        │
│  │     └─> If dry_run: log only, skip command                                  │        │
│  │                                                                             │        │
│  │  6. Visualization (optional, visualization.py)                              │        │
│  │     ├─> draw_detections() on each camera frame                              │        │
│  │     ├─> create_grid_layout() 3x2 grid for 5 cameras                         │        │
│  │     ├─> show_detections_grid() OpenCV window                                │        │
│  │     └─> Handle keyboard: 'q' quit, ESC quit                                 │        │
│  │                                                                             │        │
│  │  7. Save Images (optional, visualization.py)                                │        │
│  │     ├─> save_annotated_frames() if --save-images DIR                        │        │
│  │     ├─> Timestamp-based unique filenames (YYYYMMDD_HHMMSS_mmm)              │        │
│  │     ├─> Format: {timestamp}_iter{iteration:04d}_{camera}.jpg                │        │
│  │     └─> Validate images before saving (check not None/empty)                │        │
│  │                                                                             │        │
│  │  8. Loop Pacing                                                             │        
│  │     └─> sleep() to maintain LOOP_HZ (7 Hz = ~143ms)                         │        
│  │                                                               │             │
│  └───────────────────────────────────────────────────────────────┘             │
│                                                                                │
│  Exit Conditions:                                                              │
│  - cfg.once = True: exit after 1 iteration                                     │
│  - cfg.exit_on_detection = True: exit after successful PTZ command             │
│  - KeyboardInterrupt (Ctrl+C)                                                  │
│  - Visualization window: 'q' or ESC pressed                                    │
└────────────────────────────────────────────────────────────────────────────────

```

### Module Interaction Diagram

```
                         ┌──────────────────┐
                         │     app.py       │  Entry point, CLI parsing
                         │   (main entry)   │  Initializes clients & config
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼────────────────────┐
              │                   │                    │
              ▼                   ▼                    ▼
       ┌────────────┐      ┌────────────┐      ┌────────────┐
       │ config.py  │      │io_robot.py │      │ tracker.py │
       │            │      │            │      │            │
       │ Runtime    │      │ - connect()│      │ - run_loop │
       │ Config     │      │ - ensure   │      │ - main     │
       │ Constants  │      │   _clients │      │   detection│
       │ Dataclasses│      │ - configure│      │   loop     │
       └────────────┘      │   _stream  │      └──────┬─────┘
                           └──────┬─────┘             │
                                  │                   │
                                  │ ImageClient       │
                                  │ PtzClient         │
                                  │ CompositorClient  │
                                  │                   │
                    ┌─────────────┴───────────────────┴──────────────┐
                    │                                                 │
                    ▼                                                 ▼
         ┌──────────────────────┐                         ┌──────────────────┐
         │     cameras.py       │                         │  detection.py    │
         │                      │                         │                  │
         │ - fetch_image_sources│  ImageResponse          │ - YoloDetector   │
         │   (intrinsics cache) │  (with frame_tree)      │ - GPU/CPU auto   │
         │ - get_camera         │◄────────────────────────│ - predict_batch()│
         │   _intrinsics()      │                         │ - FP16 precision │
         │ - pixel_to_camera_ray│  BGR numpy arrays       │ - person filter  │
         │   (undistort)        │─────────────────────────>│                  │
         │ - calculate_hfov()   │                         └──────────────────┘
         │ - get_frames()       │                                  │
         └──────────┬───────────┘                                  │
                    │                                              │
                    │ Intrinsics:                                  │
                    │ - Kannala-Brandt (k1-k4)                     │
                    │ - Pinhole (fx, fy, cx, cy)                   │
                    │                                              │
                    └────────────┬─────────────────────────────────┘
                                 │ frames + detections + intrinsics
                                 ▼
                       ┌──────────────────────┐
                       │    geometry.py       │
                       │                      │
                       │ Transform Mode:      │
                       │ - pixel_to_ptz       │
                       │   _angles_transform()│
                       │   • cv2.fisheye      │
                       │     undistortPoints  │
                       │   • SDK frame_helpers│
                       │     get_a_tform_b()  │
                       │   • body→PTZ coords  │
                       │                      │
                       │ Bearing Mode:        │
                       │ - pixel_to_ptz       │
                       │   _angles_simple()   │
                       │   • HFOV projection  │
                       │   • camera yaw       │
                       └──────────┬───────────┘
                                  │ (pan_deg, tilt_deg)
                                  ▼
                       ┌──────────────────────┐
                       │   ptz_control.py     │
                       │                      │
                       │ - get_ptz_position() │
                       │   (query current)    │
                       │ - Validate ranges:   │
                       │   pan [0-360]        │
                       │   tilt [-30,100]     │
                       │   zoom [1.0-30.0]    │
                       │ - apply_deadband()   │
                       │ - clamp_step()       │
                       │ - set_ptz()          │
                       │   • Error handling   │
                       │   • Success logging  │
                       └──────────┬───────────┘
                                  │ PtzClient API
                                  ▼
                            Spot CAM PTZ
                             (hardware)
                          360° pan, ±100° tilt

              Optional Visualization & Logging:
                       ┌──────────────────────┐
                       │  visualization.py    │
                       │                      │
                       │ - draw_detections()  │
                       │ - create_grid_layout │
                       │   (3x2 grid)         │
                       │ - show_detections    │
                       │   _grid()            │
                       │ - save_annotated     │
                       │   _frames()          │
                       │   • Timestamp naming │
                       │   • Never overwrites │
                       └──────────┬───────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
            OpenCV Window                 Disk Storage
         (5 cameras, grid view)      (timestamped JPEGs)
           Press 'q' to quit          ./images/*.jpg
```

### Data Flow Details

**Frame Acquisition** (cameras.py):
```
ImageClient.get_image_from_sources(sources)
    └─> List[ImageResponse]
        └─> decode_image(response)
            └─> np.ndarray (BGR, HxWx3)
```

**Detection** (detection.py):
```
YoloDetector.predict_batch(bgr_list)
    └─> YOLO.predict(device=cuda/cpu, conf=0.30, classes=[0])
        └─> Filter: person class only
            └─> List[Detection(source, bbox_xywh, conf)]
```

**PTZ Angle Calculation** (geometry.py):
```
Transform Mode (DEFAULT - accurate with distortion correction):
  pixel (u,v)
    └─> cameras.get_camera_intrinsics() [Kannala-Brandt k1-k4 or pinhole]
    └─> cameras.pixel_to_camera_ray() [cv2.fisheye.undistortPoints()]
    └─> frame_helpers.get_a_tform_b(frame_tree, body, camera) [SE3Pose]
    └─> Transform ray: camera frame → body frame [Vec3 multiply]
    └─> bearing_rad = atan2(ray_body.y, ray_body.x) [-π, π]
    └─> tilt_rad = atan2(ray_body.z, hypot(x,y))
    └─> Convert body → PTZ coordinates:
        • PTZ pan = -bearing (negate to flip left/right)
        • Normalize pan to [0, 360]
        • PTZ tilt = body tilt

Bearing Mode (fallback when intrinsics unavailable):
  pixel (u,v)
    └─> pixel_offset = HFOV × (pixel_x / width - 0.5)
    └─> camera_yaw from CAM_YAW_DEG config
    └─> pan_deg = camera_yaw + pixel_offset
    └─> tilt_deg = DEFAULT_TILT_DEG (constant)
```

**PTZ Command** (ptz_control.py):
```
target_pan/tilt -> deadband filter -> step limiter
                -> degrees to radians
                -> PtzClient.set_ptz_position()
```

**Visualization** (visualization.py):
```
Per Frame (each camera):
  draw_detections(image, detections, camera_name)
    ├─> For each Detection:
    │   ├─> Draw green bounding box (x, y, w, h)
    │   ├─> Draw confidence label: "Person 0.87"
    │   └─> Black semi-transparent background for readability
    ├─> Camera name in top-left corner
    └─> Detection count in top-right corner

Grid Layout:
  create_grid_layout(annotated_images, cols=3, target_width=1920)
    ├─> Resize each image to cell size (640x480)
    ├─> Arrange in 3-column grid (5 cameras = 2 rows)
    └─> Fill remaining cells with black

Display:
  show_detections_grid(frames_dict, detections_dict)
    ├─> Annotate all frames
    ├─> Create grid layout
    ├─> Add stats panel: "Total detections: 2 across 5 cameras | Press 'q' to quit"
    ├─> cv2.imshow() - non-blocking (1ms wait)
    └─> Return key code ('q' or ESC = quit)
```

### Visualization Module Details

The visualization system provides real-time feedback during detection and tracking operations.

#### Layout Structure
```
┌─────────────────┬─────────────────┬─────────────────┐
│   FrontLeft     │   FrontRight    │    Left         │
│   640x480       │   640x480       │   640x480       │
│  [Person 0.89]  │  [Person 0.76]  │                 │
├─────────────────┼─────────────────┼─────────────────┤
│   Right         │     Back        │   (empty)       │
│   640x480       │   640x480       │   640x480       │
│                 │  [Person 0.82]  │                 │
└─────────────────┴─────────────────┴─────────────────┘
        Total detections: 3 across 5 cameras
              Press 'q' to quit
```

#### Color Coding
- **Green boxes** (0, 255, 0): Person detections above confidence threshold
- **White text** (255, 255, 255): Confidence scores and labels
- **Black backgrounds** (0, 0, 0): Semi-transparent for text readability

#### Usage Modes

**1. Live Tracking with Visualization**
```powershell
python -m people_observer.app --hostname <IP> --visualize
```
- Updates at 7 Hz (synchronized with detection loop)
- Non-blocking display (1ms cv2.waitKey)
- Press 'q' or ESC to gracefully exit

**2. Dry-Run Debug Mode**
```powershell
python -m people_observer.app --hostname <IP> --dry-run --visualize
```
- See detections without PTZ commands
- Verify camera alignment and detection quality
- Check confidence thresholds visually

**3. Single Frame Capture**
```powershell
python -m people_observer.app --hostname <IP> --once --visualize
```
- Process one detection cycle
- View results, then exit
- Useful for testing camera positioning

#### Performance Characteristics
- **Overhead**: ~10-20ms per frame for annotation + display
- **Resolution**: 1920x1040 total (3x640 + stats panel)
- **Memory**: ~12MB for grid (5 cameras × 640×480×3 bytes)
- **CPU Usage**: Minimal (OpenCV hardware-accelerated when available)

#### Integration with Tracker
```python
# In tracker.py main loop
if cfg.visualize:
    key = visualization.show_detections_grid(frames, detections_by_camera)
    if key == ord('q') or key == 27:  # 'q' or ESC
        logger.info("User requested quit via visualization")
        break
```

#### Debugging Features

**Camera Label Annotations:**
- Camera names cleaned: "frontleft_fisheye_image" → "FRONTLEFT"
- Detection counts per camera
- Overall statistics across all cameras

**Bounding Box Information:**
- Position: Label placed above box (or below if top clipped)
- Format: "Person 0.87" (confidence to 2 decimal places)
- Clamping: Boxes constrained to image boundaries

**Save Frames (Optional):**
```python
visualization.save_annotated_frames(
    frames_dict,
    detections_dict,
    output_dir="./debug_frames",
    iteration=42
)
# Saves: debug_frames/iter0042_frontleft_fisheye_image.jpg (×5)
```

#### Window Management
- **Window Name**: "People Observer - Detections"
- **Resize**: Auto-scaled to fit 1920px width
- **Position**: OS default (can be moved by user)
- **Focus**: Requires focus for keyboard input
- **Close**: Window closed automatically on exit

#### Constants (All Configurable in visualization.py)
```python
GRID_COLS = 3                    # Columns in grid
DEFAULT_TARGET_WIDTH = 1920      # Total grid width
GRID_ASPECT_RATIO = 3.0 / 4.0   # 4:3 for fisheye
THICKNESS = 2                    # Box line thickness
FONT_SCALE = 0.5                 # Label text size
CAMERA_LABEL_PADDING = 10        # Corner label spacing
CONFIDENCE_LABEL_PADDING = 4     # Box label spacing
STATS_PANEL_HEIGHT = 40          # Bottom panel height
WAIT_KEY_MS = 1                  # Non-blocking key check
```

## What's here
- `app.py` - Main entry point: orchestrates camera capture, detection, tracking, and PTZ aiming loop.
- `cameras.py` - Camera source management via `ImageClient`; handles multiple surround fisheye sources.
- `detection.py` - YOLO person detection wrapper using Ultralytics; returns bounding boxes and confidences.
- `tracker.py` - Main detection loop; selects best person target and computes PTZ angles.
- `ptz_control.py` - PTZ aiming logic; converts robot-frame bearings to pan/tilt commands via `PtzClient`.
- `geometry.py` - Coordinate transforms: pixel coordinates -> robot-frame bearing using per-camera yaw and FOV assumptions.
- `io_robot.py` - Robot interface wrapper; handles connection, time sync, and client initialization.
- `config.py` - Configuration management with nested dataclasses; all constants in one place (GPU device, model, thresholds, PTZ params).
- `visualization.py` - OpenCV-based live detection visualization:
  - `draw_detections()`: Annotate single camera frame with bounding boxes and confidence labels
  - `create_grid_layout()`: Arrange 5 cameras in 3x2 grid (640x480 cells, 1920px total width)
  - `show_detections_grid()`: Display interactive window with stats panel and keyboard controls
  - `save_annotated_frames()`: Save debug snapshots to disk
- `test_yolo_model.py` - Verify YOLO model loads and show available models/classes.
- `test_yolo_webcam.py` - Benchmark YOLO models on laptop webcam with GPU/CPU performance metrics.

## Requirements
Install the Boston Dynamics Spot SDK wheels first from the sibling `spot-sdk/prebuilt` directory in this workspace (v5.0.1.2), then the Python dependencies in `requirements.txt`.

**Authentication**: Your venv `Activate.ps1` supplies credentials/tokens; **do not include user/password on the CLI**. Scripts that rely on `bosdyn.client.util.authenticate` will use your stored token. Avoid committing secrets.

**Notes**:
- Depends on live robot services: Image, PTZ, Directory (for time sync).
- Camera sources: surround fisheyes (`frontleft_fisheye_image`, `frontright_fisheye_image`, `left_fisheye_image`, `right_fisheye_image`, `back_fisheye_image`).
- PTZ streaming configured via `CompositorClient` and `StreamQualityClient`; adjust settings in `config.py`.
- YOLO model: defaults to `yolov8n.pt` (lightweight); use `yolov8s.pt` or larger for better accuracy.

## Install
1) Activate your venv (with auth in `Activate.ps1`).
2) Install Spot SDK wheels from `../../spot-sdk/prebuilt/*.whl`:
   ```powershell
   pip install ../../spot-sdk/prebuilt/bosdyn_client-5.0.1.2-py3-none-any.whl
   pip install ../../spot-sdk/prebuilt/bosdyn_api-5.0.1.2-py3-none-any.whl
   pip install ../../spot-sdk/prebuilt/bosdyn_core-5.0.1.2-py3-none-any.whl
   pip install ../../spot-sdk/prebuilt/bosdyn_mission-5.0.1.2-py3-none-any.whl
   ```
3) `pip install -r requirements.txt` (includes ultralytics, opencv-python, numpy).

## Testing

### 1. Verify YOLO Model
```powershell
python -m friendly_spot.people_observer.test_yolo_model
```
Shows available YOLOv8 models and verifies the model loads correctly.

### 2. Test Detection (Dry-Run with Visualization)
```powershell
# Single cycle test
python -m friendly_spot.people_observer.app --hostname $env:ROBOT_IP --once --dry-run --visualize

# Continuous test with live visualization
python -m friendly_spot.people_observer.app --hostname $env:ROBOT_IP --dry-run --visualize
```
- `--dry-run`: Skips PTZ commands, only logs what would be sent
- `--visualize`: Shows OpenCV window with all 5 camera views and bounding boxes
- Press 'q' or ESC in visualization window to quit

### 3. Run Live (with PTZ Control)
```powershell
# Normal operation
python -m friendly_spot.people_observer.app --hostname <ROBOT_IP>

# With visualization
python -m friendly_spot.people_observer.app --hostname <ROBOT_IP> --visualize
```

## CLI Arguments
- `--hostname <IP>`: Robot IP address (required)
- `--mode <bearing|transform>`: Coordinate mapping mode (default: bearing)
  - `bearing`: Fast HFOV-based pixel->yaw mapping
  - `transform`: Uses camera intrinsics and frame transforms (more accurate)
- `--once`: Run single detection cycle and exit (for testing)
- `--dry-run`: Skip PTZ commands, log only
- `--visualize`: Show live OpenCV window with detections

## Configuration
Edit `config.py` or use environment variables:
- `YOLO_MODEL`: Path to YOLO model file
- `PEOPLE_OBSERVER_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `PEOPLE_OBSERVER_CONFIDENCE`: Minimum detection confidence (0.0-1.0)
- `LOOP_HZ`: Detection loop frequency