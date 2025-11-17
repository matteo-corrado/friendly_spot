# people_observer

Real-time person detection and tracking with PTZ following. Uses YOLO on Spot's surround fisheye cameras to detect people, maintains track identities across frames, maps detections to robot-frame bearings, and aims the Spot CAM PTZ at tracked targets.

## Architecture Overview

### System Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER EXECUTION                                  │
│  python -m people_observer.app --hostname <IP> [--dry-run] [--visualize]   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              app.py (Entry Point)                            │
│  - Parse CLI arguments                                                       │
│  - Create RuntimeConfig from config.py                                       │
│  - Connect to robot (io_robot.connect)                                       │
│  - Initialize clients (ImageClient, PtzClient, CompositorClient)            │
│  - Configure stream settings                                                 │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           config.py (Configuration)                          │
│  Constants:                                   Dataclasses:                   │
│  - SURROUND_SOURCES (5 cameras)               - RuntimeConfig               │
│  - DEFAULT_YOLO_MODEL (yolov8x.pt)            - YOLOConfig                  │
│  - YOLO_DEVICE (cuda)                         - PTZConfig                   │
│  - MIN_CONFIDENCE (0.30)                      - ConnectionConfig            │
│  - LOOP_HZ (7)                                - ObserverMode                │
│  - PTZ control parameters                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          tracker.run_loop() (Main Loop)                      │
│                                                                               │
│  ┌───────────────────────── 7 Hz Loop ─────────────────────────┐            │
│  │                                                               │            │
│  │  1. Fetch Frames (cameras.py)                                │            │
│  │     ├─> ImageClient.get_image_from_sources()                 │            │
│  │     ├─> Decode JPEG/RAW to BGR numpy arrays                  │            │
│  │     └─> Return: dict[camera_name -> image], responses        │            │
│  │                                                               │            │
│  │  2. Run Detection (detection.py)                             │            │
│  │     ├─> YoloDetector.predict_batch()                         │            │
│  │     │   ├─> GPU/CPU inference with FP16 (if CUDA)            │            │
│  │     │   ├─> Filter: person class only (class_id=0)           │            │
│  │     │   └─> Apply confidence threshold (0.30)                │            │
│  │     └─> Return: list[list[Detection]] per camera             │            │
│  │         Detection: {source, bbox_xywh, conf}                 │            │
│  │                                                               │            │
│  │  3. Select Target Person (tracker.py)                        │            │
│  │     ├─> Rank by depth if available (future)                  │            │
│  │     ├─> Fallback: rank by largest bbox area                  │            │
│  │     ├─> Filter by MIN_AREA_PX (600)                          │            │
│  │     └─> Return: best (camera_name, Detection)                │            │
│  │                                                               │            │
│  │  4. Compute Robot-Frame Bearing (geometry.py)                │            │
│  │     │                                                         │            │
│  │     ├─> Mode: "bearing" (default, fast)                      │            │
│  │     │   ├─> bbox_center() -> pixel (cx, cy)                  │            │
│  │     │   ├─> pixel_to_yaw_offset() using HFOV                 │            │
│  │     │   ├─> camera_yaw_from_transforms() or config lookup    │            │
│  │     │   └─> pan_deg = camera_yaw + pixel_offset              │            │
│  │     │                                                         │            │
│  │     └─> Mode: "transform" (accurate, uses intrinsics)        │            │
│  │         ├─> pixel_to_ray_pinhole() using camera intrinsics   │            │
│  │         ├─> transform_direction() to BODY frame              │            │
│  │         ├─> pan = atan2(y, x)                                │            │
│  │         └─> tilt = atan2(-z, hypot(x,y))                     │            │
│  │                                                               │            │
│  │  5. Command PTZ (ptz_control.py)                             │            │
│  │     ├─> apply_deadband() (1.0 deg pan, 1.0 deg tilt)         │            │
│  │     ├─> clamp_step() (max 8.0 deg per step)                  │            │
│  │     ├─> set_ptz(pan_deg, tilt_deg, zoom=0.0)                 │            │
│  │     │   ├─> Convert degrees to radians                       │            │
│  │     │   └─> PtzClient.set_ptz_position()                     │            │
│  │     └─> If dry_run: log only, skip command                   │            │
│  │                                                               │            │
│  │  6. Visualization (optional, visualization.py)               │            │
│  │     ├─> draw_detections() on each camera frame               │            │
│  │     ├─> create_grid_layout() 3x2 grid for 5 cameras          │            │
│  │     ├─> show_detections_grid() OpenCV window                 │            │
│  │     └─> Handle keyboard: 'q' quit, ESC quit                  │            │
│  │                                                               │            │
│  │  7. Loop Pacing                                               │            │
│  │     └─> sleep() to maintain LOOP_HZ (7 Hz = ~143ms)          │            │
│  │                                                               │            │
│  └───────────────────────────────────────────────────────────────┘            │
│                                                                               │
│  Exit Conditions:                                                             │
│  - cfg.once = True: exit after 1 iteration                                   │
│  - KeyboardInterrupt (Ctrl+C)                                                │
│  - Visualization window: 'q' or ESC pressed                                  │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Module Interaction Diagram

```
                    ┌──────────────┐
                    │   app.py     │  Entry point, CLI parsing
                    │ (main loop)  │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ config.py  │  │io_robot.py │  │ tracker.py │
    │            │  │            │  │            │
    │ All consts │  │ Connect    │  │ Main loop  │
    │ & dataclass│  │ Auth       │  │ orchestr.  │
    └────────────┘  │ Clients    │  └──────┬─────┘
                    └──────┬─────┘         │
                           │               │
                    ┌──────┴───────────────┴──────────┐
                    │                                  │
                    ▼                                  ▼
         ┌──────────────────┐              ┌──────────────────┐
         │   cameras.py     │              │  detection.py    │
         │                  │              │                  │
         │ - list sources   │              │ - YoloDetector   │
         │ - get_frames()   │─────────────>│ - GPU/CPU device │
         │ - decode images  │   BGR imgs   │ - predict_batch()│
         └──────────────────┘              └──────────────────┘
                    │                                  │
                    │                                  │
                    └─────────┬────────────────────────┘
                              │ frames + detections
                              ▼
                    ┌──────────────────┐
                    │   geometry.py    │
                    │                  │
                    │ - bbox_center    │
                    │ - pixel_to_yaw   │
                    │ - transforms     │
                    │ - bearing calc   │
                    └─────────┬────────┘
                              │ pan/tilt angles
                              ▼
                    ┌──────────────────┐
                    │ ptz_control.py   │
                    │                  │
                    │ - deadband       │
                    │ - step limiting  │
                    │ - set_ptz()      │
                    └─────────┬────────┘
                              │ PtzClient
                              ▼
                         Spot CAM PTZ
                          (hardware)

         Optional Visualization Branch:
                    ┌──────────────────┐
                    │visualization.py  │
                    │                  │
                    │ - draw_detections│
                    │ - grid layout    │
                    │ - OpenCV display │
                    └──────────────────┘
                              │
                              ▼
                     OpenCV Window
                  (5 cameras, live view)
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

**Bearing Calculation** (geometry.py):
```
Bearing Mode:
  pixel (u,v) -> yaw_offset (HFOV) -> robot_yaw (transforms) -> pan_deg

Transform Mode:
  pixel (u,v) -> ray_camera (intrinsics) -> ray_body (transforms)
              -> (pan, tilt) via atan2
```

**PTZ Command** (ptz_control.py):
```
target_pan/tilt -> deadband filter -> step limiter
                -> degrees to radians
                -> PtzClient.set_ptz_position()
```

## What's here
- `app.py` - Main entry point: orchestrates camera capture, detection, tracking, and PTZ aiming loop.
- `cameras.py` - Camera source management via `ImageClient`; handles multiple surround fisheye sources.
- `detection.py` - YOLO person detection wrapper using Ultralytics; returns bounding boxes and confidences.
- `tracker.py` - Main detection loop; selects best person target and computes PTZ angles.
- `ptz_control.py` - PTZ aiming logic; converts robot-frame bearings to pan/tilt commands via `PtzClient`.
- `geometry.py` - Coordinate transforms: pixel coordinates -> robot-frame bearing using per-camera yaw and FOV assumptions.
- `io_robot.py` - Robot interface wrapper; handles connection, time sync, lease management with `LeaseKeepAlive`.
- `config.py` - Configuration management with nested dataclasses for different subsystems.
- `visualization.py` - OpenCV-based live detection visualization with bounding boxes.
- `test_yolo_model.py` - Verify YOLO model loads and show available models/classes.

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
python -m friendly_spot.people_observer.app --hostname <ROBOT_IP> --once --dry-run --visualize

# Continuous test with live visualization
python -m friendly_spot.people_observer.app --hostname <ROBOT_IP> --dry-run --visualize
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