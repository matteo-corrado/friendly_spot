# Friendly Spot: Socially-Aware Robot Interaction

**COSC 69.15/169.15 - Robotics Perception and Interaction, Dartmouth College**

**Authors:** Thor Lemke, Sally Hyun Hahm, Matteo Corrado  
**Version:** 2.0.0 (Modular Architecture)  
**Date:** January 2025

A perception and behavior system enabling Boston Dynamics Spot to detect people, track their movements with PTZ camera, estimate social distance, and respond with proximity-aware behaviors. Uses YOLOv8 person detection, depth-based prioritization, and a comfort model for natural human-robot interaction.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Modules](#modules)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [References](#references)

---

## Features

### Perception System
- **YOLO Person Detection**: GPU-accelerated YOLOv8 detection across multiple cameras
- **Multi-Camera Fusion**: Batch processing of surround fisheye cameras
- **Depth Estimation**: Prioritize nearest person using depth sensors and segmentation masks
- **PTZ Tracking**: Automatic pan-tilt-zoom camera control to keep person in view
- **Coordinate Transforms**: Precise bearing calculation using SDK camera intrinsics

### Behavior Planning
- **Comfort Model**: Proximity-based social distance reasoning
- **Behavior States**: GO_CLOSE, MAINTAIN_DISTANCE, GO_AWAY, EXPLORE, STOP
- **Debouncing**: Prevents rapid behavior oscillation at zone boundaries
- **Adaptive Thresholds**: Configurable personal/social distance zones

### Robot Control
- **SE2 Trajectory Commands**: Goal-based navigation in vision frame (stable world coordinates)
- **Lease Management**: Automatic acquisition/release with force-take support
- **E-Stop Safety**: Proper safety endpoint registration
- **Client Management**: Lazy-loaded SDK service clients

### Visualization
- **Real-time Overlays**: Detection boxes, confidence scores, depth info
- **Multi-Camera Grid**: Synchronized display of multiple camera feeds
- **PTZ Indicators**: Pan/tilt/zoom angle display
- **Recording**: Save annotated videos for analysis

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Friendly Spot v2.0                          │
└────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐        ┌──────────────┐      ┌──────────────┐
   │ Webcam  │        │ Spot Cameras │      │ Spot CAM PTZ │
   │(Testing)│        │  (Surround)  │      │   (Color)    │
   └────┬────┘        └──────┬───────┘      └──────┬───────┘
        │                    │                     │
        └────────┬───────────┴──────────┬──────────┘
                 ▼                      ▼
        ┌─────────────────┐    ┌────────────────┐
        │  Video Module   │    │ Robot Module   │
        │  (src/video)    │    │  (src/robot)   │
        └────────┬────────┘    └────────┬───────┘
                 │                      │
                 ▼                      │
        ┌─────────────────┐             │
        │ Perception Mod. │             │
        │ (src/perception)│             │
        │                 │             │
        │ • YOLO Detector │             │
        │ • Depth Est.    │             │
        │ • Tracker       │             │
        │ • PTZ Control   │◄────────────┘
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Behavior Mod.  │
        │ (src/behavior)  │
        │                 │
        │ • Comfort Model │
        │ • Executor      │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Visualization   │
        │(src/visualizati)│
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Spot Robot     │
        │  (BDS SDK 5.0)  │
        └─────────────────┘
```

---

## Project Structure

```
friendly_spot/
├── src/                           # Main source code (modular)
│   ├── __init__.py               # Package exports
│   ├── perception/               # Detection and tracking
│   │   ├── yolo_detector.py     # YOLO wrapper
│   │   ├── tracker.py           # Main tracking loop
│   │   ├── cameras.py           # Camera frame acquisition
│   │   ├── geometry.py          # Coordinate transforms
│   │   ├── detection_types.py   # Data structures
│   │   ├── config.py            # Configuration
│   │   └── README.md            # Perception docs
│   │
│   ├── behavior/                 # Decision and execution
│   │   ├── planner.py           # Comfort model
│   │   ├── executor.py          # Command execution
│   │   └── README.md            # Behavior docs
│   │
│   ├── robot/                    # Spot SDK interface
│   │   ├── io.py                # Connection and clients
│   │   ├── action_monitor.py   # Async command tracking
│   │   ├── observer_bridge.py  # Perception → behavior bridge
│   │   ├── ptz_control.py      # PTZ camera control
│   │   └── README.md            # Robot docs
│   │
│   ├── video/                    # Video capture
│   │   ├── sources.py           # Camera abstractions
│   │   ├── ptz_stream.py        # PTZ streaming
│   │   ├── webrtc_client.py    # WebRTC client
│   │   └── README.md            # Video docs
│   │
│   └── visualization/            # Overlay rendering
│       ├── overlay.py           # Main visualizer
│       ├── helpers.py           # Drawing utilities
│       └── README.md            # Visualization docs
│
├── tests/                        # Test suite
│   ├── test_imports.py          # Module import validation
│   ├── test_image_sources.py   # Camera tests
│   ├── test_ptz_convention.py  # PTZ angle tests
│   └── README.md                # Testing docs
│
├── data/                         # Data storage
│   ├── models/                  # YOLO model weights
│   ├── outputs/                 # Logs, videos, images
│   ├── datasets/                # Test data
│   └── README.md                # Data docs
│
├── docs/                         # Documentation
│   ├── DOCUMENTATION_STYLE_GUIDE.md
│   └── REFACTOR_STATUS.md       # Refactor progress
│
├── Behavioral/                   # Original behavioral code
├── Facial Recognition/           # Face recognition prototypes
├── people_observer/              # Original people observer
│
├── friendly_spot_main.py         # Main entry point
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore
```

---

## Quick Start

### Prerequisites

- **Python**: 3.9 or 3.10 (required by Spot SDK)
- **Spot Robot**: Boston Dynamics Spot with SDK v5.0.1.2
- **Spot CAM** (optional): For PTZ camera tracking
- **GPU** (recommended): CUDA-capable GPU for real-time YOLO inference

### Installation

```powershell
# 1. Clone the repository
git clone https://github.com/yourusername/friendly_spot.git
cd friendly_spot

# 2. Install Spot SDK (from workspace root)
pip install ..\spot-sdk\prebuilt\*.whl

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download YOLO model
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt -o data/models/yolov8n-seg.pt

# 5. Set robot credentials (optional, avoids interactive prompt)
$env:BOSDYN_CLIENT_USERNAME = "user"
$env:BOSDYN_CLIENT_PASSWORD = "password"
```

### Basic Usage

**Test with Webcam** (no robot required):

```powershell
python friendly_spot_main.py --webcam
```

**Run with Robot** (person detection + behavior):

```powershell
python friendly_spot_main.py --robot 192.168.80.3
```

**PTZ Tracking Only** (no behaviors):

```powershell
python friendly_spot_main.py --robot 192.168.80.3 --ptz-only
```

---

## Modules

### Perception (`src/perception/`)

Person detection, tracking, and PTZ control.

**Key Components:**
- **YOLO Detector**: YOLOv8 person detection with segmentation masks
- **Tracker**: Multi-camera detection fusion and target selection
- **Geometry**: Camera intrinsics, pixel→ray transforms, bearing calculations
- **PTZ Control**: Automatic camera aiming based on person position

**Usage:**
```python
from src.perception import YoloDetector, run_loop
from src.perception.config import RuntimeConfig

detector = YoloDetector(model_path="data/models/yolov8n-seg.pt", device="cuda")
config = RuntimeConfig(mode="transform", enable_depth=True)
run_loop(robot, image_client, ptz_client, config)
```

[**Full Perception Documentation**](src/perception/README.md)

---

### Behavior (`src/behavior/`)

Comfort-based behavior planning and robot command execution.

**Key Components:**
- **Comfort Model**: Maps person distance → behavior (proxemics theory)
- **Behavior Executor**: Translates behaviors into SDK RobotCommands

**Usage:**
```python
from src.behavior import ComfortModel, BehaviorExecutor

comfort = ComfortModel(comfortable_distance=2.0)

with BehaviorExecutor(robot) as executor:
    behavior = comfort.update(detection)
    executor.execute_behavior(behavior)
```

📖 [**Full Behavior Documentation**](src/behavior/README.md)

---

### Robot (`src/robot/`)

Spot SDK connection, client management, and control utilities.

**Key Components:**
- **Connection**: `create_robot()` with authentication and time sync
- **Clients**: Lazy-loaded SDK service clients (command, image, lease, etc.)
- **Lease/E-Stop**: Context managers for safe resource management
- **PTZ Control**: Pan-tilt-zoom camera commanding

**Usage:**
```python
from src.robot import create_robot, RobotClients, ManagedLease

robot = create_robot("192.168.80.3", register_spot_cam=True)
clients = RobotClients(robot)

with ManagedLease(robot):
    clients.command.robot_command(cmd)
```

📖 [**Full Robot Documentation**](src/robot/README.md)

---

### Video (`src/video/`)

Camera sources, PTZ streaming, and image acquisition.

**Key Components:**
- **Image Sources**: Abstractions for webcam, Spot cameras, PTZ
- **Fallback Logic**: Automatic fallback (PTZ → hand → pano → fisheye)
- **PTZ Streaming**: WebRTC configuration and stream setup

**Usage:**
```python
from src.video import SpotPTZImageClient

camera = SpotPTZImageClient(robot)  # Auto-fallback
frame_bgr = camera.get_frame()  # Returns BGR numpy array
```

📖 [**Full Video Documentation**](src/video/README.md)

---

### Visualization (`src/visualization/`)

Real-time overlay rendering for detections and tracking.

**Key Components:**
- **Overlay System**: Unified visualization coordinator
- **Helper Functions**: Drawing utilities for boxes, labels, depth

**Usage:**
```python
from src.visualization import UnifiedVisualization

viz = UnifiedVisualization()
viz_frame = viz.render(frame, detections)
cv2.imshow("Detections", viz_frame)
```

📖 [**Full Visualization Documentation**](src/visualization/README.md)

---

## Usage Examples

### Example 1: Basic Person Detection

```python
from src.robot import create_robot
from src.perception import YoloDetector
from src.video import SpotImageClient
import cv2

# Setup
robot = create_robot("192.168.80.3")
detector = YoloDetector(device="cuda")
camera = SpotImageClient(robot, "frontleft_fisheye_image")

# Detection loop
while True:
    frame = camera.get_frame()
    detections = detector.predict_batch([frame])[0]
    
    # Draw results
    for det in detections:
        x, y, w, h = det.bbox_xywh
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) == ord('q'):
        break
```

### Example 2: Behavior Loop

```python
from src.robot import create_robot
from src.behavior import ComfortModel, BehaviorExecutor
from src.perception import get_current_detection  # Your detection source

robot = create_robot("192.168.80.3")
comfort = ComfortModel(comfortable_distance=2.0)

with BehaviorExecutor(robot) as executor:
    while True:
        detection = get_current_detection()  # From perception
        behavior = comfort.update(detection)
        executor.execute_behavior(behavior)
        time.sleep(0.1)
```

### Example 3: Multi-Camera PTZ Tracking

```python
from src.perception.tracker import run_loop
from src.perception.config import RuntimeConfig, SURROUND_SOURCES

config = RuntimeConfig(
    mode="transform",         # Use SDK intrinsics
    enable_depth=True,        # Prioritize nearest person
    target_ptz_distance_m=2.5
)

# Track people and command PTZ (blocking loop)
run_loop(robot, image_client, ptz_client, config)
```

---

## Configuration

### Perception Settings (`src/perception/config.py`)

```python
# Camera sources
SURROUND_SOURCES = [
    "frontleft_fisheye_image",
    "frontright_fisheye_image"
]

# YOLO detection
MIN_CONFIDENCE = 0.4          # Detection threshold
YOLO_IMG_SIZE = 640           # Input resolution
YOLO_DEVICE = "cuda"          # "cuda" or "cpu"

# PTZ tracking
TARGET_PTZ_DISTANCE_M = 2.5   # Desired tracking distance
PTZ_MOVEMENT_THRESHOLD_DEG = 5.0  # Min angle change before PTZ update
```

### Behavior Settings (`src/behavior/planner.py`)

```python
# Comfort zones (meters)
COMFORTABLE_DISTANCE = 2.0
TOO_CLOSE_THRESHOLD = 1.5
TOO_FAR_THRESHOLD = 3.0

# Debouncing (seconds)
BEHAVIOR_CHANGE_COOLDOWN = 2.0
```

### Robot Settings (`src/robot/io.py`)

```python
# Authentication
BOSDYN_CLIENT_USERNAME = os.getenv("BOSDYN_CLIENT_USERNAME")
BOSDYN_CLIENT_PASSWORD = os.getenv("BOSDYN_CLIENT_PASSWORD")

# Connection
ROBOT_HOSTNAME = "192.168.80.3"
```

---

## Troubleshooting

### Common Issues

**Problem: `No module named 'src'`**

**Solution**: Run from workspace root or set PYTHONPATH:
```powershell
$env:PYTHONPATH = "$PWD"
python friendly_spot_main.py
```

**Problem: `ResourceAlreadyClaimedError`**

**Solution**: Another client holds lease (tablet, SDK script). Force take or return lease:
```python
with BehaviorExecutor(robot, force_take_lease=True):
    ...
```

**Problem: PTZ camera unavailable**

**Solution**: Fallback is automatic. Check logs for fallback camera:
```
WARNING: PTZ service unavailable, falling back to hand_color_image
```

**Problem: Slow YOLO inference**

**Solution**: Enable GPU or reduce image size:
```python
detector = YoloDetector(device="cuda", imgsz=416)  # Smaller resolution
```

**Problem: Robot doesn't move**

**Solution**: Check lease, E-Stop, and motor power:
```python
from src.robot import ensure_motors_on

with ManagedLease(robot):
    with ManagedEstop(robot):
        ensure_motors_on(robot)
        # Now send commands
```

### Logs

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Log files stored in `data/outputs/logs/`.

---

## Development

### Running Tests

```powershell
# All tests
python -m pytest tests/

# Import validation only
python tests/test_imports.py

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Code Style

Follow [PEP 8](https://peps.python.org/pep-0008/) and [docs/DOCUMENTATION_STYLE_GUIDE.md](docs/DOCUMENTATION_STYLE_GUIDE.md).

### Adding New Behaviors

1. Add to `BehaviorLabel` enum in `src/behavior/planner.py`
2. Implement in `BehaviorExecutor` in `src/behavior/executor.py`
3. Add decision logic in `ComfortModel.update()`

### Contributing

1. Create feature branch
2. Make changes with tests
3. Update relevant README.md
4. Submit pull request

---

## References

### Spot SDK
- [Boston Dynamics Spot SDK](https://dev.bostondynamics.com/)
- [Python Quickstart](https://dev.bostondynamics.com/docs/python/quickstart)
- [Frame Helpers](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/frame_helpers)
- [Robot Command](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/robot_command)

### Computer Vision
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)

### Robotics Theory
- [Proxemics (Social Distance)](https://en.wikipedia.org/wiki/Proxemics)
- [Human-Robot Interaction](https://www.springer.com/gp/book/9783319472973)

---

## License

MIT License - see LICENSE file for details.

## Acknowledgements

- **Boston Dynamics**: Spot SDK and examples (especially fetch.py)
- **Ultralytics**: YOLOv8 person detection
- **Dartmouth College**: COSC 69.15/169.15 course support
- **Professor Alberto Quattrini Li**: Course instruction and guidance

---

**Questions or Issues?** Open an issue on GitHub or contact the authors.
