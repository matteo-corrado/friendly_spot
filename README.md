# Friendly Spot: Socially-Aware Robot Interaction

**COSC 69.15/169.15 - Robotics Perception and Interaction, Dartmouth College**

**Authors:** Thor Lemke, Sally Hyun Hahm, Matteo Corrado  
**Version:** 2.0.0 (Modular Architecture)  
**Date:** January 2025

A perception and behavior system enabling Boston Dynamics Spot to detect people, track their movements with PTZ camera, estimate social distance, and respond with proximity-aware behaviors. Uses YOLOv11 person detection, depth-based prioritization, and a comfort model for natural human-robot interaction.

## Overview

This codebase was designed to implement a simple HRI pipeline on Spot with the intention of making it somewhat socially aware and reactive. The main code to run is friendly_spot_main, and all important dependencies should have been placed in src but there may still be a couple of issues with imports from other files that we haven't quite sorted out, so we reccommend running this code in this environment. Much of the code, particularly implementing on spot, was created with Claude AI's assistance, covering for gaps in our knowledege and enabling us to code more effectively under large time pressure. This does not change that the code is less handwritten than we would like, but we still believe we achieved a significant implementation in a very short time by effectively using the resources we had and crediting them appropriately. All high level design was done by us beforehand.

## Features

### Perception System
- **YOLO Person Detection**: GPU-accelerated YOLOv11 detection across multiple cameras
- **Multi-Camera Fusion**: Batch processing of surround fisheye cameras
- **Depth Estimation**: Prioritize nearest person using depth sensors and segmentation masks
- **PTZ Tracking**: Automatic pan-tilt-zoom camera control to keep person in view
- **Coordinate Transforms**: Precise bearing calculation using SDK camera intrinsics

### Behavior Planning
- **Comfort Model**: Proximity-based social distance reasoning
- **Behavior States**: GO_CLOSE, BACK AWAY, STAND, SIT, etc
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
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11m-seg.pt -o data/models/yolo11m-seg.pt

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
