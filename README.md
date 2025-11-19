# Friendly Spot: Socially-Aware Robot Interaction

**ENGS 69.15 - Robotics Perception, Dartmouth College**

**Authors:** Thor Lemke, Sally Hyun Hahm, Matteo Corrado

A perception and behavior system enabling Boston Dynamics Spot to recognize people, interpret social cues, and respond with appropriate behaviors. Integrates computer vision (facial recognition, emotion detection, pose estimation, gesture recognition) with a comfort-based behavior model to enable natural human-robot interaction.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Friendly Spot Pipeline                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  ┌──────────┐              ┌──────────────┐            ┌─────────────┐
  │  Webcam  │              │ Spot PTZ Cam │            │ Spot Cameras│
  │(Dev/Test)│              │ (ImageClient)│            │  (Surround) │
  └─────┬────┘              └──────┬───────┘            └──────┬──────┘
        │                          │                           │
        └──────────────┬───────────┴───────────┬───────────────┘
                       ▼                       ▼
              ┌─────────────────┐   ┌──────────────────────┐
              │ Video Sources   │   │ People Observer      │
              │  (Abstraction)  │   │ (YOLO Person Track)  │
              └────────┬────────┘   └──────────┬───────────┘
                       │                       │
                       ▼                       │
              ┌─────────────────┐              │
              │   Perception    │              │
              │    Pipeline     │◄─────────────┘
              │                 │
              │ • Face ID       │
              │ • Emotion       │
              │ • Pose/Action   │
              │ • Gesture       │
              │ • Distance      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Comfort Model  │
              │  (ML Behavior   │
              │    Planner)     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    Behavior     │
              │    Executor     │
              │                 │
              │ • Approach      │
              │ • Back Away     │
              │ • Sit/Stand     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Spot Robot     │
              │  (Boston        │
              │   Dynamics SDK) │
              └─────────────────┘
```

---

## Key Features

### Perception Pipeline
- **Facial Recognition:** Identify known individuals using LBPH recognizer (OpenCV)
- **Emotion Detection:** 7-class emotion recognition via DeepFace (happy, sad, angry, surprise, fear, disgust, neutral)
- **Pose Estimation:** MediaPipe-based body pose and action classification (standing, sitting, waving, etc.)
- **Gesture Recognition:** Hand gesture detection (open hand, closed fist)
- **Distance Estimation:** Approximate human distance from pose landmark spans

### Behavior Planning
Comfort-based model maps perception inputs to robot behaviors:
- `GO_CLOSE` / `GO_CLOSE_SLOWLY` - Approach when person is comfortable
- `STAY` - Maintain position (neutral comfort)
- `BACK_AWAY` / `BACK_AWAY_SLOWLY` - Retreat when person is uncomfortable
- `SIT` - De-escalate presence (low comfort)

### People Observer
Real-time person tracking with PTZ following:
- GPU-accelerated YOLO detection on 5 surround cameras
- 3D bearing estimation using SDK intrinsics and transforms
- Automatic PTZ camera aiming at tracked individuals

---

## Project Structure

```
friendly_spot/
├── friendly_spot_main.py        # Main pipeline orchestrator
├── robot_io.py                  # Unified Spot SDK connection/client management
├── video_sources.py             # Video source abstraction (webcam, PTZ, WebRTC)
├── run_pipeline.py              # Perception pipeline (face, emotion, pose, gesture)
├── behavior_planner.py          # Comfort model and behavior decision logic
├── behavior_executor.py         # Robot command execution (sit, stand, walk)
│
├── people_observer/             # Person detection and PTZ tracking
│   ├── app.py                   # CLI entry point
│   ├── tracker.py               # Main detection/tracking loop
│   ├── detection.py             # YOLO detection wrapper
│   ├── cameras.py               # Multi-camera image fetching
│   ├── geometry.py              # 3D bearing/transform calculations
│   └── ptz_control.py           # PTZ aiming commands
│
├── Facial Recognition/          # Face recognition and emotion models
│   ├── streamlinedCombinedMemoryAndEmotion.py
│   ├── streamlinedRuleBasedEstimation.py
│   └── trainMemory.py           # LBPH face recognizer trainer
│
├── scripts/                     # Utility scripts
│   └── download_models.py       # Download YOLO/MediaPipe models
│
├── requirements.txt             # Python dependencies
├── INSTALL.md                   # Platform-specific installation guide
├── PIPELINE_README.md           # Detailed pipeline documentation
└── SECURITY.md                  # Security and credential management
```

---

## Quick Start

### Prerequisites
- Python 3.9 or 3.10 (Spot SDK requirement)
- Boston Dynamics Spot SDK v5.0.1.2
- Robot credentials configured in environment variables

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/friendly_spot.git
cd friendly_spot

# 2. Install Spot SDK wheels
pip install ../spot-sdk/prebuilt/*.whl

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials (see SECURITY.md)
cp .env.template .env
# Edit .env with your robot credentials
```

### Run Examples

```bash
# Full pipeline with webcam (development/testing)
python friendly_spot_main.py --webcam

# Full pipeline with robot PTZ camera
python friendly_spot_main.py --robot <ROBOT_IP>

# Enable verbose debugging
python friendly_spot_main.py --robot <ROBOT_IP> --verbose

# People observer (YOLO tracking + PTZ control)
python -m people_observer.app <ROBOT_IP> --visualize

# Run without executing robot commands (perception only)
python friendly_spot_main.py --robot <ROBOT_IP> --no-execute
```

---

## Documentation

- **[INSTALL.md](INSTALL.md)** - Platform-specific installation instructions (Windows, macOS, Linux)
- **[PIPELINE_README.md](PIPELINE_README.md)** - Detailed pipeline architecture and component descriptions
- **[SECURITY.md](SECURITY.md)** - Credential management and security best practices
- **[people_observer/README.md](people_observer/README.md)** - Person detection and PTZ tracking system

---

## Technical Requirements

### Software
- **Python:** 3.9 or 3.10 only
- **Spot SDK:** 5.0.1.2 (included in workspace)
- **Core Libraries:**
  - OpenCV (4.x) - Computer vision
  - TensorFlow (2.15+) - DeepFace emotion detection
  - MediaPipe (0.10+) - Pose and hand tracking
  - Ultralytics (8.x) - YOLO detection
  - DeepFace - Facial emotion analysis

### Hardware
- Boston Dynamics Spot robot with Spot CAM (PTZ camera)
- Development machine: x86_64 CPU (GPU optional but recommended for YOLO)
- Network: Local Wi-Fi connection to robot (latency < 50ms recommended)

---

## Development Notes

### Authentication
Robot credentials are managed via environment variables (`.env` file). Never commit credentials to version control. See `SECURITY.md` for setup instructions.

### Cross-Platform Support
The codebase is tested on Windows, macOS, and Linux:
- Platform-specific webcam backends (DirectShow, AVFoundation, V4L2)
- Windows-compatible signal handling (SIGINT only, no SIGTERM)
- TensorFlow GPU memory growth for all platforms
- Path handling uses `os.path.join()` for portability

### Code Organization
- **Unified robot I/O:** All robot connections via `robot_io.py` module
- **Context managers:** Automatic lease and E-Stop management
- **Lazy client initialization:** Clients created only when needed
- **No wrapper functions:** Direct usage of `robot_io` throughout codebase

---

## Acknowledgments

Built with the Boston Dynamics Spot SDK. Thanks to the teaching staff of ENGS 69.15 at Dartmouth College for guidance and support.

### References
- [Boston Dynamics Spot SDK Documentation](https://dev.bostondynamics.com)
- [DeepFace Library](https://github.com/serengil/deepface)
- [MediaPipe Solutions](https://developers.google.com/mediapipe)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

---

## License

This project is for educational purposes as part of ENGS 69.15 at Dartmouth College. Boston Dynamics SDK components are subject to the Boston Dynamics Software Development Kit License.
