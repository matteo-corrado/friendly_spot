# Friendly Spot Full Pipeline Guide

## Overview

The Friendly Spot pipeline integrates computer vision perception with behavior-based robot control, enabling Spot to interact naturally with humans using facial recognition, emotion detection, pose estimation, and gesture recognition.

**Pipeline Flow:**
```
PTZ Camera → Perception → Comfort Model → Behavior Decision → Robot Command
   (or Webcam)   (CV Models)   (ML Model)    (BehaviorLabel)    (SDK)
```

## Architecture

### 1. Video Sources (`video_sources.py`)

Three video source options:

#### **WebcamSource** - Local Camera (Development)
- USB or built-in webcam
- Windows DirectShow backend for reliability
- Use: Development, testing without robot

#### **SpotPTZImageClient** - Robot PTZ via ImageClient (RECOMMENDED)
- **Performance:** 20-30 fps typical, ~50-100ms latency per frame
- **Advantages:**
  - Synchronous API (no asyncio complexity)
  - Includes camera intrinsics and depth data
  - Direct JPEG fetch from robot
  - Proven in `people_observer` detection pipeline
- **Best for:** Real-time perception pipeline (this use case)

#### **SpotPTZWebRTC** - Robot PTZ via WebRTC (TODO)
- **Performance:** 15-30 fps, ~100-200ms startup + ~50ms per frame
- **Advantages:**
  - H.264 streaming for low bandwidth
  - Good for remote internet streaming
- **Disadvantages:**
  - Asynchronous (requires separate thread + event loop)
  - No camera intrinsics or metadata
  - More complex connection management
- **Best for:** Remote viewing, bandwidth-constrained scenarios
- **Status:** Not yet implemented (see TODOs in `video_sources.py`)

**Why ImageClient over WebRTC for perception?**
- Local perception needs low latency and high frame rate
- Synchronous API fits naturally with perception loop structure
- No need for H.264 encode/decode overhead on local network
- Camera intrinsics enable advanced features (depth, calibration)

### 2. Perception Pipeline (`run_pipeline.py`)

Extracts multi-modal perception data from video frames:

**Modules:**
- **Pose Estimation** (`streamlinedRuleBasedEstimation.py`)
  - MediaPipe Pose landmarks (33 body keypoints)
  - Action detection: standing, sitting, waving, etc.
  - Distance estimation from landmark span
  
- **Face Recognition** (`streamlinedCombinedMemoryAndEmotion.py`)
  - Haar Cascade face detection
  - LBPH face recognizer (trainable on dataset)
  - Person identification with confidence
  
- **Emotion Detection** (DeepFace)
  - 7 emotions: happy, sad, angry, surprise, fear, disgust, neutral
  - TensorFlow backend (limited threading on Windows)
  
- **Gesture Recognition** (MediaPipe Hands)
  - Hand landmark detection
  - Coarse gesture classification: open_hand, closed_fist, unknown

**Output:** `PerceptionInput` dataclass with all modalities

**Cross-Platform Compatibility:**
- Platform-specific webcam backends (DirectShow on Windows, AVFoundation on macOS, V4L2 on Linux)
- TensorFlow GPU memory growth configuration (all platforms)
- MediaPipe and OpenCV resources cleaned up properly
- Signal handling works on Windows (SIGINT) and Unix (SIGINT + SIGTERM)
- All file paths use `os.path.join()` for portability

**DeepFace Performance Optimizations:**
- **Model pre-building:** Loads models once at startup (10-50x faster)
- **GPU memory growth:** Prevents OOM errors, enables efficient GPU usage
- **Silent mode:** Reduces logging I/O overhead
- **Skip redundant detection:** Uses Haar cascade results instead of re-detecting

### 3. Behavior Planning (`behavior_planner.py`)

**ComfortModel** predicts human comfort level and selects appropriate robot behavior:

```
Perception → Comfort Score (0.0-1.0) → BehaviorLabel
```

**BehaviorLabel Options:**
- `GO_CLOSE` - Approach person (comfort > 0.8)
- `GO_CLOSE_SLOWLY` - Approach cautiously (comfort 0.6-0.8)
- `STAY` - Maintain position (comfort 0.4-0.6)
- `BACK_AWAY_SLOWLY` - Retreat cautiously (comfort 0.2-0.4)
- `BACK_AWAY` - Retreat quickly (comfort < 0.2)
- `SIT` - Sit down (person sitting or uncomfortable)

**Model:** Rule-based decision tree (TODO: train ML model on real data)

### 4. Behavior Execution (`behavior_executor.py`)

Translates `BehaviorLabel` into Spot SDK robot commands:

**Responsibilities:**
- Lease acquisition and keep-alive
- Software E-Stop registration (safety)
- Command execution with error handling
- State tracking to avoid redundant commands

**Current Status:**
- `SIT` and `STAY` (stand) commands implemented
- Locomotion commands stubbed with TODOs
- Full implementation requires distance/velocity control

**TODOs:**
- Implement `_walk_forward()` with distance control
- Implement `_walk_backward()` with distance control
- Add command feedback and verification
- Tune motion parameters (speed, distance)

### 5. Main Integration (`friendly_spot_main.py`)

Orchestrates complete pipeline:

```python
1. Connect to robot (authenticate, time sync)
2. Create video source (ImageClient/WebRTC/webcam)
3. Initialize perception pipeline
4. Initialize comfort model
5. Acquire robot lease (if execution enabled)
6. Loop:
   a. Read perception from video
   b. Predict comfort and behavior
   c. Execute robot command
   d. Rate limit to target Hz
7. Cleanup (release lease, close resources)
```

**Features:**
- Graceful shutdown (Ctrl+C handling)
- Rate limiting (default 5 Hz)
- Perception-only mode (`--no-execute`)
- Detailed logging with periodic status updates

## Installation

### System Requirements

**Python Version:**
- **Python 3.9 or 3.10 ONLY** (recommended: 3.10)
- Spot SDK requires Python 3.7-3.10
- MediaPipe requires Python 3.8-3.11
- **Intersection: Python 3.9-3.10 is the only compatible range**
- Python 3.11+ NOT SUPPORTED
- Check version: `python --version`

**Operating Systems:**
- Windows 10/11 (x86_64 only, no ARM)
- macOS 11+ (Intel or Apple Silicon)
- Linux (Ubuntu 20.04+, x86_64 or ARM64)

**Hardware:**
- Webcam for development mode
- Boston Dynamics Spot robot for full pipeline
- Optional: NVIDIA GPU (CUDA) or Apple Silicon (Metal) for GPU acceleration

### Install Dependencies

**1. Create virtual environment:**

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install requirements:**

```bash
# Master requirements (all modules)
pip install -r requirements.txt

# Or install per-module (if only using subset)
pip install -r "Facial Recognition/requirements.txt"
pip install -r people_observer/requirements.txt
```

**3. Platform-specific setup:**

**macOS Apple Silicon (M1/M2/M3):**
```bash
# For GPU acceleration with TensorFlow
pip install tensorflow-metal
```

**Linux ARM64 (Jetson, Raspberry Pi):**
```bash
# Custom TensorFlow build required
# See: https://github.com/tensorflow/tensorflow/releases
```

**Windows/Linux with NVIDIA GPU:**
```bash
# TensorFlow 2.15 requires CUDA 12.x + cuDNN 8.9
# Download from: https://developer.nvidia.com/cuda-toolkit
```

**4. Pre-download models (optional for offline use):**

```bash
python scripts/download_models.py
```

This downloads:
- DeepFace emotion models (~500MB)
- YOLOv8n weights (~6MB)
- MediaPipe model files

### Verify Installation

```bash
# Check core dependencies
python -c "import cv2, mediapipe, tensorflow, deepface; print('OK')"

# Check Spot SDK
python -c "import bosdyn.client; print('Spot SDK OK')"
```

## Usage

### Development Mode (Webcam)

Test perception without robot:

```bash
# All platforms
python friendly_spot_main.py --webcam
```

### Robot Mode (ImageClient - RECOMMENDED)

Full pipeline with PTZ camera:

```bash
# Set credentials in your environment (see Installation Guide)
# Then run:
python friendly_spot_main.py --robot 192.168.80.3

# Or use CLI flags (credentials not echoed in terminal history if set via prompt):
python friendly_spot_main.py --robot 192.168.80.3 --user <username> --password <password>
```

**Security Note:** Never commit credentials to version control. Set them in environment variables outside of your repository.

### Options

```
--robot HOSTNAME        Robot IP/hostname (required unless --webcam)
--webcam               Use local webcam (development mode)
--user USERNAME        Robot username (or use environment variable)
--password PASSWORD    Robot password (or use environment variable)
--ptz-source NAME      PTZ camera name (default: "ptz")
--rate HZ              Loop rate in Hz (default: 5.0)
--no-execute           Perception only, no robot commands
--webrtc              Use WebRTC instead of ImageClient (TODO)
--visualize           Show CV windows (TODO)
```

### Testing Strategy

**1. Test webcam mode first** (verify perception works on your platform)
```bash
python friendly_spot_main.py --webcam
```
Expected: Logs show pose/face/emotion/gesture detection every 0.2s

**2. Test robot connection** (verify authentication and lease)
```bash
python friendly_spot_main.py --robot IP --no-execute
```
Expected: Logs show "Acquired robot lease" without errors

**3. Test PTZ frame fetch** (verify ImageClient works)
```bash
python friendly_spot_main.py --robot IP --no-execute
```
Expected: Perception data in logs (pose coordinates, emotions, etc.)

**4. Test behavior execution** (verify commands work)
```bash
# Start with robot standing, should execute sit/stand commands
python friendly_spot_main.py --robot IP
```
Expected: Robot responds to behavior commands based on perception

## Troubleshooting

### Platform-Specific Issues

**Windows:**
- **MediaPipe hangs:** Ensure Python 3.9-3.11 (not 3.12+)
- **Webcam not opening:** DirectShow backend is used automatically
- **CUDA not found:** TensorFlow 2.15 requires CUDA 12.x + cuDNN 8.9
- **"NumPy 1.x required" error:** Update opencv-python to 4.10.0.84

**macOS:**
- **No GPU acceleration:** Install `pip install tensorflow-metal`
- **Webcam permission denied:** Grant Terminal camera access in System Settings
- **MediaPipe slow on M1/M2:** Try Python 3.10 (best Apple Silicon support)
- **"No module named cv2":** Ensure opencv-contrib-python (not opencv-python-headless)

**Linux:**
- **Webcam not found:** Check V4L2 device permissions (`ls -l /dev/video*`)
- **CUDA not available:** Verify CUDA toolkit installed (`nvcc --version`)
- **Protobuf version conflict:** Pin protobuf<4.0 in requirements.txt
- **ARM64 TensorFlow missing:** Use custom build or run on CPU

### Common Errors

**"Protobuf version conflict"**
```bash
# If you see errors about protobuf version conflicts:
# The requirements.txt uses a flexible range: >=3.19.4,<5.0,!=4.24.0,!=5.28.0
# This works with Spot SDK and TensorFlow 2.15

# Force reinstall if needed:
pip uninstall -y protobuf
pip install "protobuf>=3.19.4,<5.0,!=4.24.0,!=5.28.0"
```

**"Could not build model for Emotion"**
```bash
# Pre-download models
python scripts/download_models.py

# Or run once with internet, models cache to ~/.deepface/weights/
```

**"MediaPipe Hands/Pose not found"**
```bash
# Reinstall MediaPipe
pip uninstall mediapipe
pip install mediapipe==0.10.14
```

**"Robot is estopped"**
```bash
# Check E-Stop status
# Release E-Stop in tablet app or use estop_gui
```

**"Lease acquisition failed"**
```bash
# Another process may have the lease
# Check for other running scripts or tablet control
```

**"TensorFlow GPU not detected"**
```bash
# Windows/Linux: Verify CUDA installation
nvidia-smi

# macOS: Install tensorflow-metal
pip install tensorflow-metal

# Test GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Dependencies

### Master Requirements
See `requirements.txt` - includes all modules below.

### Per-Module Requirements

**Facial Recognition** (`Facial Recognition/requirements.txt`):
- opencv-contrib-python==4.10.0.84 (LBPH face recognizer)
- mediapipe==0.10.14 (pose estimation)
- deepface==0.0.92 (emotion detection)
- tensorflow==2.15.0

**People Observer** (`people_observer/requirements.txt`):
- ultralytics>=8.0.0 (YOLOv8 person detection)
- torch>=2.0.0 (PyTorch backend)
- bosdyn-client==5.0.1.2 (Spot SDK)

**Behavioral** (`Behavioral/requirements.txt`):
- ultralytics>=8.0.0 (YOLO tracking)
- tensorflow==2.15.0 (Network Compute Bridge)
- bosdyn-client==5.0.1.2

**Emotion Recognition** (`EmotionRecognition/requirements.txt`):
- deepface==0.0.92
- tensorflow==2.15.0
- opencv-python==4.10.0.84

### Optional (for WebRTC)
```
aiortc>=1.3.0
av>=10.0.0
```

Install all:
```powershell
pip install -r requirements.txt
pip install -r people_observer/requirements_webrtc.txt  # Optional
```

## File Structure

```
friendly_spot/
├── friendly_spot_main.py           # Main entry point (NEW)
├── video_sources.py                # Video source abstraction (NEW)
├── behavior_executor.py            # Robot command execution (NEW)
├── run_pipeline.py                 # Perception pipeline (MODIFIED)
├── behavior_planner.py             # Comfort model
├── behavior_demo.py                # Behavior demos
│
├── Facial Recognition/             # CV modules
│   ├── streamlinedRuleBasedEstimation.py      # Pose
│   ├── streamlinedCombinedMemoryAndEmotion.py # Face + emotion
│   └── trainMemory.py
│
├── people_observer/                # Detection + PTZ control
│   ├── ptz_stream.py              # WebRTC streaming (for future)
│   ├── ptz_webrtc_client.py       # WebRTC client
│   └── ...
│
└── requirements.txt                # Python dependencies
```

## Integration with Existing Code

### `people_observer` Detection Pipeline

The detection pipeline (`people_observer/app.py`) runs independently:
- Detects people in fisheye cameras
- Aims PTZ at nearest/largest person
- Can trigger PTZ stream on detection (TODO)

**Integration TODO:**
- Detection pipeline triggers `PtzStream.start()` when person detected
- PTZ stream feeds frames to `friendly_spot_main.py` perception
- Facial recognition results annotate detections
- Close loop: perception influences PTZ targeting

### Voice Control (`dartmouth_spot_capstone`)

Voice intent parsing can trigger behaviors:
- Intent: "come here" → Force `GO_CLOSE` behavior
- Intent: "sit down" → Force `SIT` behavior
- Intent: "back away" → Force `BACK_AWAY` behavior

**Integration TODO:**
- Wire `spot_dispatch.py` to `behavior_executor.py`
- Override comfort model decisions with voice commands
- Add intent → BehaviorLabel mapping

## Known Issues & Limitations

### Cross-Platform Compatibility (Windows, macOS, Linux)
- Fixed: `cv2.cvtColot` typo causing crashes
- Fixed: MediaPipe resources not released (memory leaks)
- Fixed: TensorFlow GPU memory configuration (works on all platforms)
- Fixed: Platform-specific webcam backends (DirectShow/AVFoundation/V4L2)
- Fixed: Signal handling (SIGINT everywhere, SIGTERM on Unix only)
- Optimized: DeepFace model pre-building (10-50x faster)
- Paths: Using `os.path.join()` for cross-platform compatibility

### Current TODOs

**High Priority:**
1. Implement locomotion commands in `behavior_executor.py`
2. Test full pipeline on Windows with robot
3. Tune behavior thresholds in `ComfortModel`
4. Add visualization mode (show CV windows)

**Medium Priority:**
5. Implement WebRTC video source in `video_sources.py`
6. Train ML comfort model on real interaction data
7. Integrate with `people_observer` detection triggering
8. Add voice command integration

**Low Priority:**
9. Add telemetry logging (comfort over time, behavior history)
10. Implement gesture-based commands
11. Add multi-person tracking support
12. Create configuration file (YAML/JSON)

## Performance Notes

### Perception Pipeline Speed
- **Webcam:** 5-10 fps typical (limited by DeepFace)
- **PTZ ImageClient:** 5-10 fps typical (same bottleneck)
- **Bottleneck:** DeepFace emotion detection (~100-200ms per face with optimizations)

**Applied Optimizations:**
- **Model pre-building:** Loads models once (saves 2-5s per call)
- **GPU memory growth:** Enables efficient GPU usage
- **Skip redundant detection:** Uses Haar cascade results
- **Silent mode:** Reduces logging overhead

**Additional Optimization Ideas:**
- Cache emotion results for same face across frames (track by face ID)
- Run emotion detection every N frames (emotions change slowly)
- Use face embeddings cache for recognition (compute once, compare many times)
- Reduce image resolution before emotion detection (e.g., 48x48 for emotion model)
- Batch process multiple faces simultaneously if available

### Robot Command Latency
- **Lease acquire:** ~100-200ms
- **Stand command:** ~2-3 seconds to complete
- **Sit command:** ~2-3 seconds to complete
- **Locomotion:** ~1-2 seconds per command

**Design Note:** 5 Hz perception loop allows ~200ms per iteration,
which fits well with command latency. Faster loops would queue commands.

## Troubleshooting

### Pipeline hangs on startup
- **Cause:** DeepFace loading models on first call (~2-5 seconds)
- **Fix:** Applied automatically (model pre-building at initialization)
- **Verify:** Check logs for "Pre-building DeepFace models" and "Emotion model loaded"

### "Failed to open webcam"
- **Cause:** Another app using camera, or wrong device index
- **Fix:** Close other camera apps, try `--device 1` or `--device 2`

### "Failed to acquire lease"
- **Cause:** Another client has lease, or E-Stop active
- **Fix:** Release lease from other clients, check E-Stop

### "Camera source 'ptz' not found"
- **Cause:** Wrong camera name or Spot CAM not available
- **Fix:** List sources with `python -m people_observer.app --list-sources`
        Use `--ptz-source` to specify correct name

### Poor perception performance (<3 fps)
- **Cause:** DeepFace emotion detection too slow
- **Fix:** Reduce loop rate `--rate 2`, or skip emotion detection

## Future Enhancements

1. **Smart PTZ Control**
   - `people_observer` detects person → aims PTZ → starts stream
   - `friendly_spot_main.py` receives stream → runs perception
   - Facial recognition identifies person → updates PTZ target
   
2. **Multi-Person Interaction**
   - Track multiple people simultaneously
   - Prioritize attention based on proximity, familiarity, emotion
   - Cycle PTZ between people
   
3. **Learning from Interaction**
   - Log perception + behavior + outcomes
   - Train ML comfort model on real data
   - Personalize behavior to individual preferences
   
4. **Voice + Vision Fusion**
   - Combine voice intent with visual perception
   - "Come here" + neutral emotion → approach slowly
   - "Come here" + happy emotion → approach quickly

## Contact

Questions? See individual file headers for authors and course info.

**Course:** COSC 69.15/169.15 at Dartmouth College, 25F  
**Instructor:** Professor Alberto Quattrini Li  
**Team:** Thor Lemke, Sally Hyun Hahm, Matteo Corrado
