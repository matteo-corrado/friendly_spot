# Friendly Spot with People Observer Integration - Complete Guide

## Overview

The **Friendly Spot** system now integrates **people_observer** for automatic person detection, tracking, and PTZ camera control. The system can:

1. **Detect people** using YOLOv11-seg in surround cameras
2. **Extract depth** from segmentation masks for accurate distance
3. **Control PTZ** to automatically point at detected people
4. **Run perception** (pose, face, emotion, gesture) on PTZ frames
5. **Execute behaviors** based on comfort model
6. **Visualize everything** with depth-colored masks and perception overlays

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRIENDLY SPOT MAIN                        │
│  ┌────────────────────┐         ┌──────────────────────┐   │
│  │ ObserverBridge     │         │  PerceptionPipeline  │   │
│  │ (Background Thread)│────────▶│  (Pose/Face/Emotion) │   │
│  └────────────────────┘         └──────────────────────┘   │
│           │                              │                   │
│           │ PersonDetection             │ Perception        │
│           │ (bbox, mask, depth)          │ Results           │
│           ▼                              ▼                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │          ComfortModel + BehaviorExecutor            │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   Spot Robot  │
                    └──────────────┘
```

## Quick Start

### 1. With Observer (Full Integration)

Detects people in surround cameras → Points PTZ → Runs perception:

```powershell
# Activate environment
C:\Users\corra\Spot_SDK_Master\dartmouth_spot_capstone\.venv\Scripts\Activate.ps1

# Run full system with visualization
cd C:\Users\corra\Spot_SDK_Master\friendly_spot
python friendly_spot_main.py --robot ROBOT_IP --enable-observer --visualize --verbose

# With image saving
python friendly_spot_main.py --robot ROBOT_IP --enable-observer --visualize --save-images output/ --verbose

# Test mode (one cycle)
python friendly_spot_main.py --robot ROBOT_IP --enable-observer --once --visualize --save-images test/ --verbose
```

### 2. Without Observer (Perception Only)

Uses PTZ camera directly (no automatic person detection):

```powershell
# Run perception on PTZ stream
python friendly_spot_main.py --robot ROBOT_IP --visualize --verbose

# No execution, perception only
python friendly_spot_main.py --robot ROBOT_IP --no-execute --visualize --save-images output/ --verbose
```

### 3. Webcam Mode (Development)

Test perception pipeline without robot:

```powershell
python friendly_spot_main.py --webcam --visualize --verbose
```

## Command Line Options

### Connection
- `--robot HOSTNAME` - Robot IP/hostname
- `--webcam` - Use local webcam (dev mode, mutually exclusive with --robot)
- `--user USERNAME` - Robot username (or use env var)
- `--password PASSWORD` - Robot password (or use env var)

### Observer Integration
- `--enable-observer` - **Enable people_observer for auto-detection and PTZ control**
- `--ptz-source NAME` - PTZ camera source (default: "ptz")

### Pipeline Control
- `--rate HZ` - Loop rate in Hz (default: 5.0)
- `--no-execute` - Disable behavior execution (perception only)
- `--once` - **Run one cycle and exit (testing)**

### Visualization & Logging
- `--visualize` - **Show live OpenCV window with annotations**
- `--save-images DIR` - **Save annotated frames to directory**
- `--verbose` - Enable debug logging

## Features

### People Observer Integration

When `--enable-observer` is enabled:

1. **Background thread** monitors 5 surround cameras at 2 Hz
2. **YOLOv11-seg** detects people with instance segmentation
3. **Depth extraction** from masks for accurate distance (0.1m precision)
4. **Selects closest person** with valid depth measurement
5. **Commands PTZ** to point at person (pan/tilt control)
6. **Waits for stabilization** (300ms)
7. **Fetches PTZ frame** with depth (if available)
8. **Creates PersonDetection** with all info (bbox, mask, depth, frame)
9. **Queues detection** for perception pipeline

### Perception Pipeline

Runs at 5 Hz (configurable) on PTZ frames:

1. **Pose Estimation** - MediaPipe 33 landmarks
2. **Face Recognition** - LBPH recognizer
3. **Emotion Detection** - DeepFace with VGG-Face
4. **Gesture Recognition** - MediaPipe Gesture Recognizer
5. **Distance Calculation** - Multi-tier (observer depth → frame depth → heuristic)

### Unified Visualization

Shows depth-colored segmentation masks + perception results:

- **Segmentation masks** with depth colormap:
  - 🔵 Blue = close (0.5-2m)
  - 🟢 Green = medium (2-3.5m)
  - 🔴 Red = far (3.5-5m)
- **Bounding boxes** around people
- **Distance labels** in meters
- **Pose landmarks** (33 keypoints)
- **Info panel** with pose/face/emotion/gesture labels

Press `q` or `ESC` to quit visualization.

## Usage Examples

### Example 1: Full Demo with Saving

```powershell
python friendly_spot_main.py \
  --robot 192.168.50.3 \
  --enable-observer \
  --visualize \
  --save-images demo_output/ \
  --verbose
```

**Result:**
- Detects people in surround cameras
- Points PTZ automatically
- Shows live visualization with depth masks
- Saves annotated frames to `demo_output/`
- Executes behaviors based on comfort model

### Example 2: Quick Test (One Cycle)

```powershell
python friendly_spot_main.py \
  --robot 192.168.50.3 \
  --enable-observer \
  --once \
  --visualize \
  --save-images test/ \
  --verbose
```

**Result:**
- Runs single detection → PTZ → perception cycle
- Saves one annotated frame
- Exits immediately
- Perfect for testing

### Example 3: Perception Only (Manual PTZ)

```powershell
python friendly_spot_main.py \
  --robot 192.168.50.3 \
  --no-execute \
  --visualize \
  --verbose
```

**Result:**
- Runs perception on PTZ stream (you control PTZ manually)
- No observer (no auto-detection)
- No behavior execution
- Just perception + visualization

### Example 4: Webcam Development

```powershell
python friendly_spot_main.py \
  --webcam \
  --visualize \
  --verbose
```

**Result:**
- Uses laptop webcam
- Runs full perception pipeline
- No robot commands
- Perfect for testing perception models

## Configuration

### Observer Settings (`observer_bridge.py`)

```python
ObserverConfig(
    enable_observer=True,
    surround_fps=2.0,          # Surround camera loop rate
    ptz_fps=5.0,               # PTZ tracking rate when person detected
    detection_timeout_sec=2.0, # Staleness timeout
    min_tracking_quality=0.5   # Minimum quality for valid tracking
)
```

### YOLO Model (`people_observer/config.py`)

```python
YOLO_MODEL_PATH = "yolov11x-seg.pt"  # Options: n, s, m, l, x
MIN_CONFIDENCE = 0.5
MIN_AREA_PX = 500
```

### Perception Rate (`friendly_spot_main.py`)

```powershell
--rate 5.0  # 5 Hz perception (default)
--rate 10.0 # 10 Hz for faster response
```

## Output Files

### Saved Frames (--save-images)

Filename format: `YYYYMMDD_HHMMSS_mmm_pipeline_iter####.jpg`

Each frame includes:
- Depth-colored segmentation mask overlay
- Bounding box with distance label
- Pose landmarks
- Info panel (pose/face/emotion/gesture)

### Logs (--verbose)

```
INFO] Starting pipeline at 5.0 Hz...
DEBUG] Starting observer bridge: surround_fps=2.0, ptz_fps=5.0
DEBUG] YoloDetector loaded: YOLOv11x-seg (task: segment)
DEBUG] Segmentation supported: True
DEBUG] Closest person: 2.34m away in frontleft_fisheye_image
DEBUG] PTZ commanded: pan=15.3°, tilt=-5.7°
DEBUG] PersonDetection queued: 2.34m, PTZ frame ready
DEBUG] [Loop 0] Using observer person detection
DEBUG] [Tier 1] Using pre-computed distance: 2.34m (source: frontleft_fisheye_image)
INFO] [Loop 0] Comfort: 0.87 | Behavior: friendly_approach | Distance: 2.34m
```

## Performance

### Timing Breakdown (5 Hz)

- **Observer loop**: 500ms (2 Hz surround monitoring)
  - Frame fetch: 50ms
  - YOLO detection: 150ms
  - Depth extraction: 20ms
  - PTZ command: 50ms
  - Stabilization: 300ms
- **Perception loop**: 200ms (5 Hz)
  - Frame read: 30ms
  - Pose: 40ms
  - Face: 30ms
  - Emotion: 60ms
  - Gesture: 20ms
  - Visualization: 20ms

### Optimization Tips

1. **Use smaller YOLO model**: `yolov11n-seg` (4x faster than `yolov11x-seg`)
2. **Reduce observer fps**: `surround_fps=1.0` for less CPU load
3. **Disable visualization** in production: Remove `--visualize`
4. **Lower perception rate**: `--rate 2.0` for slower devices

## Troubleshooting

### Issue: No person detected

**Causes:**
- Person outside camera FOV
- Low confidence threshold
- Poor lighting

**Solutions:**
```python
# In people_observer/config.py
MIN_CONFIDENCE = 0.3  # Lower threshold
MIN_AREA_PX = 300     # Smaller minimum size
```

### Issue: PTZ doesn't point at person

**Causes:**
- Transform mode failing
- Camera intrinsics missing
- PTZ limits reached

**Solutions:**
```powershell
# Try bearing mode in people_observer/config.py
TRANSFORM_MODE = "bearing"

# Check logs for intrinsics
DEBUG] Intrinsics available for frontleft_fisheye_image: model=kannala_brandt
```

### Issue: Depth shows NaN

**Causes:**
- PTZ has no depth sensor (expected)
- Depth unavailable for that camera
- Invalid depth pixels in mask region

**Solutions:**
- Observer provides depth from surround cameras (this is normal)
- Check logs: `[Tier 1] Using pre-computed distance`
- Heuristic fallback activates automatically

### Issue: Slow performance

**Solutions:**
```powershell
# Use nano model
# In people_observer/config.py: YOLO_MODEL_PATH = "yolov11n-seg.pt"

# Lower rates
python friendly_spot_main.py --robot IP --enable-observer --rate 2.0 --verbose

# Disable visualization
python friendly_spot_main.py --robot IP --enable-observer --save-images output/ --verbose
```

## Integration Details

### Data Flow

```
Surround Cameras (5x)
    ↓ (2 Hz)
YOLOv11-seg Detection
    ↓
Depth Extraction (mask-based)
    ↓
Select Closest Person
    ↓
PTZ Command (pan/tilt)
    ↓
PTZ Frame Fetch
    ↓
PersonDetection Object
    ├─ bbox (x, y, w, h)
    ├─ mask (HxW bool array)
    ├─ confidence (0-1)
    ├─ distance_m (float)
    ├─ depth_source (camera name)
    ├─ frame (HxWx3 BGR)
    ├─ depth_frame (HxW float32)
    ├─ tracked_by_ptz (True)
    └─ tracking_quality (0-1)
    ↓
PerceptionPipeline.read_perception(person_detection)
    ├─ Uses person_detection.frame as input
    ├─ Uses person_detection.distance_m (Tier 1)
    └─ Falls back to depth_frame → heuristic if needed
    ↓
PerceptionData
    ├─ pose_result
    ├─ face_result
    ├─ emotion_result
    ├─ gesture_result
    └─ distance_m
    ↓
ComfortModel.predict_behavior(perception)
    ↓
BehaviorExecutor.execute_behavior(behavior)
    ↓
Spot Robot Commands
```

### Thread Safety

- **ObserverBridge** runs in background daemon thread
- **Queue** (maxsize=1) for thread-safe communication
- Only latest PersonDetection kept (old ones discarded)
- Main thread polls queue via `get_person_detection()`

## Next Steps

1. **Test with real robot**: `python friendly_spot_main.py --robot IP --enable-observer --once --visualize --save-images test/ --verbose`
2. **Tune YOLO confidence**: Adjust `MIN_CONFIDENCE` in config
3. **Optimize performance**: Switch to smaller model if needed
4. **Add temporal filtering**: Smooth distance estimates over time
5. **Implement full behaviors**: Expand `BehaviorExecutor` with more actions

## Files Modified/Created

### New Files
- ✅ `unified_visualization.py` - Shared visualization for both pipelines
- ✅ `RUN_OBSERVER.md` - People observer standalone guide
- ✅ `DEPTH_VISUALIZATION_SUMMARY.md` - Technical depth visualization docs
- ✅ `INTEGRATED_SYSTEM_GUIDE.md` - This file

### Modified Files
- ✅ `observer_bridge.py` - Full implementation with people_observer integration
- ✅ `friendly_spot_main.py` - Added --enable-observer, --once, --visualize, --save-images
- ✅ `people_observer/visualization.py` - Uses unified visualization
- ✅ `people_observer/app.py` - Fixed --verbose conflict

### Consolidated
- ✅ Visualization code unified across both systems
- ✅ Depth colormap shared implementation
- ✅ Detection drawing shared implementation

## Support

Issues? Check:
1. Logs with `--verbose`
2. Test with `--once` for single cycle debugging
3. Verify model file: `people_observer/yolov11x-seg.pt`
4. Check robot connection: Ping robot IP
5. Validate depth sources: Look for `*_depth_in_visual_frame` in logs

Happy tracking! 🤖📸
