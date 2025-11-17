# Pre-Flight Checklist for Spot YOLO Testing

## Summary of Changes

### GPU Acceleration
- ✅ All YOLO inference now uses GPU by default with automatic CPU fallback
- ✅ Detection pipeline uses `device="cuda"` with FP16 precision when GPU available
- ✅ Device detection logs GPU name or warns about CPU fallback
- ✅ Test scripts updated to use GPU for webcam benchmarks

### Magic Numbers Eliminated
All hardcoded values replaced with named constants in `config.py`:

**Detection Constants:**
- `PERSON_CLASS_ID = 0`
- `MIN_CONFIDENCE = 0.30`
- `MIN_AREA_PX = 600`
- `YOLO_IOU_THRESHOLD = 0.5`
- `FALLBACK_HFOV_DEG = 133.0`

**Performance Constants:**
- `LOOP_HZ = 7`
- `DEFAULT_YOLO_MODEL = "yolov8x.pt"`
- `YOLO_IMG_SIZE = 640`
- `YOLO_DEVICE = "cuda"`

**PTZ Control Constants:**
- `PAN_DEADBAND_DEG = 1.0`
- `TILT_DEADBAND_DEG = 1.0`
- `MAX_DEG_PER_STEP = 8.0`
- `DEFAULT_TILT_DEG = -5.0`

**Stream Configuration:**
- `PTZ_NAME = "ptz"`
- `COMPOSITOR_SCREEN = "mech"`
- `TARGET_BITRATE = 2_000_000`

**Connection Constants:**
- `ROBOT_CONNECT_TIMEOUT_SEC = 10.0`
- `TIME_SYNC_TIMEOUT_SEC = 5.0`
- `MAX_RETRY_ATTEMPTS = 3`
- `RETRY_DELAY_SEC = 2.0`

**Visualization Constants (visualization.py):**
- Grid layout, font sizes, colors, padding values all named
- `GRID_COLS = 3`, `DEFAULT_TARGET_WIDTH = 1920`
- `GRID_ASPECT_RATIO = 3.0 / 4.0` for fisheye cameras

**Test Constants (test_yolo_webcam.py):**
- Webcam resolution, FPS calculation windows, overlay transparency
- Threshold recommendation parameters
- All documented at top of file

## Pre-Flight Checks

### 1. Environment Setup
```powershell
# Activate the virtual environment
cd C:\Users\corra\Spot_SDK_Master\dartmouth_spot_capstone
.\.venv\Scripts\Activate.ps1

# Verify GPU availability (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected Output (if GPU available):**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

### 2. Package Installation
Ensure all required packages are installed:
```powershell
pip install ultralytics opencv-python numpy torch
```

### 3. Robot Connection
- [ ] Spot is powered on and connected to network
- [ ] You have IP address: `____________`
- [ ] Credentials are set in Activate.ps1 (no CLI auth needed)
- [ ] Robot is NOT estopped
- [ ] Spot CAM PTZ is functional

### 4. Test Progression

#### Step 0: Verify YOLO Models (No Robot Required)
```powershell
cd C:\Users\corra\Spot_SDK_Master\friendly_spot\people_observer
python test_yolo_model.py
```
**Expected:** All 5 models load successfully, show 80 COCO classes, person = class 0

#### Step 0b: Webcam Benchmark (No Robot Required)
```powershell
python test_yolo_webcam.py
```
**Expected:** 
- Live webcam feed with detections
- Inference times and FPS displayed
- Confidence statistics and threshold recommendations
- GPU device logged at startup

**Success Criteria:**
- At least one model achieves < 143ms inference (7 Hz target)
- Average confidence > 0.70 for person detections
- No CUDA errors (or clean CPU fallback message)

#### Step 1: Dry-Run Mode (Spot Required)
```powershell
cd C:\Users\corra\Spot_SDK_Master\friendly_spot\people_observer
python -m people_observer.app --hostname <ROBOT_IP> --dry-run --once
```

**What This Tests:**
- Robot connection and authentication
- Image capture from 5 surround fisheye cameras
- YOLO detection on GPU
- Bearing calculation
- PTZ command computation (logged, not executed)

**Expected Output:**
```
INFO: YOLO using GPU: NVIDIA GeForce RTX 3060
INFO: DRY-RUN MODE: PTZ commands will be skipped
INFO: ONCE MODE: Will run single detection cycle and exit
INFO: Detection: camera=frontleft_fisheye_image, confidence=0.87, bbox=(320,240,150,400)
INFO: [DRY-RUN] PTZ command: pan=45.23deg, tilt=-5.00deg, zoom=0.00
INFO: ONCE MODE: Completed iteration 1, exiting
```

**Success Criteria:**
- [ ] No connection errors
- [ ] All 5 cameras provide images
- [ ] GPU detected and used
- [ ] Person detections found (if someone visible)
- [ ] Pan/tilt angles logged correctly

#### Step 2: Dry-Run with Visualization
```powershell
python -m people_observer.app --hostname <ROBOT_IP> --dry-run --visualize
```

**What This Tests:**
- All Step 1 functionality
- OpenCV live visualization of all 5 camera feeds
- Bounding box overlays on detections
- Grid layout rendering

**Expected:**
- 3x2 grid window showing all cameras
- Green bounding boxes around detected people
- Detection counts per camera
- Press 'q' to quit

**Success Criteria:**
- [ ] Visualization window appears
- [ ] All cameras visible in grid
- [ ] Detections rendered correctly
- [ ] Smooth frame updates (no lag)

#### Step 3: Live PTZ Test (Caution)
```powershell
python -m people_observer.app --hostname <ROBOT_IP> --once
```

**What This Tests:**
- FULL system including PTZ commands
- Actual camera movement

⚠️ **Safety:** Ensure clear space around Spot CAM. PTZ will move to track detected person.

**Expected:**
- PTZ camera aims at detected person
- Single iteration then exits

**Success Criteria:**
- [ ] PTZ moves smoothly to target
- [ ] Aiming accuracy within ~5 degrees
- [ ] No mechanical errors or collisions

#### Step 4: Continuous Tracking
```powershell
python -m people_observer.app --hostname <ROBOT_IP> --visualize
```

**What This Tests:**
- Continuous 7 Hz tracking loop
- Multi-person handling (largest/nearest selected)
- Smoothing and deadband behavior

**Expected:**
- PTZ continuously tracks person as they move
- 7 FPS loop (check logs for timing)
- Visualization shows live feed

**Success Criteria:**
- [ ] Smooth tracking without jitter
- [ ] Handles person entering/leaving frame
- [ ] No memory leaks over 1+ minute
- [ ] Clean exit with Ctrl+C

## Configuration Tuning

### If Detection Rate Too High (False Positives)
Edit `config.py`:
```python
MIN_CONFIDENCE = 0.40  # Increase from 0.30
MIN_AREA_PX = 800      # Increase from 600
```

### If Detection Rate Too Low (Missed People)
```python
MIN_CONFIDENCE = 0.20  # Decrease from 0.30
MIN_AREA_PX = 400      # Decrease from 600
```

### If PTZ Movement Too Jerky
```python
PAN_DEADBAND_DEG = 2.0   # Increase from 1.0
MAX_DEG_PER_STEP = 5.0   # Decrease from 8.0
```

### If Performance Issues
```python
DEFAULT_YOLO_MODEL = "yolov8n.pt"  # Use nano model
LOOP_HZ = 5                        # Reduce from 7
```

Or if GPU memory issues:
```python
YOLO_DEVICE = "cpu"  # Force CPU mode
```

## Troubleshooting

### "CUDA requested but not available"
**Cause:** PyTorch not installed with CUDA support
**Fix:** 
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "No desired surround cameras found"
**Cause:** Camera names don't match or cameras offline
**Fix:** Check available sources:
```python
from bosdyn.client.image import ImageClient
sources = image_client.list_image_sources()
for s in sources:
    print(s.name)
```

### "Frame transforms unavailable"
**Cause:** Transform snapshot missing from ImageResponse
**Fix:** System automatically falls back to bearing-only mode using HFOV

### PTZ Not Moving
**Check:**
- [ ] `--dry-run` flag NOT used
- [ ] PTZ service enabled: `robot.ensure_client(PtzClient)`
- [ ] Spot CAM powered on
- [ ] No E-stop active

### Slow Inference
**Check:**
- [ ] GPU actually being used (check logs for "YOLO using GPU")
- [ ] Model size (try yolov8n.pt instead of yolov8x.pt)
- [ ] Image resolution (640x640 is optimal)

## File Manifest

All changes made to:
- ✅ `config.py` - Added all constants, FALLBACK_HFOV_DEG, YOLO_IOU_THRESHOLD
- ✅ `detection.py` - GPU device selection with fallback, logging
- ✅ `tracker.py` - Pass device to YoloDetector, use FALLBACK_HFOV_DEG
- ✅ `io_robot.py` - Use cfg.ptz.compositor_screen and cfg.ptz.target_bitrate
- ✅ `visualization.py` - All magic numbers replaced with constants
- ✅ `test_yolo_webcam.py` - GPU device, all constants defined at top
- ✅ `test_yolo_model.py` - No changes needed (already clean)
- ✅ `cameras.py` - No changes needed (no magic numbers)
- ✅ `geometry.py` - No changes needed (mathematical constants)
- ✅ `ptz_control.py` - No changes needed (uses config values)
- ✅ `app.py` - No changes needed (uses config)

## Post-Test Review

After successful Spot testing, document:
1. Actual inference times per model on your GPU
2. Optimal confidence threshold for your environment
3. PTZ tracking quality (jitter, accuracy, speed)
4. Any configuration changes needed for your setup

## Next Steps

If all tests pass:
- Consider adding multi-person handling (group tracking)
- Integrate with voice control (dartmouth_spot_capstone)
- Add depth ranking when depth cameras available
- Implement PTZ position history for smoothing
