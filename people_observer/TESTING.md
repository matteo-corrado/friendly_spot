# Testing Guide for people_observer YOLO Detection

## Overview
This guide walks through testing the YOLO person detection system, starting with local webcam testing, then progressing to Spot's surround cameras.

## Available YOLO Models

YOLOv8 models (Ultralytics) in order of speed vs accuracy:
- **yolov8n.pt** - Nano (fastest, current default)
- **yolov8s.pt** - Small
- **yolov8m.pt** - Medium
- **yolov8l.pt** - Large
- **yolov8x.pt** - Extra Large (slowest, most accurate)

First use of any model will auto-download from Ultralytics (~6-140 MB depending on model).

## Test Steps

### Step 0: Test All YOLO Models (Local Only)
```powershell
python -m friendly_spot.people_observer.test_yolo_model
```

**What it does:**
- Tests loading all 5 YOLOv8 models (n/s/m/l/x)
- Verifies each model loads correctly
- Checks class consistency across models
- Shows all 80 COCO classes
- Reports which models are ready to use

**Expected Output:**
- Ultralytics YOLO imported successfully
- Each model tested individually (downloads if needed)
- Model information (80 classes, person = class 0)
- Class consistency check across all models
- Final summary showing successful/failed models

**If this fails:**
- Run: `pip install ultralytics opencv-python numpy`
- Ensure you're in the correct venv

### Step 0b: Benchmark Models on Webcam (Local Only)
```powershell
python -m friendly_spot.people_observer.test_yolo_webcam
```

**What it does:**
- Tests each YOLO model on your laptop webcam
- Measures inference time and FPS for each model
- Shows live detections with bounding boxes
- Provides performance comparison and recommendations

**What to observe:**
- Live webcam feed with detection boxes
- Real-time inference time (ms) per frame
- FPS counter
- Stats overlay showing model name, performance metrics
- Press 'n' to skip to next model, 'q' to quit, 's' to save frame

**Expected Performance (CPU):**
- yolov8n: ~100-200ms per frame, 5-10 FPS
- yolov8s: ~200-400ms per frame, 2-5 FPS
- yolov8m: ~400-800ms per frame, 1-2 FPS
- yolov8l/x: >1000ms per frame, <1 FPS

**Use this to:**
- Determine which model is fast enough for your needs
- See actual detection quality before testing on Spot
- Verify YOLO detects people in various conditions

### Step 1: Test Detection with Dry-Run on Spot (No PTZ Commands)

**Single Cycle Test (Recommended First Test):**
```powershell
python -m friendly_spot.people_observer.app --hostname <ROBOT_IP> --once --dry-run --visualize
```

**What happens:**
1. Connects to Spot
2. Captures one frame from all 5 surround cameras
3. Runs YOLO detection on each frame
4. Shows OpenCV window with all cameras in grid layout
5. Draws bounding boxes around detected people
6. Logs what PTZ command would be sent (but doesn't send it)
7. Exits after one cycle

**Continuous Test:**
```powershell
python -m friendly_spot.people_observer.app --hostname <ROBOT_IP> --dry-run --visualize
```

**What to observe:**
- All 5 camera views displayed in 3x2 grid
- Green bounding boxes around detected people
- Confidence scores displayed on each detection
- Console logs showing:
  - Camera name with detection
  - Confidence level (0.00-1.00)
  - Bounding box coordinates
  - Computed pan/tilt angles (not executed)
- Press 'q' or ESC to quit

**Expected console output:**
```
INFO - DRY-RUN MODE: PTZ commands will be skipped
INFO - VISUALIZATION MODE: OpenCV windows will display detections
INFO - Detection: camera=frontleft_fisheye_image, confidence=0.87, bbox=(324,156,89,234)
INFO - [DRY-RUN] PTZ command: pan=35.42deg, tilt=-5.00deg, zoom=0.00
```

### Step 2: Verify Detection Across Multiple Cameras

Have someone walk around Spot while running the continuous test:
```powershell
python -m friendly_spot.people_observer.app --hostname <ROBOT_IP> --dry-run --visualize
```

**What to verify:**
1. **Front cameras** detect when person is in front
2. **Side cameras** detect when person walks to left/right
3. **Back camera** detects when person is behind
4. System selects the "best" detection (largest/closest)
5. Pan angle changes as person moves around robot

**Troubleshooting:**
- **No detections**: Check confidence threshold (default 0.30), try lowering with env var: `$env:PEOPLE_OBSERVER_CONFIDENCE="0.25"`
- **False positives**: Increase confidence threshold: `$env:PEOPLE_OBSERVER_CONFIDENCE="0.50"`
- **Slow performance**: Switch to GPU by editing `config.py` line 66: `YOLO_DEVICE = "cuda"` (requires CUDA)

### Step 3: Test PTZ Control (Live Operation)

**Only proceed if Steps 1 & 2 show good detections!**

```powershell
# Without visualization (production mode)
python -m friendly_spot.people_observer.app --hostname <ROBOT_IP>

# With visualization (recommended for first live test)
python -m friendly_spot.people_observer.app --hostname <ROBOT_IP> --visualize
```

**What happens:**
- PTZ camera will actively track detected people
- System aims at the largest/closest person across all cameras
- PTZ movements are smoothed (max 8deg/update)
- Press Ctrl+C to stop

**Safety notes:**
- Ensure robot has clear PTZ view
- PTZ will move automatically - warn others nearby
- Emergency stop will halt all operations

### Step 4: Test Different Modes

**Bearing mode (default - faster):**
```powershell
python -m friendly_spot.people_observer.app --hostname <ROBOT_IP> --mode bearing --dry-run --visualize
```

**Transform mode (more accurate, uses camera intrinsics):**
```powershell
python -m friendly_spot.people_observer.app --hostname <ROBOT_IP> --mode transform --dry-run --visualize
```

Compare pan/tilt angles in console output between modes.

## Configuration Tuning

### Change YOLO Model
Use environment variable or edit `config.py`:
```powershell
# Use smaller/faster model
$env:YOLO_MODEL = "yolov8n.pt"

# Use larger/more accurate model
$env:YOLO_MODEL = "yolov8s.pt"  # or m, l, x
```

### Adjust Detection Sensitivity
```powershell
# More sensitive (more detections, more false positives)
$env:PEOPLE_OBSERVER_CONFIDENCE = "0.25"

# Less sensitive (fewer detections, fewer false positives)
$env:PEOPLE_OBSERVER_CONFIDENCE = "0.50"
```

### Change Loop Speed
```powershell
# Slower updates (less CPU/GPU)
$env:LOOP_HZ = "5"

# Faster updates (more CPU/GPU)
$env:LOOP_HZ = "10"
```

## Expected Performance

**YOLOv8n on CPU (default):**
- ~100-200ms per 5-camera batch
- ~5-7 Hz loop rate achievable
- Suitable for tracking walking people

**YOLOv8n on GPU (CUDA):**
- ~20-50ms per 5-camera batch  
- ~10-15 Hz loop rate achievable
- Smoother tracking

## Visualization Controls

When `--visualize` is active:
- **'q' key**: Quit application
- **ESC key**: Quit application
- Window shows all 5 cameras in grid
- Green boxes = person detections
- Text shows confidence and detection count
- Bottom bar shows total detections across cameras

## Common Issues

**Issue:** "No person detected in any camera"
**Fix:** Check lighting, distance from robot, confidence threshold

**Issue:** OpenCV window freezes
**Fix:** Ensure window has focus, try clicking on it, or restart

**Issue:** "Model not found" error
**Fix:** Run test_yolo_model.py first to download model

**Issue:** Very slow inference
**Fix:** Use yolov8n (not m/l/x) or enable GPU with CUDA

**Issue:** Connection timeout
**Fix:** Verify robot IP, ensure on same network, check firewall

## Success Criteria

[PASS] Step 0: All YOLO models load without errors
[PASS] Step 0b: Webcam test shows detections with reasonable performance
[PASS] Step 1: Single-cycle test captures and displays 5 camera views with detections
[PASS] Step 2: Detects people in multiple camera views as they walk around robot
[PASS] Step 3: PTZ tracks person smoothly (live test)
[PASS] Step 4: Both bearing and transform modes compute reasonable pan/tilt angles

## Next Steps

After successful testing:
1. Choose optimal YOLO model based on webcam performance test
2. Tune confidence threshold for environment
3. Test in different lighting conditions
4. Test with multiple people
5. Consider GPU acceleration for better performance (if using larger models)
6. Implement multi-person tracking policy
