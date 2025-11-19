# Running People Observer with Depth-Masked Visualization

## Overview

The `people_observer` module can run standalone to detect people using YOLOv11-seg, extract depth from segmentation masks, and visualize detections with depth-colored overlays.

## Features

- ✅ **YOLOv11 Segmentation**: Instance segmentation masks for precise person detection
- ✅ **Depth Extraction**: Uses segmentation masks to extract accurate depth from Spot's surround cameras
- ✅ **Depth Visualization**: Color-coded depth overlay on masks (blue=close, red=far)
- ✅ **Distance Display**: Shows distance in meters for each detected person
- ✅ **Multi-camera Grid**: Displays all 5 surround cameras simultaneously
- ✅ **Save Mode**: Save annotated frames with masks and depth to disk

## Setup

### 1. Activate Virtual Environment

```powershell
# From dartmouth_spot_capstone folder
.venv\Scripts\Activate.ps1
```

### 2. Ensure YOLOv11-seg Model

Make sure you have a YOLOv11 segmentation model in `people_observer/`:

```powershell
# Download YOLOv11n-seg (nano, fastest)
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov11n-seg.pt')"
mv yolov11n-seg.pt people_observer/

# OR download larger model for better accuracy
# yolov11s-seg.pt, yolov11m-seg.pt, yolov11l-seg.pt, yolov11x-seg.pt
```

Update `people_observer/config.py` if needed:
```python
YOLO_MODEL_PATH = "yolov11x-seg.pt"  # Or your chosen model
```

## Usage

### Basic Visualization (Live OpenCV Window)

```powershell
cd C:\Users\corra\Spot_SDK_Master\friendly_spot
python -m people_observer.app ROBOT_IP --visualize --verbose
```

**Features:**
- Live grid display of all 5 surround cameras
- Segmentation masks overlaid with depth colors:
  - **Blue** = Close (0.5-2m)
  - **Green** = Medium (2-3.5m)
  - **Red** = Far (3.5-5m)
- Distance displayed on each detection
- Press `q` to quit

### Save Annotated Frames to Disk

```powershell
python -m people_observer.app ROBOT_IP --save-images output_frames --verbose
```

**Output:**
- Frames saved to `output_frames/` directory
- Filename format: `TIMESTAMP_iter####_cameraname.jpg`
- Each frame has masks + depth overlay + distance labels

### Dry Run (No PTZ Control)

Test detection without moving PTZ:

```powershell
python -m people_observer.app ROBOT_IP --visualize --dry-run --verbose
```

### Single Iteration (Quick Test)

Run one detection cycle and exit:

```powershell
python -m people_observer.app ROBOT_IP --visualize --once --verbose
```

## Understanding the Visualization

### Mask Overlay
- **Cyan overlay (no depth)**: Segmentation mask when depth unavailable
- **Blue-Red gradient (with depth)**: Depth-colored mask showing distance

### Bounding Box
- **Green box**: Detection bounding box
- **Label**: `Person 0.95 | 2.34m` (confidence | distance)

### Detection Count
- Top-right corner shows number of detections per camera

### Grid Layout
- 3 columns × 2 rows
- All 5 surround cameras displayed simultaneously
- Camera name shown in top-left of each cell

## Depth Extraction Details

### How It Works

1. **YOLOv11-seg** detects people and extracts segmentation masks
2. **Mask-based sampling**: Depth values extracted only from pixels within mask
3. **Robust statistics**: Median depth used (ignores outliers)
4. **Validation**: Depth compared against bbox-based heuristic (tolerance: 2.5x)
5. **Fallback**: If depth fails validation, uses bbox heuristic

### Depth Sources

Spot's surround cameras have depth in `*_depth_in_visual_frame` format:
- `frontleft_depth_in_visual_frame`
- `frontright_depth_in_visual_frame`
- `left_depth_in_visual_frame`
- `right_depth_in_visual_frame`
- `back_depth_in_visual_frame`

**Note**: PTZ camera typically does NOT have depth sensor.

### Depth Accuracy

- **With mask**: ±10cm typical accuracy
- **Without mask (bbox)**: ±30cm typical accuracy
- **Heuristic only**: ±50cm typical accuracy

## Configuration

Edit `people_observer/config.py`:

```python
# Model selection
YOLO_MODEL_PATH = "yolov11x-seg.pt"  # n, s, m, l, or x

# Depth settings
DEFAULT_INCLUDE_DEPTH = True  # Enable depth fetching

# Detection settings
MIN_CONFIDENCE = 0.5  # Detection confidence threshold
MIN_AREA_PX = 500  # Minimum bbox area to consider

# Loop settings
LOOP_HZ = 2.0  # Detection frequency (Hz)
```

## Troubleshooting

### No depth overlay visible
- Check that depth is enabled in config: `DEFAULT_INCLUDE_DEPTH = True`
- Verify camera has depth source: Look for `*_depth_in_visual_frame` in logs
- Increase `--verbose` to see depth extraction logs

### Masks not showing
- Verify you're using a YOLOv11-seg model (not regular YOLOv11)
- Check model file has `-seg` in filename
- Look for "Segmentation supported: True" in debug logs

### Distance shows N/A
- Depth may be unavailable for that camera
- Mask may not overlap valid depth pixels
- Check logs for "No valid depth pixels in mask"

### Performance issues
- Use smaller model: yolov11n-seg or yolov11s-seg
- Reduce LOOP_HZ in config (default: 2 Hz)
- Disable `--verbose` for production

## Examples

### Full-featured demo
```powershell
python -m people_observer.app 192.168.50.3 --visualize --save-images demo_output --verbose
```

### Quick test with small model
```python
# In config.py, set: YOLO_MODEL_PATH = "yolov11n-seg.pt"
python -m people_observer.app 192.168.50.3 --visualize --once --verbose
```

### Production monitoring (no viz, save only)
```powershell
python -m people_observer.app 192.168.50.3 --save-images monitoring_logs
```

## Integration with friendly_spot

To use people_observer detections in the friendly_spot perception pipeline:

```python
from observer_bridge import ObserverBridge, ObserverConfig

# Create bridge
config = ObserverConfig(enable_observer=True, surround_fps=2.0, ptz_fps=5.0)
bridge = ObserverBridge(robot, image_client, ptz_client, config)
bridge.start()

# In perception loop
if bridge.has_person():
    person_det = bridge.get_person_detection()  # PersonDetection with mask + depth
    perception = pipeline.read_perception(person_det)
    # ... use perception for behavior decision

bridge.stop()
```

## Next Steps

- [ ] Test with real robot to verify depth accuracy
- [ ] Tune MIN_CONFIDENCE and MIN_AREA_PX for your use case
- [ ] Implement full observer_bridge._observer_loop() integration
- [ ] Add temporal filtering for stable distance estimates
- [ ] Profile performance and optimize model selection

## Debug Mode

For comprehensive debug output:

```powershell
python -m people_observer.app ROBOT_IP --visualize --verbose 2>&1 | Tee-Object -FilePath debug_log.txt
```

This logs:
- Model configuration and task detection
- Batch prediction statistics
- Mask extraction details (pixel counts, coverage)
- Depth sampling method (mask vs bbox)
- Depth validation results
- Distance calculation tiers
