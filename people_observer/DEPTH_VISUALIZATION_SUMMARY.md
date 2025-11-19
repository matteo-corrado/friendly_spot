# People Observer Depth Visualization - Summary

## What Was Done

Enhanced `people_observer` to run standalone with **depth-masked visualization** showing segmentation masks color-coded by depth.

## Changes Made

### 1. Enhanced Visualization Module (`visualization.py`)

#### Added `create_depth_colormap()` function
- Converts depth image (meters) to color-coded heatmap
- **Blue** = close (0.5-2m), **Red** = far (3.5-5m)
- Uses OpenCV's COLORMAP_JET
- Handles NaN values (invalid depth pixels)

#### Updated `draw_detections()` function
- **New parameter**: `depth_img` (optional depth frame)
- Renders segmentation masks with depth overlay:
  - If depth available: Apply depth colormap to mask area
  - If no depth: Use solid cyan overlay
- Extracts and displays distance on label: `Person 0.95 | 2.34m`
- Mask overlay uses alpha blending (40% transparency)

#### Updated `show_detections_grid()` function
- **New parameter**: `depth_dict` (dict of camera_name → depth_image)
- Passes depth frames to `draw_detections()`
- Displays all 5 surround cameras with depth-colored masks

#### Updated `save_annotated_frames()` function
- **New parameter**: `depth_dict` (optional)
- Saves frames with depth-colored mask overlays to disk

### 2. Updated Tracker (`tracker.py`)

- Modified `run_loop()` to fetch depth frames: `include_depth=True`
- Pass `depth_frames` dict to `show_detections_grid()`
- Pass `depth_frames` dict to `save_annotated_frames()`

### 3. Configuration (`config.py`)

Already configured:
- `DEFAULT_INCLUDE_DEPTH = True` ✅

## How to Use

### 1. Quick Test (Visualization)

```powershell
# Activate environment
C:\Users\corra\Spot_SDK_Master\dartmouth_spot_capstone\.venv\Scripts\Activate.ps1

# Run with visualization
cd C:\Users\corra\Spot_SDK_Master\friendly_spot
python -m people_observer.app ROBOT_IP --visualize --verbose
```

**You'll see:**
- Grid of 5 surround camera views
- Green bounding boxes around people
- **Segmentation masks with depth-colored overlay**
  - Blue pixels = person is close
  - Red pixels = person is far
- Distance label on each detection
- Press `q` to quit

### 2. Save Annotated Frames

```powershell
python -m people_observer.app ROBOT_IP --save-images output_frames --verbose
```

Saves frames with depth masks to `output_frames/` directory.

### 3. Dry Run (No PTZ Movement)

```powershell
python -m people_observer.app ROBOT_IP --visualize --dry-run --verbose
```

## Technical Details

### Depth Extraction Flow

1. **Fetch frames**: YOLOv11-seg detects people → extracts masks
2. **Fetch depth**: Get `*_depth_in_visual_frame` for each camera
3. **Sample depth**: Extract depth values only within mask pixels
4. **Compute median**: Robust distance estimate (ignores outliers)
5. **Validate**: Compare against bbox-based heuristic (2.5x tolerance)
6. **Visualize**: Apply depth colormap to mask region, overlay on image

### Visualization Pipeline

```
Visual Frame (BGR) + Depth Frame (float32) + Detection (mask + bbox)
    ↓
create_depth_colormap(depth_img)  → colored depth image
    ↓
Apply mask to depth colormap  → masked depth region
    ↓
Alpha blend with visual frame  → semi-transparent overlay
    ↓
Draw bbox + distance label  → final annotated frame
```

### Color Scale

- **0.5m - 2.0m**: Blue → Cyan (close range)
- **2.0m - 3.5m**: Green → Yellow (medium range)
- **3.5m - 5.0m**: Orange → Red (far range)
- **Invalid/NaN**: Black (no depth data)

## Testing Checklist

- ✅ Imports work (visualization module loads)
- ✅ Depth colormap function added
- ✅ Mask overlay with depth support added
- ✅ Distance extraction and display
- ⏳ Test with real robot (verify depth overlay appears)
- ⏳ Test with saved frames (verify files written correctly)

## Expected Output

When running with `--visualize --verbose`:

```
INFO] Fetching camera intrinsics from robot...
INFO] frontleft_fisheye_image: kannala_brandt model (fx=286.5, fy=286.5)
...
DEBUG] YoloDetector loaded: YOLOv11x-seg (task: segment)
DEBUG] Segmentation supported: True
...
DEBUG] Mask-based depth extraction: 234 mask pixels (12.3% coverage)
DEBUG] Valid depths: 189/234 pixels, median=2.34m, min=2.12m, max=2.67m
...
INFO] CLOSEST PERSON: 2.34m away in frontleft_fisheye_image
```

In the OpenCV window:
- Green boxes around people
- **Blue/cyan/green/red masks** showing depth
- Label: `Person 0.93 | 2.34m`

## Files Modified

1. `people_observer/visualization.py` - Added depth colormap and mask overlay
2. `people_observer/tracker.py` - Pass depth frames to visualization
3. `people_observer/RUN_OBSERVER.md` - Complete usage guide (NEW)

## Next Steps

1. **Test with robot**: Run `python -m people_observer.app ROBOT_IP --visualize --verbose`
2. **Verify depth overlay**: Check that masks show blue/red gradient
3. **Tune color scale**: Adjust min_dist/max_dist in `create_depth_colormap()` if needed
4. **Profile performance**: Check if depth processing adds latency

## Integration with friendly_spot

The depth-masked visualizations are already integrated into the main pipeline:
- `PersonDetection` class includes mask + depth
- `run_pipeline.py` uses mask-based depth extraction
- `observer_bridge.py` can pass detections to perception pipeline

Just run `people_observer.app` standalone to **see the depth masks in action**! 🎨📏
