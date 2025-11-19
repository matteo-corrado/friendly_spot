# Friendly Spot YOLOv11 Segmentation & Depth Integration - Implementation Progress

## Overview
Upgrading the Friendly Spot pipeline to use YOLOv11 instance segmentation with depth-based distance estimation, integrating people_observer tracking with perception pipeline.

## Completed Changes

### 1. YOLOv11 Segmentation Support ✅
**File:** `people_observer/detection.py`

- **Updated `Detection` dataclass** to include optional segmentation mask:
  ```python
  mask: np.ndarray = None  # Optional segmentation mask (H, W) boolean array
  ```

- **Enhanced `YoloDetector`**:
  - Detects if model supports segmentation (checks for 'segment' in model.task)
  - Extracts segmentation masks from YOLO results (`r.masks.data`)
  - Resizes masks to match original image dimensions
  - Converts masks to boolean format (threshold at 0.5)
  - Falls back gracefully for detection-only models

**Usage:** Use YOLOv11-seg models (e.g., `yolov11n-seg.pt`, `yolov11s-seg.pt`) for instance segmentation.

### 2. Mask-Based Depth Extraction ✅
**File:** `people_observer/tracker.py`

- **Enhanced `estimate_detection_depth_m()`** function:
  - **Method 1 (Preferred):** Uses segmentation mask to sample depth only within person pixels
    - More accurate than bbox sampling (excludes background)
    - Returns median depth of masked region
  - **Method 2 (Fallback):** Original bbox-based sampling (inner 60% of bbox)
  - Parameter `use_mask=True` enables mask-based extraction

**Benefits:**
- Eliminates background depth contamination
- Works with partial occlusions
- More robust in crowded scenes

### 3. Depth Image Support in VideoSource ✅
**File:** `video_sources.py`

- **Updated VideoSource ABC**:
  - Changed `read()` signature to return 3-tuple: `(success, visual_frame, depth_frame)`
  - Standardized across all implementations

- **Enhanced `SpotPTZImageClient`**:
  - Added `include_depth=False` parameter
  - Automatically detects depth source availability (e.g., `ptz_depth_in_visual_frame`)
  - Implements `_decode_depth_image()` method:
    - Decodes uint16 depth to float32 meters using `depth_scale`
    - Masks invalid pixels (0 and 65535) as NaN
    - Returns aligned depth image matching visual frame dimensions

- **Updated `WebcamSource`**:
  - Returns `(ret, frame, None)` - webcam has no depth sensor

- **Updated `SpotPTZWebRTC`**:
  - Signature updated for consistency (implementation still TODO)

**Usage:**
```python
# Enable depth fetching
source = create_video_source('imageclient', robot=robot, 
                             source_name='ptz', include_depth=True)
ret, visual, depth = source.read()
```

## Remaining Implementation Tasks

### 4. Update run_pipeline.py with Depth-Based Distance 🔄
**File:** `run_pipeline.py`

**Changes Needed:**
1. Update `PerceptionPipeline.read_perception()` to handle 3-tuple from video source:
   ```python
   ret, frame, depth_frame = self.video_source.read()
   ```

2. Replace heuristic `estimate_distance_m()` with depth-based calculation:
   - Extract person bounding box from pose landmarks or face detection
   - Create Detection object with bbox
   - Call `estimate_detection_depth_m(depth_frame, det)` from tracker module
   - Fallback to heuristic if depth unavailable

3. Pass depth-based distance to `PerceptionInput`:
   ```python
   distance_m = estimate_detection_depth_m(depth_frame, det) if depth_frame is not None else estimate_distance_m(landmarks, frame_h)
   ```

### 5. Create Unified PersonDetection Class 🔄
**New File:** `detection_types.py` (centralized)

**Purpose:** Shared data structure between people_observer and friendly_spot

**Proposed Structure:**
```python
@dataclass
class PersonDetection:
    """Unified person detection with segmentation and tracking."""
    # Detection info
    bbox_xywh: Tuple[int, int, int, int]  # Bounding box
    mask: Optional[np.ndarray]            # Segmentation mask (H, W)
    confidence: float                      # Detection confidence
    
    # Depth/distance
    distance_m: Optional[float]           # Distance to person in meters
    depth_source: str                     # 'mask', 'bbox', or 'heuristic'
    
    # Tracking info
    source_camera: str                    # Which camera detected (e.g., 'frontleft_fisheye')
    tracked_by_ptz: bool                  # Is PTZ currently tracking this person
    tracking_quality: float               # PTZ tracking quality [0.0, 1.0]
    
    # Timestamps
    detection_time: float                 # When detected
    last_seen_time: float                 # Last update timestamp
```

### 6. Integrate people_observer into friendly_spot_main 🔄
**File:** `friendly_spot_main.py`

**Architecture:**
1. **Start people_observer tracking thread** (surround camera monitoring):
   ```python
   from people_observer.tracker import ObserverThread
   observer = ObserverThread(robot, image_client, ptz_client, cfg)
   observer.start()  # Runs in background
   ```

2. **Monitor for person detection events**:
   - Observer detects person in surround cameras
   - Commands PTZ to point at person
   - Signals when person is centered in PTZ view
   - Provides PersonDetection with bbox, mask, depth

3. **Trigger perception pipeline** when person in PTZ:
   ```python
   if observer.person_in_ptz_view():
       person_det = observer.get_current_detection()
       # Pass to perception pipeline
       perception = pipeline.read_perception(person_det)
   ```

4. **Pass PTZ frames to perception**:
   - Observer provides PTZ frames directly (avoid duplicate fetching)
   - Perception uses PTZ frames for pose/face/emotion/gesture analysis
   - Distance already calculated by observer using mask + depth

**Benefits:**
- Efficient: Only run expensive perception when person detected
- Accurate: PTZ provides high-resolution centered view of person
- Integrated: Depth and segmentation inform both tracking and perception

### 7. Simplify and Consolidate 🔄
**Goal:** Remove redundant code, streamline architecture

**Files to Review:**
- `Behavioral/human_image_extractor.py` - may be redundant with new pipeline
- `Behavioral/spot_yolo_person_to_ptz.py` - logic now in people_observer
- Duplicate YOLO detection code
- Unused imports and modules

**Consolidation:**
- Merge detection logic into `detection_types.py`
- Single source of truth for person detection data structures
- Remove or archive experimental scripts

## Model Requirements

### YOLOv11 Segmentation Models
Download from Ultralytics:
- **Nano:** `yolov11n-seg.pt` (~6.5 MB, fastest)
- **Small:** `yolov11s-seg.pt` (~24 MB, balanced)
- **Medium:** `yolov11m-seg.pt` (~52 MB, accurate)
- **Large:** `yolov11l-seg.pt` (~136 MB, very accurate)
- **XLarge:** `yolov11x-seg.pt` (~287 MB, most accurate)

**Recommended:** Start with `yolov11s-seg.pt` for balance of speed and accuracy.

Place model in: `people_observer/` directory or update `DEFAULT_YOLO_MODEL` in `config.py`.

## Testing Plan

### Phase 1: Component Testing
1. **Test YOLOv11-seg detection**:
   ```bash
   python -m people_observer.test_yolo_model yolov11s-seg.pt
   ```

2. **Test depth extraction with mask**:
   - Run people_observer with visualization
   - Verify mask-based depth values match visual expectations

3. **Test video source depth fetching**:
   ```python
   source = create_video_source('imageclient', robot=robot, 
                                source_name='frontleft_fisheye_image', 
                                include_depth=True)
   ret, visual, depth = source.read()
   assert depth is not None
   ```

### Phase 2: Integration Testing
1. **Test people_observer → friendly_spot handoff**:
   - Observer detects person
   - PTZ points to person
   - Perception pipeline receives person detection
   - Distance calculation uses depth + mask

2. **End-to-end pipeline**:
   ```bash
   python friendly_spot_main.py --robot ROBOT_IP --user USER --password PASS
   ```

### Phase 3: Performance Validation
- **Target:** 5 Hz full pipeline rate
- **Monitor:** CPU/GPU usage, frame latency, detection accuracy
- **Optimize:** Model size, image resolution, processing parallelization

## API Changes Summary

### Breaking Changes
1. **VideoSource.read()** now returns 3-tuple: `(success, visual, depth)`
   - **Migration:** Update all `ret, frame = source.read()` to `ret, frame, depth = source.read()`

2. **Detection** dataclass includes optional `mask` field
   - **Backward compatible:** Old code ignores mask, new code can use it

### New Parameters
- `SpotPTZImageClient(include_depth=False)` - enable depth fetching
- `estimate_detection_depth_m(depth_img, det, use_mask=True)` - use mask-based extraction

## Configuration Updates

### Update config.py
```python
# YOLOv11 segmentation model
DEFAULT_YOLO_MODEL = "yolov11s-seg.pt"  # Changed from yolov8n.pt

# Enable depth in video sources
DEFAULT_INCLUDE_DEPTH = True  # New setting
```

### Update friendly_spot_main.py arguments
```python
ap.add_argument("--enable-depth", action="store_true", default=True,
                help="Enable depth sensor for distance estimation")
ap.add_argument("--use-observer", action="store_true", default=True,
                help="Use people_observer for person detection and tracking")
```

## Next Steps

1. ✅ **Complete `run_pipeline.py` updates** - handle depth frames, use mask-based distance
2. ✅ **Create `detection_types.py`** - unified PersonDetection class
3. ✅ **Integrate observer into main pipeline** - background tracking thread
4. ✅ **Test with YOLOv11-seg model** - download and validate segmentation
5. ✅ **End-to-end testing** - full pipeline with robot
6. ✅ **Performance optimization** - profile and optimize bottlenecks
7. ✅ **Documentation** - update README with new architecture

## Notes

- **PTZ Depth Availability:** PTZ camera may not have depth sensor - code handles gracefully
- **Surround Camera Depth:** Fisheye cameras have depth in `*_depth_in_visual_frame` format
- **Mask Quality:** Segmentation mask quality depends on model size and scene complexity
- **Distance Accuracy:** Depth sensors have ~5% error at 1-3 meters, increases with distance

---
**Last Updated:** 2025-11-19
**Status:** Phase 1-3 Complete, Phases 4-7 In Progress
