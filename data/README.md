# Data Directory

Storage for models, outputs, and datasets used by Friendly Spot pipeline.

## Overview

This directory organizes data files separate from code, including YOLO models, output logs, captured images, and test datasets.

## Structure

```
data/
├── models/               # YOLO and ML model weights
│   ├── yolov8n.pt       # Detection model
│   ├── yolov8n-seg.pt   # Segmentation model
│   └── README.md        # Model download instructions
│
├── outputs/             # Generated outputs
│   ├── logs/           # Application logs
│   ├── videos/         # Recorded sessions
│   ├── images/         # Captured frames
│   └── detections/     # Detection results (JSON/CSV)
│
└── datasets/            # Training/evaluation data
    ├── test_images/     # Test set
    └── calibration/     # Camera calibration data
```

## Models

### YOLO Weights

Download pre-trained models:

```powershell
# YOLOv8n (detection only, 11 MB)
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o data/models/yolov8n.pt

# YOLOv8n-seg (with segmentation, 12 MB)
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt -o data/models/yolov8n-seg.pt

# YOLOv8s (larger, more accurate, 22 MB)
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -o data/models/yolov8s.pt
```

### Model Selection

| Model | Size | Speed (CPU) | Speed (GPU) | mAP | Use Case |
|-------|------|-------------|-------------|-----|----------|
| yolov8n | 11 MB | ~100ms | ~5ms | 37.3 | Real-time on CPU |
| yolov8s | 22 MB | ~200ms | ~7ms | 44.9 | Balanced |
| yolov8m | 52 MB | ~500ms | ~12ms | 50.2 | High accuracy |
| yolov8n-seg | 12 MB | ~120ms | ~7ms | 36.7 | Instance segmentation |

Configure in `src/perception/config.py`:

```python
DEFAULT_YOLO_MODEL = "data/models/yolov8n-seg.pt"
```

## Outputs

### Logs

Application logs stored in `outputs/logs/`:

```
outputs/logs/
├── friendly_spot_2024-01-15.log      # Daily log files
├── friendly_spot_2024-01-16.log
└── error.log                          # Error-only log
```

Configure logging in main script:

```python
import logging

logging.basicConfig(
    filename='data/outputs/logs/friendly_spot.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Videos

Recorded sessions in `outputs/videos/`:

```
outputs/videos/
├── session_2024-01-15_14-30-00.mp4
└── debug_run.avi
```

Record with OpenCV:

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/outputs/videos/session.mp4', fourcc, 30.0, (1920, 1080))

while True:
    frame = camera.get_frame()
    out.write(frame)

out.release()
```

### Images

Captured frames in `outputs/images/`:

```
outputs/images/
├── detection_001.jpg
├── detection_002.jpg
└── ...
```

Save frames:

```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"data/outputs/images/frame_{timestamp}.jpg"
cv2.imwrite(filename, frame)
```

### Detection Results

Structured detection logs in `outputs/detections/`:

```json
// detections_2024-01-15.json
{
  "timestamp": "2024-01-15T14:30:00Z",
  "frame_id": 123,
  "detections": [
    {
      "class": "person",
      "confidence": 0.87,
      "bbox": [100, 200, 50, 150],
      "depth_m": 2.3
    }
  ]
}
```

Save detections:

```python
import json

detection_data = {
    "timestamp": datetime.now().isoformat(),
    "frame_id": frame_id,
    "detections": [
        {
            "class": "person",
            "confidence": det.conf,
            "bbox": det.bbox_xywh,
            "depth_m": depth
        }
        for det in detections
    ]
}

with open('data/outputs/detections/detections.json', 'a') as f:
    f.write(json.dumps(detection_data) + '\n')
```

## Datasets

### Test Images

Curated test images in `datasets/test_images/`:

```
datasets/test_images/
├── single_person/
│   ├── close_001.jpg      # Person <2m
│   ├── medium_002.jpg     # Person 2-5m
│   └── far_003.jpg        # Person >5m
│
├── multi_person/
│   ├── group_001.jpg
│   └── crowd_002.jpg
│
└── edge_cases/
    ├── occlusion.jpg      # Partially hidden
    ├── dark.jpg           # Low light
    └── motion_blur.jpg    # Fast movement
```

Use in tests:

```python
test_image = cv2.imread("data/datasets/test_images/single_person/close_001.jpg")
detections = detector.predict_batch([test_image])[0]
assert len(detections) == 1  # Expect one person
```

### Calibration Data

Camera calibration files in `datasets/calibration/`:

```
datasets/calibration/
├── frontleft_fisheye_intrinsics.yaml
├── hand_camera_intrinsics.yaml
└── ptz_calibration.json
```

Load calibration:

```python
import yaml

with open('data/datasets/calibration/frontleft_fisheye_intrinsics.yaml') as f:
    intrinsics = yaml.safe_load(f)
    
fx = intrinsics['fx']
fy = intrinsics['fy']
cx = intrinsics['cx']
cy = intrinsics['cy']
```

## Storage Management

### Cleanup Old Files

Automatic cleanup script:

```python
# cleanup_outputs.py
import os
import time
from pathlib import Path

MAX_AGE_DAYS = 7  # Delete files older than 7 days

def cleanup_old_files(directory, max_age_days):
    now = time.time()
    cutoff = now - (max_age_days * 86400)
    
    for file_path in Path(directory).rglob('*'):
        if file_path.is_file() and file_path.stat().st_mtime < cutoff:
            file_path.unlink()
            print(f"Deleted: {file_path}")

cleanup_old_files('data/outputs/logs', MAX_AGE_DAYS)
cleanup_old_files('data/outputs/images', MAX_AGE_DAYS)
```

### Disk Usage

Check data directory size:

```powershell
# PowerShell
Get-ChildItem -Recurse data | Measure-Object -Property Length -Sum | Select-Object @{Name="Size (MB)"; Expression={[math]::Round($_.Sum / 1MB, 2)}}
```

### Compression

Compress old logs/videos:

```powershell
# Compress logs older than 30 days
$cutoff = (Get-Date).AddDays(-30)
Get-ChildItem data/outputs/logs -File | Where-Object {$_.LastWriteTime -lt $cutoff} | Compress-Archive -DestinationPath data/outputs/logs/archive.zip
```

## .gitignore

Exclude large files from version control:

```gitignore
# data/.gitignore

# Models (download separately)
models/*.pt
models/*.onnx

# Outputs (generated at runtime)
outputs/logs/*.log
outputs/videos/*.mp4
outputs/videos/*.avi
outputs/images/*.jpg
outputs/images/*.png
outputs/detections/*.json

# Keep directory structure
!models/README.md
!outputs/.gitkeep
!datasets/.gitkeep
```

## Dependencies

None (pure data storage).

## References

- [YOLO Models](https://github.com/ultralytics/assets/releases)
- [Data Management Best Practices](https://docs.python.org/3/library/pathlib.html)
