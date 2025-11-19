# Friendly Spot + People Observer - Quick Reference

## 🚀 Quick Commands

### Test Mode (One Cycle, Save Images)
```powershell
python friendly_spot_main.py --robot $env:ROBOT_IP --enable-observer --once --visualize --save-images test_output/ --verbose
```

### Full Demo (Live + Save)
```powershell
python friendly_spot_main.py --robot $env:ROBOT_IP --enable-observer --visualize --save-images demo_output/ --verbose
```

### Perception Only (No Observer)
```powershell
python friendly_spot_main.py --robot $env:ROBOT_IP --visualize --no-execute --verbose
```

### Webcam Development
```powershell
python friendly_spot_main.py --webcam --visualize --verbose
```

## ✨ What It Does

### With `--enable-observer`:
1. 🔍 Detects people in 5 surround cameras (YOLOv11-seg)
2. 📏 Measures distance with depth from segmentation masks
3. 🎥 Points PTZ camera at closest person automatically
4. 🧠 Runs perception on PTZ frame (pose/face/emotion/gesture)
5. 💚 Computes comfort and executes behavior
6. 🎨 Shows depth-colored masks + perception overlay

### Without `--enable-observer`:
1. 🎥 Uses PTZ camera stream directly (manual pointing)
2. 🧠 Runs perception pipeline
3. 💚 Computes comfort and executes behavior
4. 🎨 Shows perception overlay (no depth masks)

## 🎛️ Key Options

| Flag | Effect |
|------|--------|
| `--enable-observer` | ✅ Auto-detect people + PTZ control |
| `--once` | 🔄 Run one cycle and exit |
| `--visualize` | 👁️ Show live OpenCV window |
| `--save-images DIR` | 💾 Save annotated frames |
| `--no-execute` | ⛔ Perception only (no robot commands) |
| `--verbose` | 📝 Debug logging |
| `--rate HZ` | ⏱️ Loop frequency (default: 5 Hz) |

## 🎨 Visualization

### Live Window Shows:
- **Depth-colored masks**: Blue (close) → Green → Red (far)
- **Bounding boxes**: Green boxes around people
- **Distance labels**: "Person 0.95 | 2.34m"
- **Pose landmarks**: 33 keypoints
- **Info panel**: Pose, face, emotion, gesture labels

### Press `q` or `ESC` to quit

## 📁 Output Files

Format: `YYYYMMDD_HHMMSS_mmm_pipeline_iter####.jpg`

Each saved frame includes all visualizations.

## 🔧 Configuration

### Change YOLO Model
Edit `people_observer/config.py`:
```python
YOLO_MODEL_PATH = "yolov11n-seg.pt"  # n=fastest, x=most accurate
```

### Change Detection Threshold
```python
MIN_CONFIDENCE = 0.5  # 0.3 for more detections, 0.7 for fewer
```

### Change Loop Rates
```powershell
--rate 2.0  # Slower (2 Hz)
--rate 10.0 # Faster (10 Hz)
```

Observer rate is in `ObserverConfig` (2 Hz surround, 5 Hz PTZ).

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No person detected | Lower `MIN_CONFIDENCE` to 0.3 |
| PTZ doesn't move | Check `--dry-run` not set in observer |
| Slow performance | Use `yolov11n-seg`, lower `--rate` |
| No depth overlay | Normal for PTZ (uses surround depth) |
| Import errors | Activate venv first |

## 📊 Performance

- **YOLO detection**: 150ms (yolov11x) → 40ms (yolov11n)
- **Perception**: 200ms total @ 5 Hz
- **Observer loop**: 500ms @ 2 Hz (surround monitoring)

## ✅ System Status

- ✅ YOLOv11-seg with mask extraction
- ✅ Mask-based depth extraction
- ✅ PTZ auto-control from detections
- ✅ Full perception pipeline integration
- ✅ Unified visualization
- ✅ Save frames to disk
- ✅ Once mode for testing
- ✅ Observer bridge fully implemented

## 🎯 Recommended Workflow

1. **Test setup**: `--once --visualize --save-images test/`
2. **Check output**: Look at `test/` frames
3. **Full run**: Remove `--once`, let it loop
4. **Tune if needed**: Adjust confidence/model/rate
5. **Production**: Remove `--visualize` for performance

## 📚 Full Documentation

See `INTEGRATED_SYSTEM_GUIDE.md` for complete details.
