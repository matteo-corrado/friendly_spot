# =============================================================================
# Dependency Compatibility Matrix
# =============================================================================
# This document tracks version constraints and compatibility issues across
# all packages in the friendly_spot pipeline.
#
# Last updated: November 18, 2025
# =============================================================================

## Critical Python Version Constraint

**REQUIRED: Python 3.9 or 3.10 ONLY**

| Package                | Min Version | Max Version | Notes                                    |
|------------------------|-------------|-------------|------------------------------------------|
| **bosdyn-client**      | 3.7         | 3.10        | Spot SDK hard limit                      |
| **bosdyn-api**         | 3.6         | 3.10        | Protobuf generation compatibility        |
| **mediapipe**          | 3.8         | 3.11        | No Python 3.12+ support yet              |
| **tensorflow**         | 3.8         | 3.11        | TF 2.15 tested on 3.8-3.11               |
| **deepface**           | 3.7         | 3.11        | Follows TensorFlow                       |
| **ultralytics**        | 3.8         | 3.12        | YOLOv8, wide compatibility               |
| **opencv-python**      | 3.7         | 3.12        | Very wide support                        |

**Intersection: Python 3.9-3.10 is the ONLY compatible range for all packages.**

**Recommendation: Use Python 3.10** for best stability and performance.

---

## Protobuf Version Compatibility

**Critical Issue:** Different packages require different protobuf versions.

### Package Requirements

| Package                | Protobuf Requirement                     | Notes                                 |
|------------------------|------------------------------------------|---------------------------------------|
| **bosdyn-api**         | `>=3.19.4,!=4.24.0,!=5.28.0`             | Spot SDK official constraint          |
| **tensorflow 2.15**    | `>=3.19.0,<4.24`                         | Works with 3.x and early 4.x          |
| **deepface 0.0.92**    | `<4.0` or `>=4.0` depending on backend   | Backend-dependent                     |
| **grpcio**             | `>=3.19.0`                               | Works with most versions              |
| **ultralytics**        | No strict requirement                    | Flexible                              |

### Known Broken Versions

- **protobuf 4.24.0**: Breaks Spot SDK (bosdyn-api explicitly excludes it)
- **protobuf 5.28.0**: Crashes on Windows ([protobuf #18045](https://github.com/protocolbuffers/protobuf/issues/18045))
- **protobuf 6.x**: Not yet widely supported, avoid

### Solution

**Use flexible range that satisfies all:**
```
protobuf>=3.19.4,<5.0,!=4.24.0,!=5.28.0
```

This works with:
- Spot SDK (requires >=3.19.4)
- TensorFlow 2.15 (works up to 4.23)
- DeepFace (compatible with 3.x and 4.x)
- Avoids known broken versions

---

## NumPy Version Compatibility

**Critical Issue:** NumPy 2.0+ breaks OpenCV 4.12+

### Package Requirements

| Package                | NumPy Requirement     | Notes                                    |
|------------------------|-----------------------|------------------------------------------|
| **opencv-python 4.10** | `<2.0`                | OpenCV 4.10 works with NumPy 1.x         |
| **opencv-python 4.12** | `<2.3`                | NumPy 2.x support added but buggy        |
| **tensorflow 2.15**    | `<2.0`                | TF 2.15 compiled against NumPy 1.x       |
| **mediapipe 0.10.14**  | `>=1.21.0`            | Flexible                                 |
| **deepface 0.0.92**    | `>=1.14.0`            | Flexible                                 |
| **scipy 1.11.4**       | `<2.0`                | Compiled against NumPy 1.x               |

### Solution

**Pin to NumPy 1.26.4:**
```
numpy==1.26.4
```

This is:
- Latest stable NumPy 1.x release
- Compatible with all packages
- Avoids NumPy 2.x breaking changes

---

## TensorFlow / Keras Compatibility

**Critical Issue:** Keras version must match TensorFlow version.

### Package Relationships

| TensorFlow | Keras     | tf-keras  | Notes                                  |
|------------|-----------|-----------|----------------------------------------|
| 2.15.0     | 2.15.0    | 2.15.0    | Must match exactly                     |
| 2.16.0+    | 3.0.0+    | N/A       | Keras 3.0 is separate package          |

### DeepFace Dependencies

DeepFace compatibility varies by version:
- DeepFace 0.0.92: Works with TensorFlow 2.15.0, Keras 2.15.0, tf-keras 2.15.0
- DeepFace 0.0.95: Claims TensorFlow 2.19.1, Keras 3.12.0 support (UNTESTED)

### Solution Options

**Option 1: Newer versions (NEEDS TESTING):**
```
tensorflow==2.19.1
keras==3.12.0
deepface==0.0.95
# No tf-keras needed
```

**Option 2: Proven stable (FALLBACK):**
```
tensorflow==2.15.0
keras==2.15.0
tf-keras==2.15.0
deepface==0.0.92
```

---

## Platform-Specific Considerations

### Windows

**Issues:**
- MediaPipe: Python 3.9-3.10 only (no 3.12+ support)
- TensorFlow: Requires CUDA 12.x + cuDNN 8.9 for GPU
- OpenCV: Uses DirectShow backend (handles automatically)
- Protobuf 5.28.0: Crashes on Windows (avoid)

**Recommendations:**
- Use Python 3.10 from python.org
- Install Visual C++ Redistributable
- For GPU: Install CUDA Toolkit 12.x

### macOS Intel

**Issues:**
- Standard TensorFlow 2.15 is CPU-only
- Homebrew Python sometimes conflicts with system Python

**Recommendations:**
- Use Python 3.10 from Homebrew
- TensorFlow works but is CPU-only (acceptable performance)

### macOS Apple Silicon (M1/M2/M3)

**Issues:**
- TensorFlow requires `tensorflow-metal` plugin for GPU
- Some NumPy operations slower on ARM64
- MediaPipe has occasional stability issues

**Recommendations:**
- Use Python 3.10 (best M1/M2 support)
- Install `tensorflow-metal` for GPU acceleration
- Test thoroughly, fallback to CPU if issues

### Linux x86_64

**Best support across the board.**

**Recommendations:**
- Use Python 3.10 from system repos
- For GPU: Install CUDA 12.x + cuDNN 8.9
- Use opencv-python-headless for Docker/servers

### Linux ARM64 (Jetson, Raspberry Pi)

**Issues:**
- TensorFlow 2.15 not available via pip
- Limited PyTorch support

**Recommendations:**
- Use JetPack SDK TensorFlow (Jetson)
- Consider TensorFlow Lite runtime (Raspberry Pi)
- May need custom builds

---

## Testing Commands

### Verify Python Version
```bash
python --version
# Expected: Python 3.9.x or 3.10.x
```

### Verify Package Versions
```bash
pip list | grep -E "bosdyn|protobuf|numpy|tensorflow|mediapipe|opencv"
```

### Expected Output
```
bosdyn-api                5.0.1.2
bosdyn-client             5.0.1.2
bosdyn-core               5.0.1.2
keras                     2.15.0
mediapipe                 0.10.14
numpy                     1.26.4
opencv-contrib-python     4.10.0.84
protobuf                  3.20.3  (or 4.x within allowed range)
tensorflow                2.15.0
tf-keras                  2.15.0
```

### Test Imports
```bash
python -c "import bosdyn.client; print('Spot SDK OK')"
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
python -c "import mediapipe as mp; print('MediaPipe', mp.__version__)"
python -c "import cv2; print('OpenCV', cv2.__version__)"
python -c "import deepface; print('DeepFace OK')"
```

---

## Dependency Resolution Strategy

When `pip install -r requirements.txt` fails with conflicts:

### Step 1: Identify Conflict
```bash
pip install -r requirements.txt 2>&1 | grep -i conflict
```

### Step 2: Check Installed Versions
```bash
pip list | grep <conflicting-package>
```

### Step 3: Force Reinstall with Constraints
```bash
pip install --force-reinstall --no-deps <package>==<version>
```

### Step 4: Verify Resolution
```bash
pip check
```

---

## Future Compatibility Concerns

### When Spot SDK Adds Python 3.11+ Support

- MediaPipe may still be blocking factor
- Test TensorFlow 2.16+ compatibility (uses Keras 3.0)
- Re-evaluate protobuf version constraints

### When MediaPipe Adds Python 3.12+ Support

- Re-test all packages with Python 3.12
- Update documentation and requirements
- Test on all platforms (Windows/macOS/Linux)

### When TensorFlow 2.16+ Becomes Required

- **Breaking change:** Keras 3.0 is separate package
- DeepFace may need updates
- Re-test all perception pipelines

---

## Summary

**Current Setup (November 2025):**

```bash
Python: 3.10.8
bosdyn-*: 5.0.1.2
numpy: 1.26.4
opencv-contrib-python: 4.11.0.86
mediapipe: 0.10.21
tensorflow: 2.19.1
keras: 3.12.0 (Note: Keras 3.x is separate package with TF 2.19+)
deepface: 0.0.95
protobuf: 4.25.8 (range: >=4.23.0,<5.0, avoid 4.24.0 and 5.28.0)
torch: 2.9.1
ultralytics: 8.3.228
```

**Key Changes from Earlier Versions:**
- TensorFlow upgraded to 2.19.1 (from 2.15.0)
- Keras now uses 3.x architecture (breaking change from 2.x)
- tf-keras package no longer needed (integrated into TensorFlow 2.19+)
- protobuf 4.25.8 works with both Spot SDK and TensorFlow 2.19
- MediaPipe updated to 0.10.21 for better stability

**Testing Status:**
- TensorFlow 2.19.1 with Keras 3.12.0: NEEDS TESTING with DeepFace pipeline
- Previous tested config: TensorFlow 2.15.0 with Keras 2.15.0
- Fallback option available if compatibility issues arise
