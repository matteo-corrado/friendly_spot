# =============================================================================
# Platform-Specific Installation Guide
# =============================================================================

This document provides detailed installation instructions for different
platforms and environments.

## Table of Contents
1. [Windows 10/11](#windows-1011)
2. [macOS (Intel)](#macos-intel)
3. [macOS (Apple Silicon M1/M2/M3)](#macos-apple-silicon)
4. [Linux (Ubuntu/Debian)](#linux-ubuntudebian)
5. [Linux (ARM64 - Jetson/Raspberry Pi)](#linux-arm64)
6. [Docker/Headless Servers](#dockerheadless-servers)

---

## Windows 10/11

### Prerequisites
- **Python:** 3.9 or 3.10 ONLY from [python.org](https://www.python.org/downloads/)
  - **Recommended: Python 3.10** (best overall compatibility)
  - WARNING: Do NOT use Python 3.11+ (Spot SDK max is 3.10)
  - WARNING: Python 3.8 or below NOT recommended (missing dependency features)
  - Add Python to PATH during installation
- **Architecture:** x86_64 only (ARM64 not supported)
- **Visual C++ Redistributable:** Usually included, but if needed: [Download](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Optional: GPU Acceleration (NVIDIA)
```powershell
# 1. Install CUDA Toolkit 12.x
# Download: https://developer.nvidia.com/cuda-downloads

# 2. Install cuDNN 8.9
# Download: https://developer.nvidia.com/cudnn

# 3. Verify
nvcc --version
nvidia-smi
```

### Installation Steps
```powershell
# 1. Clone repository
git clone https://github.com/yourusername/friendly_spot.git
cd friendly_spot

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Pre-download models (optional for offline use)
python scripts/download_models.py

# 6. Test installation
python -c "import cv2, mediapipe, tensorflow, deepface; print('✓ All OK')"
```

---

## macOS (Intel)

### Prerequisites
```bash
# Install Homebrew (if not present)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.10 (REQUIRED for Spot SDK compatibility)
brew install python@3.10

# Verify version
python3.10 --version  # Should show Python 3.10.x
```

### Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/yourusername/friendly_spot.git
cd friendly_spot

# 2. Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Pre-download models
python scripts/download_models.py

# 6. Test installation
python -c "import cv2, mediapipe, tensorflow, deepface; print('✓ All OK')"
```

### Camera Permissions
Grant Terminal camera access:
1. Open **System Settings → Privacy & Security → Camera**
2. Enable **Terminal** (or your IDE)

---

## macOS (Apple Silicon)

### Prerequisites
```bash
# macOS 12+ includes Python 3.9, but 3.10 is better
# Install Python 3.10 for best compatibility

# Check system version
python3 --version

# If not 3.9 or 3.10, install via Homebrew:
brew install python@3.10
```

### Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/yourusername/friendly_spot.git
cd friendly_spot

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install Metal plugin for GPU acceleration
pip install tensorflow-metal

# 6. Pre-download models
python scripts/download_models.py

# 7. Test installation (should show Metal GPU)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

### Expected Output
```
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Troubleshooting
- **"No GPU found":** Ensure tensorflow-metal is installed
- **Slow performance:** Try Python 3.10 specifically (best M1/M2 support)
- **Webcam issues:** Check camera permissions in System Settings

---

## Linux (Ubuntu/Debian)

### Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10 (REQUIRED - 3.9 also works but 3.10 recommended)
sudo apt install python3.10 python3.10-venv python3.10-dev
sudo apt install build-essential libssl-dev libffi-dev

# Verify version
python3.10 --version  # Should show Python 3.10.x

# Install system dependencies for OpenCV
sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1
```

### Optional: GPU Acceleration (NVIDIA)
```bash
# 1. Install CUDA (if NVIDIA GPU present)
# Follow: https://developer.nvidia.com/cuda-downloads

# 2. Verify
nvcc --version
nvidia-smi

# 3. Install cuDNN
# Download .deb from: https://developer.nvidia.com/cudnn
sudo dpkg -i cudnn-local-repo-*.deb
sudo apt update
sudo apt install libcudnn8
```

### Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/yourusername/friendly_spot.git
cd friendly_spot

# 2. Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Pre-download models
python scripts/download_models.py

# 6. Test installation
python -c "import cv2, mediapipe, tensorflow, deepface; print('✓ All OK')"
```

### Webcam Permissions
```bash
# Check camera device
ls -l /dev/video*

# If permission denied, add user to video group
sudo usermod -a -G video $USER
# Log out and back in for changes to take effect
```

---

## Linux (ARM64)

### Jetson Nano/Xavier/Orin

#### Prerequisites
```bash
# Use JetPack SDK Python (pre-configured for TensorRT)
python3 --version  # Should be 3.8-3.11

# Install system dependencies
sudo apt update
sudo apt install libopencv-dev python3-opencv
```

#### Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/yourusername/friendly_spot.git
cd friendly_spot

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies (skip TensorFlow, use JetPack version)
pip install numpy==1.26.4
pip install opencv-contrib-python==4.10.0.84
pip install mediapipe==0.10.14
pip install deepface==0.0.92

# 4. Use system TensorFlow (optimized for Jetson)
# Don't install tensorflow via pip

# 5. Test
python -c "import cv2, mediapipe; print('OK')"
```

### Raspberry Pi (64-bit OS)

**Limited Support:** TensorFlow 2.15 not available for ARM64. Use TensorFlow Lite or custom build.

```bash
# Use TensorFlow Lite runtime instead
pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

---

## Docker/Headless Servers

### Dockerfile Example

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Use headless OpenCV (no GUI dependencies)
RUN pip uninstall -y opencv-contrib-python && \
    pip install opencv-contrib-python-headless==4.10.0.84

# Pre-download models at build time
COPY scripts/download_models.py scripts/
RUN python scripts/download_models.py

# Copy application code
COPY . .

CMD ["python", "friendly_spot_main.py", "--robot", "192.168.80.3"]
```

### Build and Run
```bash
# Build image
docker build -t friendly-spot .

# Run with environment variables
docker run --rm \
  -e BOSDYN_CLIENT_USERNAME=<your_username> \
  -e BOSDYN_CLIENT_PASSWORD=<your_password> \
  friendly-spot
```

### For Webcam in Docker
```bash
# Linux only: pass /dev/video0 device
docker run --rm \
  --device=/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  friendly-spot --webcam
```

---

## Verification Tests

### 1. Python Version
```bash
python --version
# Expected: Python 3.9.x, 3.10.x, or 3.11.x
```

### 2. Core Dependencies
```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import mediapipe; print('MediaPipe:', mediapipe.__version__)"
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import deepface; print('DeepFace:', deepface.__version__)"
```

### 3. GPU Detection (if applicable)
```bash
# CUDA (NVIDIA)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Metal (Apple Silicon)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

### 4. Model Downloads
```bash
python scripts/download_models.py
# Expected: All models downloaded and verified successfully!
```

---

## Getting Help

If you encounter issues:

1. **Check Python version:** Must be 3.9-3.11
2. **Check error message:** Common issues documented in PIPELINE_README.md
3. **Check platform compatibility:** See CROSS_PLATFORM_ANALYSIS.md
4. **Open an issue:** Include platform, Python version, and full error output
