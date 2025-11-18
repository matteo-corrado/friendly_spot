# PTZ WebRTC Streaming Module

Real-time H.264 video streaming from Spot's PTZ camera via WebRTC protocol.

## Overview

This module provides a **separate, independent streaming pipeline** from the person detection system. It can be enabled/disabled on-demand and is designed for integration with downstream facial recognition and emotion detection models.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Main Detection Pipeline (tracker.py)                        │
│ ┌─────────────┐   ┌──────────────┐   ┌─────────────┐      │
│ │ Fisheye     │──▶│ YOLO Person  │──▶│ PTZ Control │      │
│ │ Cameras     │   │ Detection    │   │             │      │
│ └─────────────┘   └──────────────┘   └─────────────┘      │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │ TODO: Integration hook
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ PTZ WebRTC Streaming (ptz_stream.py)                        │
│ ┌─────────────┐   ┌──────────────┐   ┌─────────────┐      │
│ │ WebRTC      │──▶│ Frame Queue  │──▶│ Facial/     │      │
│ │ Connection  │   │ (async)      │   │ Emotion     │      │
│ │             │   │              │   │ Models      │      │
│ └─────────────┘   └──────────────┘   └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Core Modules

- **`ptz_stream.py`**: Main streaming manager
  - `PtzStream` class: High-level stream control
  - `PtzStreamConfig`: Configuration options
  - Thread-safe frame queue
  - Statistics tracking

- **`ptz_webrtc_client.py`**: WebRTC protocol implementation
  - `SpotPtzWebRTCClient`: Handles SDP negotiation
  - `SpotPtzVideoTrack`: Frame queuing
  - ICE connection management

- **`test_ptz_stream.py`**: Standalone test utility
  - Command-line interface for testing
  - Video recording capability
  - Frame display and statistics

## Installation

### 1. Install WebRTC Dependencies

```bash
cd people_observer
pip install -r requirements_webrtc.txt
```

**Required packages:**
- `aiortc>=1.3.0` - WebRTC implementation
- `av>=10.0.0` - Video codec support (H.264 decoding)
- `opencv-python>=4.8.0` - Already installed for detection
- `numpy>=1.24.0` - Already installed for detection

### 2. Verify Installation

```python
python -c "import aiortc; print(f'aiortc {aiortc.__version__} installed')"
```

## Usage

### Standalone Testing

Test PTZ streaming independently of the detection pipeline:

```bash
# Basic usage - stream for 10 seconds
python -m people_observer.test_ptz_stream ROBOT_IP --duration 10

# Stream indefinitely (Ctrl+C to stop)
python -m people_observer.test_ptz_stream 192.168.80.3

# Save stream to video file
python -m people_observer.test_ptz_stream ROBOT_IP --duration 30 --save-video output.mp4

# No display window (headless mode)
python -m people_observer.test_ptz_stream ROBOT_IP --no-display

# Custom SDP port
python -m people_observer.test_ptz_stream ROBOT_IP --sdp-port 31102

# With authentication
python -m people_observer.test_ptz_stream ROBOT_IP --username admin --password password
```

### Programmatic Usage

```python
from people_observer.ptz_stream import PtzStream, PtzStreamConfig
from people_observer.io_robot import connect_robot, configure_stream
from people_observer.config import RuntimeConfig

# Connect to robot
robot = connect_robot("192.168.80.3")

# Configure PTZ compositor and stream quality
cfg = RuntimeConfig()
configure_stream(robot, cfg)

# Create stream with custom config
stream_config = PtzStreamConfig(
    frame_queue_size=30,
    connection_timeout_sec=10.0,
)
ptz_stream = PtzStream(robot, stream_config)

# Start streaming
ptz_stream.start()

# Process frames
try:
    while ptz_stream.is_running():
        frame = ptz_stream.get_frame(timeout=1.0)
        if frame is None:
            continue
        
        # Run your facial recognition / emotion detection here
        # frame is a numpy array (H, W, 3) in BGR format
        
        # Example: Simple face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow("Faces", frame)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    ptz_stream.stop()
```

### Get Streaming Statistics

```python
stats = ptz_stream.get_stats()
print(f"Frames received: {stats['frames_received']}")
print(f"Average FPS: {stats['fps']:.1f}")
print(f"Frames dropped: {stats['frames_dropped']}")
print(f"Queue size: {stats['queue_size']}")
print(f"Duration: {stats['duration_sec']:.1f}s")
```

## Configuration

### Stream Configuration

```python
PtzStreamConfig(
    sdp_filename="h264.sdp",           # SDP endpoint on Spot CAM
    sdp_port=31102,                    # WebRTC SDP port
    cam_ssl_cert=None,                 # SSL cert path or None
    frame_queue_size=30,               # Max buffered frames (drop oldest if full)
    connection_timeout_sec=10.0,       # Timeout for ICE connection
)
```

### Compositor Screen Options

Set in `config.py` → `COMPOSITOR_SCREEN`:

- `"mech"` - Mechanical PTZ view (default)
- `"mech_full"` - Full-screen mechanical PTZ
- `"mech_overlay"` - PTZ with overlay
- `"digi"` - Digital PTZ view
- `"digi_full"` - Full-screen digital PTZ
- `"pano_full"` - Panoramic view

### Stream Quality

Set in `config.py` → `TARGET_BITRATE`:

- `2_000_000` - 2 Mbps (default, good quality)
- `4_000_000` - 4 Mbps (high quality)
- `1_000_000` - 1 Mbps (lower bandwidth)

## Integration with Detection Pipeline

### Current Status

**✅ Implemented:**
- Standalone streaming capability
- Manual start/stop control
- Frame queue for downstream processing
- Statistics and monitoring

**🚧 TODO (Future Enhancement):**
- Automatic start when person detected
- PTZ positioning before stream start
- Stream timeout if detection lost
- Frame metadata (PTZ pose, detection bbox)

### Integration Plan

The module includes extensive TODO comments in `ptz_stream.py` outlining the integration strategy:

1. **Callback-based Start/Stop**
   ```python
   # In tracker.py, after person detection:
   if best_detection and not ptz_stream.is_running():
       ptz_stream.start_on_detection(detection, camera_name)
   ```

2. **Automatic PTZ Positioning**
   ```python
   # Compute PTZ angles from detection
   pan, tilt = pixel_to_ptz_angles(detection.bbox_xywh, camera_name)
   set_ptz(ptz_client, "mech", pan, tilt)
   
   # Wait for PTZ to reach position
   time.sleep(1.0)
   
   # Start streaming
   ptz_stream.start()
   ```

3. **Stream Management**
   - Auto-stop after timeout (e.g., 30s no detection update)
   - Re-acquire if person moves significantly
   - Handle multiple simultaneous detections

4. **Frame Metadata**
   - Attach PTZ position to each frame
   - Include detection bbox in PTZ coordinates
   - Timestamp for synchronization

### Facial Recognition Integration

Example pipeline for facial recognition models:

```python
import asyncio
from people_observer.ptz_stream import PtzStream

# Your facial recognition models
face_detector = load_face_detector()
face_recognizer = load_face_recognizer()
emotion_detector = load_emotion_detector()

async def facial_recognition_pipeline(ptz_stream: PtzStream):
    """Process PTZ stream for facial recognition."""
    
    while ptz_stream.is_running():
        # Get next frame
        frame = ptz_stream.get_frame(timeout=1.0)
        if frame is None:
            continue
        
        # 1. Detect faces in frame
        faces = face_detector.detect(frame)
        
        for face_bbox in faces:
            # Extract face ROI
            x, y, w, h = face_bbox
            face_roi = frame[y:y+h, x:x+w]
            
            # 2. Recognize identity
            identity = face_recognizer.identify(face_roi)
            
            # 3. Detect emotion
            emotion = emotion_detector.predict(face_roi)
            
            # 4. Log or take action
            print(f"Detected: {identity}, Emotion: {emotion}")
            
            # TODO: Send to database, trigger alerts, etc.
        
        await asyncio.sleep(0.01)  # Yield control

# Run pipeline
asyncio.run(facial_recognition_pipeline(ptz_stream))
```

## Performance

### Typical Performance

- **Frame rate**: 15-30 fps (depends on network and bitrate)
- **Latency**: 200-500ms (WebRTC overhead + network)
- **Resolution**: 1920x1080 (configurable via stream params)
- **Format**: H.264 compressed, decoded to BGR numpy array

### Frame Queue Behavior

- **Queue size**: 30 frames (configurable)
- **Drop policy**: When full, drop oldest frame to make room
- **Thread-safe**: Can be accessed from multiple threads

### Resource Usage

- **CPU**: ~10-20% (H.264 decoding)
- **Memory**: ~200MB (frame buffers + WebRTC overhead)
- **Network**: 2 Mbps default (configurable)

## Troubleshooting

### "aiortc not installed"

```bash
pip install aiortc av
```

### "ICE connection timeout"

- Check robot IP address is correct
- Verify robot is on same network
- Check firewall allows port 31102
- Try increasing `connection_timeout_sec`

### "Failed to start stream"

- Verify compositor screen is valid (`mech_full`, `digi_full`, etc.)
- Check robot is not E-stopped
- Verify authentication token is valid
- Check Spot CAM is enabled

### High frame drops

- Increase `frame_queue_size` (default: 30)
- Lower `TARGET_BITRATE` to reduce bandwidth
- Reduce processing time per frame
- Check network quality

### Low FPS

- Increase `TARGET_BITRATE` for better quality
- Check network bandwidth
- Verify CPU is not overloaded
- Try different compositor screen

## Examples

See `test_ptz_stream.py` for complete working examples.

## Future Enhancements

- [ ] Add support for multiple simultaneous streams
- [ ] Implement adaptive bitrate based on network conditions
- [ ] Add stream recording with metadata
- [ ] Integrate with person detection pipeline
- [ ] Add facial recognition callback hooks
- [ ] Implement PTZ auto-tracking based on face position
- [ ] Add stream health monitoring and auto-restart
- [ ] Support for IR camera streaming

## References

- Boston Dynamics SDK: `spot-sdk/python/examples/spot_cam/webrtc.py`
- aiortc documentation: https://aiortc.readthedocs.io/
- WebRTC specification: https://webrtc.org/
