# PTZ WebRTC Streaming - Quick Reference

## Installation
```powershell
pip install -r requirements_webrtc.txt
```

## Test Commands
```powershell
# Basic test (10 seconds)
python -m people_observer.test_ptz_stream ROBOT_IP --duration 10

# Save video
python -m people_observer.test_ptz_stream ROBOT_IP --duration 30 --save-video ptz.mp4

# Infinite stream (Ctrl+C to stop)
python -m people_observer.test_ptz_stream ROBOT_IP

# Headless mode
python -m people_observer.test_ptz_stream ROBOT_IP --no-display
```

## Python API
```python
from people_observer.ptz_stream import PtzStream, PtzStreamConfig
from people_observer.io_robot import connect_robot, configure_stream
from people_observer.config import RuntimeConfig

# Setup
robot = connect_robot("192.168.80.3")
configure_stream(robot, RuntimeConfig())

# Create stream
stream = PtzStream(robot, PtzStreamConfig(frame_queue_size=30))

# Start streaming
stream.start()

# Get frames
while stream.is_running():
    frame = stream.get_frame(timeout=1.0)  # BGR numpy array
    if frame is not None:
        # Process frame here
        pass

# Stop streaming
stream.stop()

# Get statistics
stats = stream.get_stats()
print(f"FPS: {stats['fps']:.1f}")
```

## Frame Format
- **Type**: `numpy.ndarray`
- **Shape**: `(height, width, 3)`
- **Dtype**: `uint8`
- **Color space**: `BGR` (OpenCV format)
- **Resolution**: `1920x1080` (default)

## Key Files
- `ptz_stream.py` - Main streaming manager
- `ptz_webrtc_client.py` - WebRTC protocol implementation
- `test_ptz_stream.py` - Standalone test utility
- `PTZ_STREAMING_README.md` - Complete documentation
- `requirements_webrtc.txt` - Dependencies

## Configuration
```python
PtzStreamConfig(
    sdp_port=31102,                 # WebRTC port
    frame_queue_size=30,            # Buffer size
    connection_timeout_sec=10.0,    # Connection timeout
)
```

## Troubleshooting
| Issue | Solution |
|-------|----------|
| "aiortc not installed" | `pip install aiortc av` |
| "ICE connection timeout" | Check network, increase timeout |
| High frame drops | Increase `frame_queue_size` or lower bitrate |
| Low FPS | Check network bandwidth, verify CPU not overloaded |

## Performance
- **Frame rate**: 15-30 fps
- **Latency**: 200-500ms
- **CPU**: ~10-20% (H.264 decode)
- **Memory**: ~200MB
- **Network**: 2 Mbps (configurable)

## TODO Integration
See extensive comments in `ptz_stream.py`:
- Auto-start on detection
- PTZ positioning
- Stream timeout
- Frame metadata
- Adaptive quality

## Documentation
Full docs: `PTZ_STREAMING_README.md`
Implementation notes: `PTZ_STREAMING_IMPLEMENTATION.md`
