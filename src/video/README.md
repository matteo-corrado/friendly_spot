# Video Module

Video streaming and camera source abstraction for Spot cameras.

## Overview

The video module provides video capture interfaces for Spot's cameras, including standard image sources (fisheye, hand, panoramic) and PTZ (pan-tilt-zoom) camera streaming via WebRTC. Handles camera availability, fallback logic, and format conversion.

## Components

### Image Sources
- **`sources.py`**: Camera source abstraction
  - `SpotImageClient`: Standard camera image acquisition (JPEG → BGR)
  - `SpotPTZImageClient`: PTZ camera with automatic fallback (hand → pano → fisheye)
  - `WebcamSource`: Laptop webcam for testing without robot
  - Common interface: `get_frame()` → BGR numpy array

### PTZ Streaming
- **`ptz_stream.py`**: PTZ camera stream configuration
  - Setup compositor screens for PTZ views
  - Configure stream quality (bitrate, framerate, compression)
  - Used with WebRTC for remote viewing

- **`webrtc_client.py`**: WebRTC client for PTZ video
  - Real-time PTZ video streaming over WebRTC
  - Browser-based viewing or programmatic frame capture
  - Handles offer/answer SDP negotiation

## Usage

### Standard Camera Capture

```python
from src.robot import create_robot
from src.video import SpotImageClient

robot = create_robot("192.168.80.3")

# Single camera
camera = SpotImageClient(
    robot=robot,
    source_name="frontleft_fisheye_image"
)

while True:
    frame_bgr = camera.get_frame()  # Returns (H, W, 3) BGR array
    cv2.imshow("Spot Camera", frame_bgr)
    if cv2.waitKey(1) == ord('q'):
        break
```

### Multi-Camera Capture

```python
from src.video import SpotImageClient

cameras = [
    SpotImageClient(robot, "frontleft_fisheye_image"),
    SpotImageClient(robot, "frontright_fisheye_image"),
    SpotImageClient(robot, "left_fisheye_image"),
]

while True:
    frames = [cam.get_frame() for cam in cameras]
    # Process frames...
```

### PTZ Camera with Fallback

```python
from src.video import SpotPTZImageClient

# Automatically tries: PTZ → hand camera → panoramic → fisheye
ptz_camera = SpotPTZImageClient(robot)

frame = ptz_camera.get_frame()
print(f"Using camera: {ptz_camera.source_name}")  # e.g., "hand_color_image"
```

### Webcam Testing (No Robot)

```python
from src.video import WebcamSource

# Test code without robot hardware
webcam = WebcamSource(device_id=0)  # Default webcam

while True:
    frame = webcam.get_frame()
    cv2.imshow("Webcam Test", frame)
    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
```

### PTZ WebRTC Streaming

```python
from src.robot import create_robot, RobotClients
from src.video.ptz_stream import setup_ptz_stream
from src.video.webrtc_client import WebRTCClient

robot = create_robot("192.168.80.3", register_spot_cam=True)
clients = RobotClients(robot)

# Configure PTZ stream
setup_ptz_stream(
    compositor_client=clients.compositor,
    stream_quality_client=clients.stream_quality,
    ptz_client=clients.ptz
)

# Start WebRTC stream
webrtc = WebRTCClient(robot)
webrtc.start()

# View at: http://<robot-ip>:31102/h264.sdp.html
```

## Camera Sources

### Standard Cameras

Available on all Spot robots:

| Source Name | Location | Resolution | FPS | FOV |
|-------------|----------|------------|-----|-----|
| `frontleft_fisheye_image` | Front-left surround | 640x480 | 15 | 180° |
| `frontright_fisheye_image` | Front-right surround | 640x480 | 15 | 180° |
| `left_fisheye_image` | Left surround | 640x480 | 15 | 180° |
| `right_fisheye_image` | Right surround | 640x480 | 15 | 180° |
| `back_fisheye_image` | Rear surround | 640x480 | 15 | 180° |

### Spot CAM Cameras

Only on robots with Spot CAM:

| Source Name | Location | Resolution | FPS | Notes |
|-------------|----------|------------|-----|-------|
| `hand_color_image` | Gripper camera | 640x480 | 30 | Color RGB |
| `hand_depth_image` | Gripper depth | 224x171 | 30 | Depth map (mm) |
| `pano_image` | PTZ panoramic | 1920x1080 | 30 | Wide angle |
| PTZ stream | PTZ camera | 1920x1080 | 30 | Via WebRTC only |

### Depth Sensors

Depth images for distance estimation:

| Source Name | Location | Resolution | Format |
|-------------|----------|------------|--------|
| `frontleft_depth_in_visual_frame` | Front-left | 424x240 | Depth (meters) |
| `frontright_depth_in_visual_frame` | Front-right | 424x240 | Depth (meters) |
| `hand_depth_image` | Gripper | 224x171 | Depth (mm) |

## Camera Fallback Logic

`SpotPTZImageClient` implements automatic fallback for robustness:

1. **Try PTZ**: `spot-cam-image` service (requires Spot CAM hardware)
2. **Fallback to Hand**: `hand_color_image` (requires Spot CAM)
3. **Fallback to Pano**: `pano_image` (requires Spot CAM)
4. **Fallback to Fisheye**: `frontleft_fisheye_image` (always available)

Logs fallback path:

```
WARNING: PTZ image service unavailable: ServiceUnavailableError
INFO: Falling back to hand_color_image
```

Override fallback order:

```python
ptz_camera = SpotPTZImageClient(
    robot,
    fallback_sources=["pano_image", "hand_color_image"]  # Custom priority
)
```

## Format Conversion

All sources return **BGR** numpy arrays for OpenCV compatibility:

```python
# Image acquisition handles format conversion:
# JPEG → RGB → BGR (via cv2.imdecode)
frame_bgr = camera.get_frame()  # (H, W, 3) uint8 BGR

# Convert to other formats:
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
```

Depth images are float32 meters:

```python
depth_source = SpotImageClient(robot, "frontleft_depth_in_visual_frame")
depth_map = depth_source.get_frame()  # (H, W) float32, values in meters
# NaN values indicate invalid depth
```

## Configuration

### Image Compression Quality

Control JPEG compression (affects bandwidth/latency):

```python
# In sources.py
JPEG_QUALITY = 85  # Higher = better quality, larger size (0-100)
```

### PTZ Stream Quality

Configure bitrate and framerate:

```python
from src.video.ptz_stream import setup_ptz_stream

setup_ptz_stream(
    compositor_client=clients.compositor,
    stream_quality_client=clients.stream_quality,
    ptz_client=clients.ptz,
    bitrate_kbps=4000,  # Higher = better quality, more bandwidth
    framerate_fps=30     # Max 30 FPS for PTZ
)
```

### WebRTC Ports

Default WebRTC ports (ensure firewall allows):

- **SDP**: 31102 (HTTPS)
- **Video**: UDP ports negotiated dynamically

## Troubleshooting

### No Image from PTZ
- **Spot CAM not registered**: Use `create_robot(hostname, register_spot_cam=True)`
- **Service unavailable**: Check robot has Spot CAM hardware with `robot.list_services()`
- **Fallback successful**: Check logs for fallback to hand/pano/fisheye

### Slow Frame Rate
- **Network congestion**: Reduce JPEG quality or image resolution
- **CPU bottleneck**: Use hardware JPEG decoding if available
- **Multiple cameras**: Fetch frames in parallel or reduce camera count

### Depth Image All NaN
- **Out of range**: Depth sensors have limited range (0.2m - 10m for hand depth)
- **No returns**: Reflective or transparent surfaces may not return depth
- **Service issue**: Verify depth source available with `image_client.list_image_sources()`

### WebRTC Connection Fails
- **Firewall**: Allow port 31102 and UDP for WebRTC
- **Network routing**: Ensure client can reach robot IP
- **Browser compatibility**: Use Chrome/Edge (best WebRTC support)

## Performance Optimization

### Batch Frame Capture

Fetch multiple cameras simultaneously:

```python
from bosdyn.client.image import ImageClient

image_client = robot.ensure_client(ImageClient.default_service_name)

# Request multiple sources in one call
sources = ["frontleft_fisheye_image", "frontright_fisheye_image"]
image_responses = image_client.get_image_from_sources(sources)

for response in image_responses:
    # Decode JPEG
    img_data = np.frombuffer(response.shot.image.data, dtype=np.uint8)
    frame_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
```

### Caching Image Client

Reuse `ImageClient` instead of creating per-request:

```python
# Good: Reuse client
image_client = robot.ensure_client(ImageClient.default_service_name)
for i in range(100):
    response = image_client.get_image_from_sources(["frontleft_fisheye_image"])

# Bad: Recreate client each time
for i in range(100):
    image_client = robot.ensure_client(ImageClient.default_service_name)  # Slow!
    response = image_client.get_image_from_sources(["frontleft_fisheye_image"])
```

### Downsampling

Reduce resolution for faster processing:

```python
frame = camera.get_frame()
frame_small = cv2.resize(frame, (320, 240))  # Half resolution
# Process frame_small...
```

## Dependencies

- `opencv-python` >= 4.8.0: Image decoding and conversion
- `numpy` >= 1.24.0: Array operations
- `bosdyn-client` == 5.0.1.2: Spot SDK image service
- `aiortc` (optional): WebRTC client for PTZ streaming

## References

- [Spot SDK Image Service](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/image)
- [Spot CAM WebRTC](https://dev.bostondynamics.com/docs/payload/spot_cam_webrtc)
- [Image Sources Documentation](https://dev.bostondynamics.com/docs/concepts/cameras)
- OpenCV Image Decoding: [cv2.imdecode()](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga26a67788faa58ade337f8d28ba0eb19e)
