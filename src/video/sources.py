# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Abstract video source layer supporting webcam, Spot PTZ via ImageClient, and WebRTC streaming.
# Provides unified interface for PerceptionPipeline to consume frames from different sources.
# Key Discovery: Spot CAM uses separate 'spot-cam-image' service (not standard 'image' service).
# PTZ camera accessed via service='spot-cam-image', source='ptz'. Configure via global variables.
# Acknowledgements: Boston Dynamics Spot SDK for ImageClient and WebRTC patterns,
# people_observer.ptz_stream for WebRTC integration

"""Video source abstraction for perception pipeline.

Supports three video sources:
1. WebcamSource: Local USB/built-in camera (development/testing)
2. SpotPTZImageClient: Spot PTZ camera via synchronous ImageClient (RECOMMENDED)
3. SpotPTZWebRTC: Spot PTZ camera via asynchronous WebRTC stream

Spot CAM PTZ Camera Notes:
===========================
- Resolution: 1920x1080 (Full HD) - 6.25x more pixels than surround cameras (640x480)
- Network impact: Larger JPEG transfers (~200-500KB vs 50-100KB for fisheye)
- Processing: YOLO auto-resizes to 640x640 for inference (no manual resize needed)
- Performance: Expect ~20% slower frame rate vs surround cameras on same network
- Recommendation: Use quality=75 (default) or lower for better throughput

ImageClient vs WebRTC Comparison:
==================================

ImageClient (RECOMMENDED for perception pipeline):
- Synchronous API matches pipeline structure (no asyncio complexity)
- Lower latency: direct JPEG fetch (~50-100ms per frame)
- Frame rate: Can request 10-30 fps depending on network and processing
- Includes camera intrinsics and depth data (for advanced features)
- Simpler error handling and retry logic
- Used by people_observer detection pipeline successfully

WebRTC (Use for remote viewing or bandwidth-constrained scenarios):
- Asynchronous streaming via aiortc (requires separate thread + event loop)
- Higher latency: SDP negotiation + H.264 decode (~100-200ms startup + 50ms per frame)
- Frame rate: 15-30 fps from robot's H.264 encoder
- No metadata (intrinsics, depth, timestamps)
- Complex connection management (SDP offer/answer, ICE)
- Best for remote internet streaming or low-bandwidth scenarios

For local perception pipeline, ImageClient provides:
- 20-30 fps typical performance on good network
- Direct integration with robot's camera system
- Ability to request specific image quality/format
- Synchronous frame fetch fits naturally with perception loop
"""

import sys
import time
import logging
from typing import Optional, Tuple
from abc import ABC, abstractmethod

import cv2
import numpy as np

# ==============================================================================
# SPOT CAM CAMERA CONFIGURATION
# ==============================================================================
# Customize these variables to select different Spot CAM cameras
# Run test_list_image_sources.py to see all available sources on your robot

# Spot CAM image service name (separate from standard 'image' service)
SPOT_CAM_IMAGE_SERVICE = 'spot-cam-image'

# PTZ camera source (mechanical pan/tilt/zoom camera)
# Resolution: 1920x1080 (Full HD) - significantly higher than surround cameras (640x480)
SPOT_CAM_PTZ_SOURCE = 'ptz'

# 360-degree panoramic camera source
SPOT_CAM_360_SOURCE = 'pano'

# Individual ring cameras (if needed)
SPOT_CAM_RING_CAMERAS = ['c0', 'c1', 'c2', 'c3', 'c4']

# Compositor stream (composite view of multiple cameras)
SPOT_CAM_STREAM_SOURCE = 'stream'

# ==============================================================================
# SURROUND CAMERA CONFIGURATION
# ==============================================================================
# Standard Spot fisheye cameras (from 'image' service)
SURROUND_IMAGE_SERVICE = 'image'
SURROUND_CAMERA_SOURCES = [
    'frontleft_fisheye_image',
    'frontright_fisheye_image',
    'left_fisheye_image',
    'right_fisheye_image',
    'back_fisheye_image'
]

# ==============================================================================

logger = logging.getLogger(__name__)


class VideoSource(ABC):
    """Abstract base class for video sources."""
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Read single frame from video source.
        
        Returns:
            (success, visual_frame, depth_frame): Tuple of success flag, BGR image, and depth in meters (or None).
                                                   Returns (False, None, None) on error or end of stream.
        """
        pass
    
    @abstractmethod
    def release(self):
        """Release video source resources."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class WebcamSource(VideoSource):
    """Local webcam video source (fallback for development/testing)."""
    
    def __init__(self, device: int = 0):
        """Initialize webcam capture.
        
        Args:
            device: Camera device index (0 for default camera)
        """
        # Platform-specific backends for best reliability
        if sys.platform == 'win32':
            # Windows: DirectShow backend (most reliable)
            self.cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
        elif sys.platform == 'darwin':
            # macOS: AVFoundation backend (native)
            self.cap = cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
        else:
            # Linux: V4L2 backend (default, works well)
            self.cap = cv2.VideoCapture(device)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam device {device} on {sys.platform}")
        
        logger.info(f"Opened webcam device {device}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        return (ret, frame, None)  # Webcam has no depth
    
    def release(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            logger.info("Released webcam")


class SpotPTZImageClient(VideoSource):
    """Spot PTZ camera via synchronous ImageClient (Spot CAM image service).
    
    Fetches JPEG frames from Spot CAM PTZ camera using the 'spot-cam-image' service.
    Uses global configuration variables (SPOT_CAM_IMAGE_SERVICE, SPOT_CAM_PTZ_SOURCE)
    for easy customization. Provides best performance for local perception pipeline:
    - 20-30 fps typical frame rate on good network
    - Low latency: ~50-100ms per frame fetch
    - Synchronous API matches pipeline structure
    - Includes camera intrinsics for advanced features
    
    Configuration:
        Modify global variables at top of file:
        - SPOT_CAM_IMAGE_SERVICE: Image service name (default: 'spot-cam-image')
        - SPOT_CAM_PTZ_SOURCE: Camera source name (default: 'ptz')
        - SPOT_CAM_360_SOURCE: 360 camera (default: 'pano')
    
    Args:
        robot: Authenticated bosdyn.client.Robot instance with Spot CAM registered
        image_service: Image service name (default: uses SPOT_CAM_IMAGE_SERVICE)
        source_name: Camera source name (default: uses SPOT_CAM_PTZ_SOURCE)
        quality: JPEG quality percentage (default: 75)
        max_retries: Max retry attempts for failed frame fetches (default: 3)
        include_depth: Also fetch depth image if available (default: False)
                       Note: Spot CAM PTZ typically does not have depth
    
    Raises:
        ValueError: If image_service or source_name not found on robot
    """
    
    def __init__(self, 
                 robot, 
                 image_service: Optional[str] = None,
                 source_name: Optional[str] = None, 
                 quality: int = 75, 
                 max_retries: int = 3, 
                 include_depth: bool = False):
        from bosdyn.client.image import ImageClient, build_image_request
        from bosdyn.api import image_pb2
        
        self.robot = robot
        self.quality = quality
        self.max_retries = max_retries
        self.include_depth = include_depth
        
        # Use global config defaults if not specified
        if image_service is None:
            image_service = SPOT_CAM_IMAGE_SERVICE
        if source_name is None:
            source_name = SPOT_CAM_PTZ_SOURCE
        
        self.image_service_name = image_service
        self.source_name = source_name
        
        # Initialize ImageClient for specified service
        # For Spot CAM: use 'spot-cam-image' service instead of default 'image' service
        logger.info(f"Connecting to image service: {image_service}")
        try:
            self.image_client = robot.ensure_client(image_service)
            
            # Validate source exists in this service
            available_sources = {s.name for s in self.image_client.list_image_sources()}
            logger.debug(f"Available sources in '{image_service}': {sorted(available_sources)}")
        except Exception as e:
            logger.error(f"Failed to connect to image service '{image_service}': {e}")
            if image_service == SPOT_CAM_IMAGE_SERVICE:
                logger.error("Spot CAM PTZ service unavailable - attempting fallback cameras...")
                # Fallback to standard image service
                self.image_client = robot.ensure_client(ImageClient.default_service_name)
                available_sources = {s.name for s in self.image_client.list_image_sources()}
                logger.info(f"Available sources in standard service: {sorted(available_sources)}")
                
                # Preferred fallback order: hand camera > pano camera > front fisheye
                fallback_cameras = ['hand_color_image', 'pano_image', 'frontleft_fisheye_image']
                source_name = None
                for camera in fallback_cameras:
                    if camera in available_sources:
                        source_name = camera
                        logger.info(f"Using fallback camera: {source_name}")
                        break
                
                if source_name is None:
                    raise RuntimeError(
                        f"No suitable camera sources available. "
                        f"Tried: {fallback_cameras}. "
                        f"Found: {sorted(available_sources)}"
                    )
            else:
                raise
        
        if source_name not in available_sources:
            raise ValueError(
                f"Camera source '{source_name}' not found in service '{image_service}'.\n"
                f"Available sources: {sorted(available_sources)}\n\n"
                "To see all services and sources, run:\n"
                "  python test_list_image_sources.py --hostname ROBOT_IP\n\n"
                "To use a different source, modify global variables at top of video_sources.py:\n"
                f"  SPOT_CAM_IMAGE_SERVICE = '{image_service}'\n"
                f"  SPOT_CAM_PTZ_SOURCE = 'your_source_name'"
            )
        
        logger.info(f"Initialized Spot PTZ ImageClient: service='{image_service}', source='{source_name}', depth={include_depth}")
        
        # Pre-build image requests for efficiency
        self.image_request = build_image_request(
            source_name,
            quality_percent=quality,
            image_format=image_pb2.Image.FORMAT_JPEG
        )
        
        # Build depth request if enabled
        self.depth_request = None
        if include_depth:
            # Construct depth source name (e.g., "ptz" -> "ptz_depth_in_visual_frame")
            # Note: PTZ camera may not have depth - this will fail gracefully
            depth_source_name = f"{source_name}_depth_in_visual_frame"
            logger.debug(f"Looking for depth source: {depth_source_name}")
            logger.debug(f"Available sources: {sorted(available_sources)}")
            if depth_source_name in available_sources:
                self.depth_request = build_image_request(
                    depth_source_name,
                    pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
                    image_format=image_pb2.Image.FORMAT_RAW
                )
                logger.info(f"Depth source '{depth_source_name}' available and configured")
            else:
                logger.warning(f"Depth source '{depth_source_name}' not available, depth will be None")
                logger.debug(f"Depth sources that exist: {[s for s in available_sources if 'depth' in s.lower()]}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Fetch and decode frame(s) from PTZ camera.
        
        Returns:
            (success, visual_frame, depth_frame): BGR image, depth in meters (or None), or (False, None, None) on error
        """
        for attempt in range(self.max_retries):
            try:
                # Build request list
                requests = [self.image_request]
                if self.depth_request is not None:
                    requests.append(self.depth_request)
                
                # Fetch image(s) from robot
                responses = self.image_client.get_image(requests)
                if not responses:
                    logger.warning(f"Empty response from ImageClient (attempt {attempt + 1}/{self.max_retries})")
                    continue
                
                # Decode visual image
                logger.debug(f"Decoding visual image from response (format: {responses[0].shot.image.format})")
                img = self._decode_image(responses[0])
                if img is None:
                    logger.warning(f"Failed to decode visual image (attempt {attempt + 1}/{self.max_retries})")
                    continue
                logger.debug(f"Visual image decoded: {img.shape[1]}x{img.shape[0]} pixels")
                
                # Decode depth image if requested
                depth_img = None
                if self.depth_request is not None and len(responses) > 1:
                    logger.debug(f"Decoding depth image from response (format: {responses[1].shot.image.pixel_format})")
                    depth_img = self._decode_depth_image(responses[1])
                    if depth_img is None:
                        logger.debug("Depth image unavailable (continuing with visual only)")
                    else:
                        valid_pixels = np.sum(~np.isnan(depth_img))
                        total_pixels = depth_img.shape[0] * depth_img.shape[1]
                        logger.debug(f"Depth image decoded: {depth_img.shape[1]}x{depth_img.shape[0]} pixels, {valid_pixels}/{total_pixels} valid ({100*valid_pixels/total_pixels:.1f}%)")
                elif self.depth_request is not None:
                    logger.debug(f"Depth requested but only {len(responses)} responses received (expected 2)")
                
                return (True, img, depth_img)
                
            except Exception as e:
                logger.warning(f"Frame fetch error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(0.1)  # Brief delay before retry
        
        logger.error(f"Failed to fetch frame after {self.max_retries} attempts")
        return (False, None, None)
    
    def _decode_image(self, response) -> Optional[np.ndarray]:
        """Decode ImageResponse to BGR numpy array."""
        from bosdyn.api import image_pb2
        
        img_data = response.shot.image.data
        img_format = response.shot.image.format
        
        if img_format == image_pb2.Image.FORMAT_JPEG:
            # Decode JPEG
            buf = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return img
        
        elif img_format == image_pb2.Image.FORMAT_RAW:
            # Decode RAW (grayscale to BGR)
            rows = response.shot.image.rows
            cols = response.shot.image.cols
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(rows, cols)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        
        else:
            logger.error(f"Unsupported image format: {img_format}")
            return None
    
    def _decode_depth_image(self, response) -> Optional[np.ndarray]:
        """Decode depth ImageResponse to float32 ndarray in meters.
        
        Args:
            response: ImageResponse with PIXEL_FORMAT_DEPTH_U16 depth data
        
        Returns:
            np.ndarray of shape (rows, cols) with depth in meters, or None if unavailable
            Invalid depth pixels (0 and 65535) are set to NaN
        """
        from bosdyn.api import image_pb2
        
        # Verify this is depth format
        if response.shot.image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            logger.warning(f"Expected DEPTH_U16 format, got {response.shot.image.pixel_format}")
            return None
        
        rows = response.shot.image.rows
        cols = response.shot.image.cols
        data = response.shot.image.data
        logger.debug(f"Decoding depth image: {cols}x{rows} pixels, {len(data)} bytes")
        
        # Decode uint16 depth data
        depth_u16 = np.frombuffer(data, dtype=np.uint16).reshape(rows, cols)
        
        # Get scale factor from ImageSource (converts uint16 -> meters)
        depth_scale = response.source.depth_scale if response.source.depth_scale > 0 else 1.0
        logger.debug(f"Depth scale factor: {depth_scale} (1 uint16 unit = {1.0/depth_scale:.6f} meters)")
        
        # Convert to float32 meters
        depth_m = depth_u16.astype(np.float32) / depth_scale
        
        # Mask out invalid depth values (0 and 65535 = no data)
        invalid_mask = (depth_u16 == 0) | (depth_u16 == 65535)
        invalid_count = np.sum(invalid_mask)
        depth_m[invalid_mask] = np.nan
        
        # Log depth statistics
        valid_depths = depth_m[~np.isnan(depth_m)]
        if len(valid_depths) > 0:
            logger.debug(f"Depth stats: min={np.min(valid_depths):.2f}m, max={np.max(valid_depths):.2f}m, mean={np.mean(valid_depths):.2f}m, {invalid_count} invalid pixels")
        else:
            logger.warning(f"All {rows*cols} depth pixels are invalid (NaN)")
        
        return depth_m
    
    def release(self):
        """Release resources (ImageClient managed by SDK)."""
        logger.info(f"Released Spot PTZ ImageClient for '{self.source_name}'")


class SpotPTZWebRTC(VideoSource):
    """Spot PTZ camera via WebRTC streaming (alternative to ImageClient).
    
    Uses aiortc WebRTC protocol for H.264 video streaming. Runs in separate
    thread with asyncio event loop. More complex than ImageClient but useful
    for remote streaming scenarios.
    
    Typical performance:
    - 15-30 fps from robot's H.264 encoder
    - ~100-200ms startup latency for SDP negotiation
    - ~50ms per-frame latency after connection established
    
    Args:
        robot: Authenticated bosdyn.client.Robot instance
        timeout: Connection timeout in seconds (default: 10.0)
    
    TODO: Implement WebRTC integration using people_observer.ptz_stream module
    """
    
    def __init__(self, robot, timeout: float = 10.0):
        self.robot = robot
        self.timeout = timeout
        self.stream = None
        
        # TODO: Initialize PtzStream from people_observer.ptz_stream
        # from people_observer.ptz_stream import PtzStream, PtzStreamConfig
        # config = PtzStreamConfig(robot=robot, screen='mech_full', bitrate_bps=2000000)
        # self.stream = PtzStream(config)
        # self.stream.start()
        
        logger.warning("SpotPTZWebRTC not yet implemented - use SpotPTZImageClient instead")
        raise NotImplementedError(
            "WebRTC video source not yet implemented. "
            "Use SpotPTZImageClient for PTZ camera access. "
            "See TODO in video_sources.py for integration steps."
        )
    
    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Get next frame from WebRTC stream queue.
        
        TODO: Implement frame queue polling
        """
        # TODO: Poll frame from self.stream.frame_queue with timeout
        # try:
        #     frame = self.stream.frame_queue.get(timeout=1.0)
        #     return (True, frame)
        # except queue.Empty:
        #     return (False, None)
        
        raise NotImplementedError("WebRTC read() not implemented")
    
    def release(self):
        """Stop WebRTC stream and release resources.
        
        TODO: Implement stream shutdown
        """
        # TODO: Call self.stream.stop() and join worker thread
        # if self.stream:
        #     self.stream.stop()
        #     logger.info("Stopped WebRTC stream")
        
        pass


def create_video_source(mode: str, robot=None, **kwargs) -> VideoSource:
    """Factory function to create video source based on mode.
    
    Args:
        mode: Video source mode ('webcam', 'imageclient', 'webrtc')
        robot: Robot instance (required for 'imageclient' and 'webrtc' modes)
        **kwargs: Additional arguments passed to VideoSource constructor
    
    Returns:
        VideoSource instance
    
    Raises:
        ValueError: If mode is invalid or robot is missing for robot modes
    
    Examples:
        >>> # Development with webcam
        >>> source = create_video_source('webcam', device=0)
        
        >>> # Production with robot PTZ via ImageClient (RECOMMENDED)
        >>> source = create_video_source('imageclient', robot=robot, source_name='ptz')
        
        >>> # Remote streaming with WebRTC (TODO)
        >>> source = create_video_source('webrtc', robot=robot)
    """
    mode = mode.lower()
    
    if mode == 'webcam':
        return WebcamSource(**kwargs)
    
    elif mode == 'imageclient':
        if robot is None:
            raise ValueError("Robot instance required for 'imageclient' mode")
        return SpotPTZImageClient(robot, **kwargs)
    
    elif mode == 'webrtc':
        if robot is None:
            raise ValueError("Robot instance required for 'webrtc' mode")
        return SpotPTZWebRTC(robot, **kwargs)
    
    else:
        raise ValueError(
            f"Invalid video source mode: '{mode}'. "
            f"Valid modes: 'webcam', 'imageclient', 'webrtc'"
        )
