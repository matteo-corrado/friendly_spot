# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/18/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Abstract video source layer supporting webcam, Spot PTZ via ImageClient, and WebRTC streaming.
# Provides unified interface for PerceptionPipeline to consume frames from different sources.
# Acknowledgements: Boston Dynamics Spot SDK for ImageClient and WebRTC patterns,
# people_observer.ptz_stream for WebRTC integration

"""Video source abstraction for perception pipeline.

Supports three video sources:
1. WebcamSource: Local USB/built-in camera (development/testing)
2. SpotPTZImageClient: Spot PTZ camera via synchronous ImageClient (RECOMMENDED)
3. SpotPTZWebRTC: Spot PTZ camera via asynchronous WebRTC stream

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

logger = logging.getLogger(__name__)


class VideoSource(ABC):
    """Abstract base class for video sources."""
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame from video source.
        
        Returns:
            (success, frame): Tuple of success flag and BGR image array.
                             Returns (False, None) on error or end of stream.
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
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()
    
    def release(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            logger.info("Released webcam")


class SpotPTZImageClient(VideoSource):
    """Spot PTZ camera via synchronous ImageClient (RECOMMENDED).
    
    Fetches JPEG frames directly from robot's PTZ camera using ImageClient.
    Provides best performance for local perception pipeline:
    - 20-30 fps typical frame rate on good network
    - Low latency: ~50-100ms per frame fetch
    - Synchronous API matches pipeline structure
    - Includes camera intrinsics for advanced features
    
    Args:
        robot: Authenticated bosdyn.client.Robot instance
        source_name: Camera source name (default: "ptz" for mech PTZ)
        quality: JPEG quality percentage (default: 75)
        max_retries: Max retry attempts for failed frame fetches (default: 3)
    """
    
    def __init__(self, robot, source_name: str = "ptz", quality: int = 75, max_retries: int = 3):
        from bosdyn.client.image import ImageClient, build_image_request
        from bosdyn.api import image_pb2
        
        self.robot = robot
        self.source_name = source_name
        self.quality = quality
        self.max_retries = max_retries
        
        # Initialize ImageClient
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        
        # Validate source exists
        available_sources = {s.name for s in self.image_client.list_image_sources()}
        if source_name not in available_sources:
            raise ValueError(
                f"Camera source '{source_name}' not found. "
                f"Available sources: {sorted(available_sources)}"
            )
        
        logger.info(f"Initialized Spot PTZ ImageClient for source '{source_name}'")
        
        # Pre-build image request for efficiency
        self.image_request = build_image_request(
            source_name,
            quality_percent=quality,
            image_format=image_pb2.Image.FORMAT_JPEG
        )
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Fetch and decode single frame from PTZ camera.
        
        Returns:
            (success, frame): BGR image array or (False, None) on error
        """
        for attempt in range(self.max_retries):
            try:
                # Fetch image from robot
                responses = self.image_client.get_image([self.image_request])
                if not responses:
                    logger.warning(f"Empty response from ImageClient (attempt {attempt + 1}/{self.max_retries})")
                    continue
                
                # Decode JPEG
                img = self._decode_image(responses[0])
                if img is not None:
                    return (True, img)
                
                logger.warning(f"Failed to decode image (attempt {attempt + 1}/{self.max_retries})")
                
            except Exception as e:
                logger.warning(f"Frame fetch error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(0.1)  # Brief delay before retry
        
        logger.error(f"Failed to fetch frame after {self.max_retries} attempts")
        return (False, None)
    
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
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
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
