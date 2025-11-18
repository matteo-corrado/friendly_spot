# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/18/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Independent WebRTC video streaming module for PTZ camera providing real-time H.264 frames
# in thread-safe queue for downstream facial recognition and emotion detection processing
# Acknowledgements: Boston Dynamics Spot SDK webrtc.py example for WebRTC patterns,
# aiortc documentation for async streaming, Claude for thread-safe queue design and TODO integration hooks

"""WebRTC video streaming from Spot PTZ camera.

This module provides real-time H.264 video streaming from the PTZ camera via WebRTC.
Designed as a separate component that can be enabled/disabled independently of the
person detection pipeline.

Architecture:
- Runs in separate thread with dedicated asyncio event loop
- Provides frame queue for downstream processing (facial recognition, emotion detection)
- Clean start/stop interface for integration

Usage:
    stream = PtzStream(robot, options)
    stream.start()
    
    # Process frames
    while stream.is_running():
        frame = await stream.get_frame(timeout=1.0)
        if frame:
            # Process with facial recognition models
            pass
    
    stream.stop()

Functions:
- PtzStream: Main WebRTC streaming manager
- start(): Begin streaming in background thread
- stop(): Gracefully shutdown stream
- get_frame(): Retrieve next video frame (async)
- is_running(): Check if stream is active

TODO: Integration with person detection pipeline
    - Add callback hook to start stream when person detected
    - Add automatic PTZ positioning before stream start
    - Add timeout to stop stream if person lost
    - Add stream quality adjustment based on detection confidence
    - Add frame metadata (timestamp, PTZ position, detection bbox)
"""
import asyncio
import logging
import queue
import threading
import time
from typing import Optional, Callable

import cv2
import numpy as np

try:
    from aiortc import RTCConfiguration
    from aiortc.contrib.media import MediaBlackhole
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    logging.warning("aiortc not installed. WebRTC streaming unavailable. Install with: pip install aiortc")

from bosdyn.client import Robot

logger = logging.getLogger(__name__)


class PtzStreamConfig:
    """Configuration for PTZ WebRTC streaming."""
    
    def __init__(
        self,
        sdp_filename: str = "h264.sdp",
        sdp_port: int = 31102,
        cam_ssl_cert: Optional[str] = None,
        frame_queue_size: int = 30,
        connection_timeout_sec: float = 10.0,
    ):
        """Initialize PTZ stream configuration.
        
        Args:
            sdp_filename: SDP endpoint on Spot CAM (default: h264.sdp)
            sdp_port: Port for SDP negotiation (default: 31102)
            cam_ssl_cert: Path to SSL cert for Spot CAM, or None to disable verification
            frame_queue_size: Maximum frames to buffer before dropping
            connection_timeout_sec: Timeout for WebRTC connection establishment
        """
        self.sdp_filename = sdp_filename
        self.sdp_port = sdp_port
        self.cam_ssl_cert = cam_ssl_cert if cam_ssl_cert else False
        self.frame_queue_size = frame_queue_size
        self.connection_timeout_sec = connection_timeout_sec


class PtzStream:
    """WebRTC video stream manager for Spot PTZ camera.
    
    Manages WebRTC connection in separate thread, provides frame queue for processing.
    """
    
    def __init__(self, robot: Robot, config: Optional[PtzStreamConfig] = None):
        """Initialize PTZ stream manager.
        
        Args:
            robot: Authenticated Spot robot instance
            config: Stream configuration, or None for defaults
        """
        if not AIORTC_AVAILABLE:
            raise RuntimeError(
                "aiortc library required for WebRTC streaming. "
                "Install with: pip install aiortc av"
            )
        
        self.robot = robot
        self.config = config or PtzStreamConfig()
        
        # Thread management
        self._stream_thread: Optional[threading.Thread] = None
        self._shutdown_flag = threading.Event()
        self._running_flag = threading.Event()
        
        # Frame queue (thread-safe)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.config.frame_queue_size)
        
        # WebRTC client (created in stream thread)
        self._webrtc_client = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Statistics
        self._frames_received = 0
        self._frames_dropped = 0
        self._start_time: Optional[float] = None
        
        logger.info(f"PtzStream initialized (queue_size={self.config.frame_queue_size})")
    
    def start(self) -> bool:
        """Start WebRTC streaming in background thread.
        
        Returns:
            True if stream started successfully, False otherwise
        """
        if self.is_running():
            logger.warning("Stream already running")
            return False
        
        # Reset state
        self._shutdown_flag.clear()
        self._running_flag.clear()
        self._frames_received = 0
        self._frames_dropped = 0
        self._start_time = time.time()
        
        # Clear any stale frames
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Start streaming thread
        self._stream_thread = threading.Thread(
            target=self._stream_worker,
            name="PtzStreamThread",
            daemon=True
        )
        self._stream_thread.start()
        
        # Wait for connection to establish (with timeout)
        start_wait = time.time()
        while not self._running_flag.is_set():
            if time.time() - start_wait > self.config.connection_timeout_sec:
                logger.error(f"Stream connection timeout after {self.config.connection_timeout_sec}s")
                self.stop()
                return False
            time.sleep(0.1)
        
        logger.info("PTZ stream started successfully")
        return True
    
    def stop(self):
        """Stop WebRTC streaming and cleanup resources."""
        if not self._stream_thread or not self._stream_thread.is_alive():
            logger.debug("Stream not running, nothing to stop")
            return
        
        logger.info("Stopping PTZ stream...")
        self._shutdown_flag.set()
        
        # Wait for thread to finish (with timeout)
        self._stream_thread.join(timeout=5.0)
        if self._stream_thread.is_alive():
            logger.warning("Stream thread did not terminate cleanly")
        
        self._running_flag.clear()
        
        # Log statistics
        if self._start_time:
            duration = time.time() - self._start_time
            fps = self._frames_received / duration if duration > 0 else 0
            logger.info(
                f"Stream stopped. Stats: {self._frames_received} frames, "
                f"{self._frames_dropped} dropped, {fps:.1f} fps, {duration:.1f}s duration"
            )
    
    def is_running(self) -> bool:
        """Check if stream is currently active.
        
        Returns:
            True if streaming, False otherwise
        """
        return self._running_flag.is_set()
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Retrieve next video frame from stream (blocking).
        
        Args:
            timeout: Maximum seconds to wait for frame
            
        Returns:
            BGR image as numpy array, or None if timeout/stream stopped
        """
        try:
            frame = self._frame_queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None
    
    def get_stats(self) -> dict:
        """Get streaming statistics.
        
        Returns:
            Dict with keys: frames_received, frames_dropped, fps, duration_sec, queue_size
        """
        duration = time.time() - self._start_time if self._start_time else 0
        fps = self._frames_received / duration if duration > 0 else 0
        
        return {
            "frames_received": self._frames_received,
            "frames_dropped": self._frames_dropped,
            "fps": fps,
            "duration_sec": duration,
            "queue_size": self._frame_queue.qsize(),
            "is_running": self.is_running(),
        }
    
    def _stream_worker(self):
        """WebRTC streaming worker (runs in separate thread)."""
        # Create new event loop for this thread
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)
        
        try:
            # Import WebRTC client (must be in this thread's context)
            from .ptz_webrtc_client import SpotPtzWebRTCClient
            
            # Create WebRTC client
            self._webrtc_client = SpotPtzWebRTCClient(
                hostname=self.robot.address,
                token=self.robot.user_token,
                sdp_port=self.config.sdp_port,
                sdp_filename=self.config.sdp_filename,
                cam_ssl_cert=self.config.cam_ssl_cert,
            )
            
            # Run streaming coroutines
            self._event_loop.run_until_complete(
                self._stream_coroutine()
            )
        except Exception as e:
            logger.error(f"Stream worker error: {e}", exc_info=True)
        finally:
            # Cleanup
            if self._event_loop:
                self._event_loop.close()
            self._running_flag.clear()
    
    async def _stream_coroutine(self):
        """Main streaming coroutine (runs in event loop)."""
        try:
            # Start WebRTC connection
            await self._webrtc_client.start()
            
            # Wait for ICE connection
            logger.info("Waiting for WebRTC connection...")
            connection_start = time.time()
            while self._webrtc_client.pc.iceConnectionState != 'completed':
                if time.time() - connection_start > self.config.connection_timeout_sec:
                    raise TimeoutError("WebRTC ICE connection timeout")
                await asyncio.sleep(0.1)
            
            logger.info(f"WebRTC connected (state={self._webrtc_client.pc.iceConnectionState})")
            self._running_flag.set()
            
            # Process video frames
            await self._process_frames()
            
        except Exception as e:
            logger.error(f"Streaming coroutine error: {e}", exc_info=True)
        finally:
            # Cleanup WebRTC connection
            if self._webrtc_client:
                await self._webrtc_client.stop()
    
    async def _process_frames(self):
        """Process incoming video frames from WebRTC stream."""
        logger.info("Processing video frames...")
        
        while not self._shutdown_flag.is_set():
            try:
                # Get frame from WebRTC client (with timeout)
                frame = await asyncio.wait_for(
                    self._webrtc_client.video_frame_queue.get(),
                    timeout=1.0
                )
                
                # Convert AVFrame to numpy array (BGR for OpenCV)
                img = frame.to_ndarray(format='bgr24')
                
                self._frames_received += 1
                
                # Put frame in queue (non-blocking)
                try:
                    self._frame_queue.put_nowait(img)
                except queue.Full:
                    # Drop oldest frame and retry
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(img)
                        self._frames_dropped += 1
                    except:
                        pass
                
                # Log stats periodically
                if self._frames_received % 100 == 0:
                    stats = self.get_stats()
                    logger.debug(
                        f"Stream stats: {stats['frames_received']} frames, "
                        f"{stats['fps']:.1f} fps, {stats['frames_dropped']} dropped"
                    )
                
            except asyncio.TimeoutError:
                # No frame received, check if we should continue
                if self._shutdown_flag.is_set():
                    break
                logger.debug("No frame received (timeout)")
            except Exception as e:
                logger.error(f"Frame processing error: {e}", exc_info=True)
                break
        
        logger.info("Frame processing stopped")


# TODO: Integration with person detection pipeline
# ================================================
# 
# Future enhancements to integrate PTZ streaming with detection:
#
# 1. CALLBACK-BASED START/STOP:
#    - Add start_on_detection(detection, camera_name) method
#    - Automatically start stream when person detected in fisheye cameras
#    - Include detection metadata (bbox, confidence, camera source)
#
# 2. AUTOMATIC PTZ POSITIONING:
#    - Before starting stream, command PTZ to detection location
#    - Use geometry.pixel_to_ptz_angles() to compute pan/tilt
#    - Wait for PTZ movement to complete before streaming
#
# 3. STREAM MANAGEMENT:
#    - Auto-stop stream after timeout (e.g., 30s no detection update)
#    - Re-acquire if person moves to different location
#    - Handle multiple simultaneous detections (prioritize nearest)
#
# 4. FRAME METADATA:
#    - Attach PTZ position to each frame (pan, tilt, zoom)
#    - Attach detection bbox in PTZ frame coordinates
#    - Attach timestamp for synchronization
#
# 5. ADAPTIVE QUALITY:
#    - Adjust stream bitrate based on detection confidence
#    - Increase zoom if person is distant (small bbox)
#    - Switch to lower quality if bandwidth constrained
#
# Example integration in tracker.py:
#
#     # In run_loop(), after selecting best detection:
#     if best and not ptz_stream.is_running():
#         name, det, width, resp, rank_value = best
#         
#         # Command PTZ to detection
#         pan, tilt = pixel_to_ptz_angles(det.bbox_xywh, name, ...)
#         set_ptz(ptz_client, "mech", pan, tilt, zoom=2.0)
#         
#         # Start stream with metadata
#         ptz_stream.start_on_detection(det, name)
#     
#     # Stop stream if detection lost
#     elif not best and ptz_stream.is_running():
#         ptz_stream.stop()
#
# Example facial recognition pipeline:
#
#     async def facial_recognition_worker(ptz_stream):
#         while True:
#             frame = ptz_stream.get_frame(timeout=1.0)
#             if frame is None:
#                 continue
#             
#             # Run facial detection
#             faces = face_detector.detect(frame)
#             
#             # Run facial recognition
#             for face in faces:
#                 identity = face_recognizer.identify(face)
#                 emotion = emotion_detector.predict(face)
#                 
#                 logger.info(f"Face detected: {identity}, emotion: {emotion}")
