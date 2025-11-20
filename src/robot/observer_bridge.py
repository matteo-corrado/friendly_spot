# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Integration bridge between people_observer tracking and friendly_spot perception modules
# coordinating surround camera detection with PTZ-based emotion and gesture analysis
# Acknowledgements: Claude for bridge architecture design

"""Integration helper for people_observer tracking with friendly_spot perception.

This module provides a bridge between:
- people_observer: Person detection/tracking in surround cameras, PTZ control
- friendly_spot: Perception pipeline (pose, face, emotion, gesture analysis)

The integration flow:
1. people_observer monitors surround cameras for people
2. When person detected, commands PTZ to point at them
3. Passes PTZ frames + depth + detection to friendly_spot perception
4. Perception pipeline analyzes person and returns behavior decision
"""
import logging
import threading
import queue
import time
from typing import Optional, Callable
from dataclasses import dataclass

from ..perception.detection_types import PersonDetection

logger = logging.getLogger(__name__)


@dataclass
class ObserverConfig:
    """Configuration for people_observer integration."""
    enable_observer: bool = True  # Use people_observer for detection/tracking
    surround_fps: float = 2.0  # Frame rate for surround camera monitoring
    ptz_fps: float = 5.0  # Frame rate for PTZ tracking when person detected
    detection_timeout_sec: float = 2.0  # How long to wait without detection before stopping PTZ
    min_tracking_quality: float = 0.5  # Minimum quality to consider PTZ tracking stable


class ObserverBridge:
    """Bridge between people_observer tracking and friendly_spot perception.
    
    Runs people_observer in background thread, monitors for person detections,
    and provides PersonDetection objects to perception pipeline when person is
    in PTZ view.
    
    Usage:
        bridge = ObserverBridge(robot, image_client, ptz_client, config)
        bridge.start()
        
        while True:
            if bridge.has_person():
                person_det = bridge.get_person_detection()
                perception = pipeline.read_perception(person_det)
                # ... use perception for behavior decision
            else:
                # No person detected, wait or do other tasks
                time.sleep(0.1)
        
        bridge.stop()
    """
    
    def __init__(self, robot, image_client, ptz_client, config: ObserverConfig):
        """Initialize observer bridge.
        
        Args:
            robot: Authenticated Robot instance
            image_client: ImageClient for fetching frames
            ptz_client: PtzClient for PTZ control
            config: ObserverConfig with integration settings
        """
        self.robot = robot
        self.image_client = image_client
        self.ptz_client = ptz_client
        self.config = config
        
        # Threading
        self.thread = None
        self.running = False
        self.ready = threading.Event()  # Signals when initialization complete
        self.detection_queue = queue.Queue(maxsize=1)  # Only keep latest detection
        
        # State
        self.current_detection: Optional[PersonDetection] = None
        self.last_detection_time = 0.0
        
    def start(self):
        """Start background observer thread."""
        if self.running:
            logger.warning("Observer bridge already running")
            return
        
        logger.debug(f"Starting observer bridge: surround_fps={self.config.surround_fps}, ptz_fps={self.config.ptz_fps}")
        self.running = True
        self.ready.clear()  # Reset ready flag
        self.thread = threading.Thread(target=self._observer_loop, daemon=True)
        self.thread.start()
        logger.info("Observer bridge started (initializing in background...)")
    
    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        """Wait for observer bridge to finish initialization.
        
        Args:
            timeout: Maximum seconds to wait (default: 10.0)
        
        Returns:
            True if ready within timeout, False if timeout
        """
        logger.debug(f"Waiting for observer bridge to be ready (timeout={timeout}s)...")
        ready = self.ready.wait(timeout=timeout)
        if ready:
            logger.info("Observer bridge ready")
        else:
            logger.warning(f"Observer bridge not ready after {timeout}s timeout")
        return ready
    
    def stop(self):
        """Stop background observer thread."""
        if not self.running:
            logger.debug("Observer bridge not running")
            return
        
        logger.debug("Stopping observer bridge...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Observer thread did not terminate cleanly")
        logger.info("Observer bridge stopped")
    
    def has_person(self) -> bool:
        """Check if person is currently detected and tracked by PTZ."""
        if self.current_detection is None:
            logger.debug("has_person: No current detection")
            return False
        
        # Check if detection is stale
        if self.current_detection.is_stale(timeout_sec=self.config.detection_timeout_sec):
            age = self.current_detection.age_seconds()
            logger.debug(f"has_person: Detection stale (age={age:.2f}s > timeout={self.config.detection_timeout_sec}s)")
            self.current_detection = None
            return False
        
        # Check if PTZ is tracking with sufficient quality
        if self.current_detection.tracked_by_ptz and \
           self.current_detection.tracking_quality >= self.config.min_tracking_quality:
            logger.debug(f"has_person: True (quality={self.current_detection.tracking_quality:.2f})")
            return True
        
        logger.debug(f"has_person: False (tracked_by_ptz={self.current_detection.tracked_by_ptz}, quality={self.current_detection.tracking_quality:.2f})")
        return False
    
    def get_person_detection(self) -> Optional[PersonDetection]:
        """Get current PersonDetection with frame, depth, and tracking info.
        
        Returns:
            PersonDetection if person is in PTZ view, None otherwise
        """
        # Try to get latest detection from queue (non-blocking)
        try:
            self.current_detection = self.detection_queue.get_nowait()
            self.last_detection_time = time.time()
            logger.debug(f"get_person_detection: New detection from queue (distance={self.current_detection.distance_m:.2f}m, depth_source={self.current_detection.depth_source})")
        except queue.Empty:
            logger.debug("get_person_detection: No new detection in queue")
            pass
        
        if self.has_person():
            logger.debug(f"get_person_detection: Returning detection (bbox={self.current_detection.bbox_xywh}, has_mask={self.current_detection.has_mask()})")
            return self.current_detection
        logger.debug("get_person_detection: No person available")
        return None
    
    def _observer_loop(self):
        """Background thread that runs people_observer tracking loop.
        
        Full implementation:
        1. Fetch frames from surround cameras
        2. Run YOLO detection
        3. Select target person (nearest with depth)
        4. Command PTZ to point at person
        5. Fetch PTZ frame + depth
        6. Create PersonDetection and put in queue
        """
        logger.info("Observer loop starting (full implementation)")
        
        try:
            # Import people_observer modules
            from people_observer.detection import YoloDetector, Detection
            from people_observer.cameras import get_frames, fetch_image_sources
            from people_observer.config import RuntimeConfig
            from people_observer.tracker import estimate_detection_depth_m
            from people_observer import geometry, ptz_control
            import numpy as np
            
            # Get camera intrinsics
            image_sources = fetch_image_sources(self.image_client)
            
            # Create runtime config
            cfg = RuntimeConfig()
            cfg.dry_run = False  # Enable PTZ control
            
            # Initialize YOLO detector
            detector = YoloDetector(
                model_path=cfg.yolo.model_path,
                imgsz=cfg.yolo.img_size,
                conf=cfg.yolo.min_confidence,
                iou=cfg.yolo.iou_threshold,
                device=cfg.yolo.device
            )
            logger.info(f"YOLO detector initialized: {cfg.yolo.model_path}")
            
            # Signal that initialization is complete
            self.ready.set()
            logger.info("Observer bridge initialization complete, entering detection loop")
            
            surround_period = 1.0 / self.config.surround_fps
            ptz_period = 1.0 / self.config.ptz_fps
            ptz_tracking_person = False
            last_ptz_time = 0
            
            while self.running:
                try:
                    cycle_start = time.time()
                    
                    # 1. Fetch surround camera frames + depth
                    frames, responses, depth_frames = get_frames(
                        self.image_client, cfg.sources, include_depth=True
                    )
                    
                    if not frames:
                        time.sleep(surround_period)
                        continue
                    
                    # 2. Run YOLO detection on all cameras
                    names = list(frames.keys())
                    imgs = list(frames.values())
                    all_dets = detector.predict_batch(imgs)
                    
                    # 3. Select target person (nearest with depth)
                    best_detection = None
                    best_distance = float('inf')
                    best_camera = None
                    
                    for name, dets, img in zip(names, all_dets, imgs):
                        if not dets:
                            continue
                        
                        # Attach source name
                        for d in dets:
                            d.source = name
                        
                        # Find closest person with depth
                        depth_img = depth_frames.get(name)
                        for det in dets:
                            dist = estimate_detection_depth_m(depth_img, det, use_mask=True)
                            if dist is not None and dist < best_distance:
                                best_distance = dist
                                best_detection = det
                                best_camera = name
                    
                    if best_detection is None:
                        # No person detected, stop PTZ tracking
                        if ptz_tracking_person:
                            logger.debug("No person detected, stopping PTZ tracking")
                            ptz_tracking_person = False
                        time.sleep(surround_period)
                        continue
                    
                    logger.debug(f"Closest person: {best_distance:.2f}m in {best_camera}")
                    
                    # 4. Command PTZ to point at person
                    intrinsics = geometry.get_camera_intrinsics(best_camera, image_sources)
                    cam_yaw_deg = geometry.get_camera_yaw_fallback(best_camera)
                    
                    # Get target point (upper third of bbox for head/torso)
                    x, y, w, h = best_detection.bbox_xywh
                    cx = x + w / 2.0
                    cy = y + h * 0.3
                    img_w = frames[best_camera].shape[1]
                    img_h = frames[best_camera].shape[0]
                    
                    # Compute PTZ angles
                    if cfg.observer_mode == "transform" and intrinsics:
                        ptz_pan, ptz_tilt = geometry.pixel_to_ptz_transform(
                            cx, cy, img_w, img_h, best_camera, intrinsics, self.robot
                        )
                    else:
                        ptz_pan, ptz_tilt = geometry.pixel_to_ptz_bearing(
                            cx, cy, img_w, img_h, cam_yaw_deg, intrinsics
                        )
                    
                    # Send PTZ command
                    ptz_control.set_ptz(self.ptz_client, cfg.ptz.name, ptz_pan, ptz_tilt)
                    ptz_tracking_person = True
                    logger.debug(f"PTZ commanded: pan={ptz_pan:.1f}°, tilt={ptz_tilt:.1f}°")
                    
                    # 5. Wait for PTZ to stabilize (brief pause)
                    time.sleep(0.3)
                    
                    # 6. Fetch PTZ frame + depth (if available)
                    ptz_source = cfg.ptz.source_name
                    ptz_frames, ptz_responses, ptz_depths = get_frames(
                        self.image_client, [ptz_source], include_depth=True
                    )
                    
                    if not ptz_frames or ptz_source not in ptz_frames:
                        logger.warning("Failed to fetch PTZ frame")
                        time.sleep(surround_period)
                        continue
                    
                    ptz_frame = ptz_frames[ptz_source]
                    ptz_depth = ptz_depths.get(ptz_source)
                    
                    # 7. Create PersonDetection with all info
                    person_det = PersonDetection(
                        bbox_xywh=best_detection.bbox_xywh,
                        mask=best_detection.mask,
                        confidence=best_detection.conf,
                        distance_m=best_distance,
                        depth_source=best_camera,
                        source_camera=ptz_source,
                        frame=ptz_frame,
                        depth_frame=ptz_depth,
                        tracked_by_ptz=True,
                        tracking_quality=0.8,  # Fixed quality for now
                        detection_time=time.time()
                    )
                    
                    # 8. Put detection in queue (replace old if queue full)
                    try:
                        self.detection_queue.get_nowait()  # Clear old
                    except queue.Empty:
                        pass
                    self.detection_queue.put(person_det)
                    logger.debug(f"PersonDetection queued: {best_distance:.2f}m, PTZ frame ready")
                    
                    # Rate limiting
                    elapsed = time.time() - cycle_start
                    sleep_time = max(0, ptz_period - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error in observer loop iteration: {e}", exc_info=True)
                    time.sleep(1.0)
                    
        except ImportError as e:
            logger.error(f"Failed to import people_observer modules: {e}")
            logger.error("Observer loop cannot run - people_observer not available")
        
        except Exception as e:
            logger.error(f"Fatal error in observer loop: {e}", exc_info=True)
        
        finally:
            logger.info("Observer loop exiting")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Convenience function for simple integration
def create_observer_bridge(robot, image_client, ptz_client, 
                           enable: bool = True) -> Optional[ObserverBridge]:
    """Create and start observer bridge if enabled.
    
    Args:
        robot: Authenticated Robot instance
        image_client: ImageClient for fetching frames
        ptz_client: PtzClient for PTZ control  
        enable: Whether to enable observer (False = perception-only mode)
    
    Returns:
        ObserverBridge if enabled and successfully created, None otherwise
    """
    if not enable:
        logger.info("Observer bridge disabled - running in perception-only mode")
        return None
    
    try:
        config = ObserverConfig(enable_observer=True)
        bridge = ObserverBridge(robot, image_client, ptz_client, config)
        bridge.start()
        logger.info("Observer bridge created and started")
        return bridge
    except Exception as e:
        logger.error(f"Failed to create observer bridge: {e}")
        logger.info("Falling back to perception-only mode")
        return None
