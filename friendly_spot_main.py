# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Main entry point for integrated Friendly Spot pipeline combining perception,
# behavior planning, and robot command execution with PTZ camera streaming.
# Acknowledgements: Boston Dynamics Spot SDK for robot connection patterns,
# Claude for pipeline integration and optimization

"""Friendly Spot Main Pipeline

Integrated perception to decision to execution loop:

1. Connect to Spot robot and authenticate
2. Initialize PTZ camera stream (ImageClient or WebRTC)
3. Run perception pipeline (pose, face, emotion, gesture)
4. Behavior planning via ComfortModel
5. Execute robot commands based on decisions

Usage:
    # Robot mode with PTZ ImageClient (RECOMMENDED)
    python friendly_spot_main.py --robot ROBOT_IP --user USER --password PASS
    
    # Webcam mode for development (no robot required)
    python friendly_spot_main.py --webcam
    
    # Robot mode with WebRTC streaming (TODO)
    python friendly_spot_main.py --robot ROBOT_IP --user USER --password PASS --webrtc

Options:
    --robot HOSTNAME        Robot hostname or IP address
    --user USERNAME         Robot username (or use environment variable)
    --password PASSWORD     Robot password (or use environment variable)
    --webcam               Use local webcam instead of robot (development mode)
    --webrtc               Use WebRTC instead of ImageClient (TODO: not implemented)
    --ptz-source NAME      PTZ camera source name (default: "ptz")
    --rate HZ              Perception loop rate in Hz (default: 5.0)
    --no-execute           Disable robot command execution (perception only)
    --enable-observer      Enable people_observer for person detection/tracking
    --once                 Run one perception cycle and exit (testing)
    --visualize            Show live perception visualizations
    --save-images DIR      Save annotated frames to directory
    --verbose              Enable verbose debug logging
"""

import sys
import time
import argparse
import logging
import signal

from src.robot import create_robot, RobotActionMonitor, ObserverBridge, ObserverConfig
from src.video import create_video_source
from src.perception import PerceptionPipeline
from src.behavior import ComfortModel, BehaviorExecutor
from src.visualization import visualize_pipeline_frame, close_all_windows

# Configure logging (will be updated if --verbose is set)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FriendlySpotPipeline:
    """Main pipeline orchestrator for Friendly Spot.
    
    Supports two modes:
    1. Integrated (default): Person detection in main loop (simpler, good for 2-5 Hz)
    2. Threaded observer (--enable-observer): Background thread for fast tracking (advanced)
    """
    
    def __init__(self, args):
        self.args = args
        self.robot = None
        self.video_source = None
        self.pipeline = None
        self.comfort_model = None
        self.executor = None
        self.action_monitor = None  # Robot action state monitor
        self.observer_bridge = None  # Optional threaded observer
        self.integrated_detector = None  # Integrated YOLO detector
        self.running = False
        
        # Setup signal handlers for graceful shutdown (cross-platform)
        signal.signal(signal.SIGINT, self._signal_handler)  # Ctrl+C works everywhere
        if sys.platform != 'win32':
            # SIGTERM only available on Unix-like systems
            signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and SIGTERM gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def initialize(self):
        """Initialize all pipeline components."""
        logger.info("Initializing Friendly Spot pipeline...")
        logger.debug(f"Arguments: webcam={self.args.webcam}, robot={self.args.robot}, rate={self.args.rate}Hz, no_execute={self.args.no_execute}")
        
        # Connect to robot if specified
        if not self.args.webcam:
            logger.info(f"Connecting to robot at {self.args.robot}...")
            logger.debug(f"Client name: FriendlySpot, register_spot_cam=True")
            self.robot = create_robot(
                hostname=self.args.robot,
                client_name='FriendlySpot',
                register_spot_cam=True,  # Enable PTZ/compositor clients
                verbose=self.args.bosdyn_verbose  # Use separate bosdyn verbose flag
            )
            logger.info("Robot connection established")
            logger.debug(f"Robot ID: {self.robot.id if hasattr(self.robot, 'id') else 'unknown'}")
        else:
            logger.debug("Webcam mode: skipping robot connection")
        
        # Create video source
        logger.debug("Creating video source...")
        self.video_source = self._create_video_source()
        logger.debug(f"Video source created: {type(self.video_source).__name__}")
        
        # Initialize perception pipeline
        logger.info("Initializing perception pipeline...")
        self.pipeline = PerceptionPipeline(self.video_source, robot=self.robot)
        logger.debug("Perception pipeline initialized (pose, face, emotion, gesture)")
        
        # Initialize behavior planner
        logger.info("Initializing comfort model...")
        self.comfort_model = ComfortModel()
        logger.debug("Comfort model initialized")
        
        # Initialize robot action monitor (if robot connected)
        if self.robot:
            logger.info("Initializing robot action monitor...")
            self.action_monitor = RobotActionMonitor(self.robot)
            logger.debug("Robot action monitor initialized")
        
        # Initialize behavior executor (only if robot connected and execution enabled)
        if self.robot and not self.args.no_execute:
            logger.info("Initializing behavior executor...")
            self.executor = BehaviorExecutor(
                self.robot,
                force_take_lease=self.args.force_take_lease
            )
            logger.debug("Behavior executor initialized")
            # Note: Lease/E-Stop managed automatically via context managers in execute_behavior()
        else:
            logger.debug(f"Skipping behavior executor: robot={self.robot is not None}, no_execute={self.args.no_execute}")
        
        # Initialize person detection (threaded observer OR integrated detector)
        if self.robot:
            if self.args.enable_observer:
                # THREADED MODE: Background observer for fast tracking
                logger.info("Initializing THREADED observer bridge (advanced mode)...")
                from bosdyn.client.image import ImageClient
                from bosdyn.client.spot_cam.ptz import PtzClient
                
                image_client = self.robot.ensure_client(ImageClient.default_service_name)
                ptz_client = self.robot.ensure_client(PtzClient.default_service_name)
                
                obs_config = ObserverConfig(
                    enable_observer=True,
                    surround_fps=2.0,
                    ptz_fps=5.0,
                    detection_timeout_sec=2.0
                )
                
                self.observer_bridge = ObserverBridge(self.robot, image_client, ptz_client, obs_config)
                self.observer_bridge.start()
                
                # Wait for observer bridge to finish initialization (loading YOLO model, etc.)
                logger.info("Waiting for observer bridge initialization (loading YOLO model)...")
                ready = self.observer_bridge.wait_until_ready(timeout=15.0)
                if not ready:
                    logger.error("Observer bridge failed to initialize within timeout!")
                else:
                    logger.info("Observer bridge ready (threaded tracking active)")
            else:
                # INTEGRATED MODE: Detection in main loop (default, simpler)
                logger.info("Initializing INTEGRATED person detector (default mode)...")
                self._initialize_integrated_detector()
                logger.info("Integrated detector ready")
        else:
            logger.debug(f"No robot connection, skipping person detection")
        
        logger.info("Initialization complete")
    
    def _initialize_integrated_detector(self):
        """Initialize components for integrated person detection (no threading)."""
        from bosdyn.client.image import ImageClient
        from bosdyn.client.spot_cam.ptz import PtzClient
        from people_observer.detection import YoloDetector
        
        # Create SDK clients for image access
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.ptz_client = self.robot.ensure_client(PtzClient.default_service_name)
        
        # Create separate ImageClient for spot-cam-image service (PTZ camera)
        # Note: ensure_client signature is ensure_client(service_name) when client type is known
        from people_observer.config import SPOT_CAM_IMAGE_SERVICE
        self.spot_cam_image_client = self.robot.ensure_client(SPOT_CAM_IMAGE_SERVICE)
        
        # Initialize YOLO detector (loads model in main thread)
        logger.info("Loading YOLO model for integrated detection...")
        from people_observer.config import RuntimeConfig
        cfg = RuntimeConfig()
        self.integrated_detector = YoloDetector(
            model_path=cfg.yolo.model_path,
            imgsz=cfg.yolo.img_size,
            conf=cfg.yolo.min_confidence,
            iou=cfg.yolo.iou_threshold,
            device=cfg.yolo.device
        )
        logger.info(f"YOLO model loaded: {cfg.yolo.model_path}")
        
        # Store config and image sources for detection
        self.detection_config = cfg
        from people_observer.cameras import fetch_image_sources
        self.image_sources = fetch_image_sources(self.image_client)
        
        # Counter for detection frame saves
        self.detection_frame_count = 0
    
    def _wait_for_ptz_stable(self, target_pan: float, target_tilt: float, ptz_name: str, timeout: float = 2.0):
        """Wait for PTZ to reach target position or timeout.
        
        Polls PTZ position until it's within tolerance or timeout expires.
        
        Args:
            target_pan: Target pan angle in degrees
            target_tilt: Target tilt angle in degrees
            ptz_name: PTZ device name ('mech' or 'digi')
            timeout: Maximum time to wait in seconds
        """
        from bosdyn.api.spot_cam import ptz_pb2
        import time
        
        tolerance_deg = 5.0  # Acceptable error in degrees
        poll_interval = 0.05  # Poll every 50ms
        start_time = time.time()
        
        ptz_desc = ptz_pb2.PtzDescription(name=ptz_name)
        
        while (time.time() - start_time) < timeout:
            try:
                current_pos = self.ptz_client.get_ptz_position(ptz_desc)
                pan_error = abs(current_pos.pan.value - target_pan)
                tilt_error = abs(current_pos.tilt.value - target_tilt)
                
                logger.debug(f"PTZ position: pan={current_pos.pan.value:.1f}° (target={target_pan:.1f}°), tilt={current_pos.tilt.value:.1f}° (target={target_tilt:.1f}°)")
                
                if pan_error < tolerance_deg and tilt_error < tolerance_deg:
                    elapsed = time.time() - start_time
                    logger.info(f"PTZ stabilized in {elapsed:.2f}s (pan_error={pan_error:.1f}°, tilt_error={tilt_error:.1f}°)")
                    return True
                    
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.warning(f"Failed to query PTZ position: {e}")
                # Fall back to fixed wait
                break
        
        # Timeout or error - use remaining time as fixed wait
        remaining = timeout - (time.time() - start_time)
        if remaining > 0:
            logger.warning(f"PTZ stabilization timeout, waiting {remaining:.2f}s more")
            time.sleep(remaining)
        
        return False
    
    def _save_detection_frames(self, frames, all_detections, camera_names):
        """Save surround camera frames with YOLO detection annotations."""
        import cv2
        import os
        from pathlib import Path
        from datetime import datetime
        
        save_dir = Path(self.args.save_images) / "detections"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this batch of frames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        for camera_name, frame, detections in zip(camera_names, frames.values(), all_detections):
            # Create annotated copy
            annotated = frame.copy()
            
            # Draw detections
            for det in detections:
                x, y, w, h = det.bbox_xywh
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence
                label = f"Person {det.conf:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw mask if available
                if det.mask is not None:
                    mask_overlay = cv2.resize(det.mask.astype('uint8') * 255, 
                                             (frame.shape[1], frame.shape[0]))
                    mask_colored = cv2.applyColorMap(mask_overlay, cv2.COLORMAP_JET)
                    annotated = cv2.addWeighted(annotated, 0.7, mask_colored, 0.3, 0)
            
            # Save annotated frame with timestamp-based naming
            filename = save_dir / f"{timestamp}_{camera_name}.jpg"
            cv2.imwrite(str(filename), annotated)
            
        logger.debug(f"Saved detection frames to {save_dir}")
        self.detection_frame_count += 1
    
    def _integrated_detect_person(self):
        """
        Detect person using integrated mode (no threading).
        Flow: Fetch surround → YOLO detect → Select nearest → Point PTZ → Fetch PTZ frame
        Returns: PersonDetection or None
        """
        from people_observer.cameras import get_frames
        from people_observer.tracker import estimate_detection_depth_m
        from people_observer.ptz_control import set_ptz
        from people_observer import geometry,cameras
        from src.perception.detection_types import PersonDetection
        import time
        import cv2
        import os
        from pathlib import Path
        
        cfg = self.detection_config
        
        try:
            # 1. Fetch surround camera frames + depth
            frames, responses, depth_frames = get_frames(
                self.image_client, cfg.sources, include_depth=True
            )
            
            if not frames:
                logger.debug("No surround frames available")
                return None
            
            # 2. Run YOLO detection on all cameras
            names = list(frames.keys())
            imgs = list(frames.values())
            all_dets = self.integrated_detector.predict_batch(imgs)
            
            # Save surround camera frames with detections if requested
            if self.args.save_images:
                self._save_detection_frames(frames, all_dets, names)
            
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
                logger.debug("No person detected in surround cameras")
                return None
            
            logger.debug(f"Closest person: {best_distance:.2f}m in {best_camera}")
            
            # 4. Command PTZ to point at person
            intrinsics = cameras.get_camera_intrinsics(best_camera, self.image_sources)
            cam_yaw_deg = geometry.get_camera_yaw_fallback(best_camera)
            
            # Get target point (upper third of bbox for head/torso)
            x, y, w, h = best_detection.bbox_xywh
            cx = x + w / 2.0
            cy = y + h * 0.3
            img_w = frames[best_camera].shape[1]
            img_h = frames[best_camera].shape[0]
            
            logger.debug(f"Target pixel: ({cx:.1f}, {cy:.1f}) in {img_w}x{img_h} image")
            
            # Get image response for the best camera (needed for frame transforms)
            # responses is a list, need to find matching source
            best_camera_response = None
            for resp in responses:
                if resp.source.name == best_camera:
                    best_camera_response = resp
                    break
            
            # Compute PTZ angles (with fallback pattern from tracker.py)
            try:
                if cfg.observer_mode == "transform" and intrinsics and best_camera_response:
                    logger.debug(f"Using TRANSFORM method (observer_mode={cfg.observer_mode}, intrinsics available)")
                    ptz_pan, ptz_tilt = geometry.pixel_to_ptz_angles_transform(
                        cx, cy, intrinsics, best_camera_response, self.robot, img_w, img_h
                    )
                    logger.debug(f"Transform result: pan={ptz_pan:.2f}°, tilt={ptz_tilt:.2f}°")
                else:
                    # Fallback to simple projection
                    if cfg.observer_mode == "transform":
                        if not intrinsics:
                            logger.warning(f"FALLBACK TO SIMPLE PROJECTION: Transform mode requested but no intrinsics available for {best_camera}")
                        elif not best_camera_response:
                            logger.warning(f"FALLBACK TO SIMPLE PROJECTION: Transform mode requested but no image response for {best_camera}")
                        else:
                            logger.warning(f"FALLBACK TO SIMPLE PROJECTION: Transform mode requested but intrinsics invalid for {best_camera}")
                    else:
                        logger.debug(f"Using simple projection mode (bearing mode explicitly selected)")
                    
                    # Calculate HFOV from intrinsics if available, otherwise use fallback
                    if intrinsics:
                        hfov = cameras.calculate_hfov_from_intrinsics(intrinsics)
                    else:
                        hfov = 133.0  # Fisheye fallback
                        logger.warning(f"No intrinsics for {best_camera}, using fallback HFOV={hfov}°")
                    
                    ptz_pan, ptz_tilt = geometry.pixel_to_ptz_angles_simple(
                        cx, cy, img_w, img_h, hfov, cam_yaw_deg, cfg.ptz.default_tilt_deg
                    )
                    logger.debug(f"Bearing result: pan={ptz_pan:.2f}°, tilt={ptz_tilt:.2f}°")
            except Exception as e:
                logger.error(f"Failed to compute PTZ angles: {e}, using simple projection fallback")
                hfov = 133.0
                ptz_pan, ptz_tilt = geometry.pixel_to_ptz_angles_simple(
                    cx, cy, img_w, img_h, hfov, cam_yaw_deg, cfg.ptz.default_tilt_deg
                )
            
            # Send PTZ command
            logger.info(f"Sending PTZ command: pan={ptz_pan:.1f}°, tilt={ptz_tilt:.1f}° (device={cfg.ptz.name})")
            set_ptz(self.ptz_client, cfg.ptz.name, ptz_pan, ptz_tilt)
            logger.info(f"PTZ command completed successfully")
            
            # 5. Wait for PTZ to stabilize and verify position
            self._wait_for_ptz_stable(ptz_pan, ptz_tilt, cfg.ptz.name, timeout=2.0)
            
            # 6. Fetch PTZ frame (PTZ camera has no depth sensor, so skip depth request)
            ptz_source = cfg.ptz.source_name
            logger.info(f"Fetching PTZ frame from source: '{ptz_source}' via spot-cam-image service")
            
            # Use spot-cam ImageClient for PTZ (different service than surround cameras)
            # PTZ camera has no depth sensor - always use include_depth=False
            try:
                ptz_frames, ptz_responses, ptz_depths = get_frames(
                    self.spot_cam_image_client, [ptz_source], include_depth=False
                )
            except Exception as e:
                logger.error(f"Failed to fetch PTZ frame: {e}")
                logger.error(f"Available sources in spot-cam-image service may not include '{ptz_source}'")
                logger.error(f"Check with test_list_image_sources.py or verify PTZ_SOURCE_NAME in config")
                return None
            
            if not ptz_frames or ptz_source not in ptz_frames:
                logger.warning(f"PTZ source '{ptz_source}' not in returned frames. Available: {list(ptz_frames.keys())}")
                return None
            
            ptz_frame = ptz_frames[ptz_source]
            ptz_depth = ptz_depths.get(ptz_source)
            
            # 7. Create PersonDetection with all info (store PTZ angles for behavior execution)
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
                tracking_quality=0.8  # Fixed quality for integrated mode
            )
            
            # Store PTZ bearing for behavior execution (attach as custom attribute)
            person_det.ptz_pan = ptz_pan
            person_det.ptz_tilt = ptz_tilt
            logger.debug(f"Stored PTZ bearing: pan={ptz_pan:.1f}°, tilt={ptz_tilt:.1f}°")
            
            return person_det
            
        except Exception as e:
            logger.error(f"Error in integrated detection: {e}", exc_info=True)
            return None
    
    def _create_video_source(self):
        """Create video source based on arguments."""
        if self.args.webcam:
            logger.info("Using webcam video source")
            logger.debug("Webcam device: 0")
            return create_video_source('webcam', device=0)
        
        elif self.args.webrtc:
            logger.info("Using WebRTC video source")
            logger.debug("WebRTC not yet implemented, creating placeholder")
            # TODO: WebRTC not yet implemented
            return create_video_source('webrtc', robot=self.robot)
        
        else:
            logger.info(f"Using ImageClient video source (camera: {self.args.ptz_source})")
            # Import depth setting from config (PTZ likely doesn't have depth, but try anyway)
            try:
                from people_observer.config import DEFAULT_INCLUDE_DEPTH
                include_depth = DEFAULT_INCLUDE_DEPTH
                logger.debug(f"Loaded DEFAULT_INCLUDE_DEPTH from config: {include_depth}")
            except ImportError:
                include_depth = False
                logger.debug("Could not import DEFAULT_INCLUDE_DEPTH, using False")
            
            logger.info(f"Depth fetching: {'enabled' if include_depth else 'disabled'}" + 
                       " (note: PTZ camera may not have depth sensor)")
            logger.debug(f"ImageClient params: source={self.args.ptz_source}, quality=75, include_depth={include_depth}")
            
            try:
                return create_video_source(
                    'imageclient',
                    robot=self.robot,
                    source_name=self.args.ptz_source,
                    quality=75,
                    include_depth=include_depth
                )
            except Exception as e:
                logger.error(f"Failed to initialize video source: {e}")
                logger.error("If your robot doesn't have a PTZ camera, the system will use a fallback camera")
                logger.info("The video source initialization will attempt automatic fallback...")
                raise
    
    def run(self):
        """Main perception to decision to execution loop."""
        self.running = True
        logger.info(f"Starting pipeline at {self.args.rate} Hz...")
        if self.args.once:
            logger.info("ONCE MODE: Will run single cycle and exit")
        else:
            logger.info("Press Ctrl+C to stop")
        logger.debug(f"Loop period: {1.0 / self.args.rate:.3f}s")
        
        loop_count = 0
        loop_period = 1.0 / self.args.rate
        
        # Use BehaviorExecutor as context manager to hold lease/estop for entire session
        executor_context = self.executor if self.executor else None
        
        try:
            # Acquire and hold robot control for entire session
            if executor_context:
                executor_context.__enter__()
            
            while self.running:
                loop_start = time.time()
                
                # Get person detection (threaded observer OR integrated detection)
                person_detection = None
                if self.observer_bridge and self.observer_bridge.has_person():
                    # THREADED MODE: Get detection from observer queue
                    person_detection = self.observer_bridge.get_person_detection()
                    logger.debug(f"[Loop {loop_count}] Using threaded observer detection")
                elif self.integrated_detector:
                    # INTEGRATED MODE: Detect person in main loop
                    person_detection = self._integrated_detect_person()
                    if person_detection:
                        logger.debug(f"[Loop {loop_count}] Using integrated detection (distance={person_detection.distance_m:.2f}m)")
                    else:
                        logger.debug(f"[Loop {loop_count}] No person detected by integrated detector")
                
                # 1. PERCEPTION: Read frame and extract perception data
                logger.debug(f"[Loop {loop_count}] Reading perception...")
                perception = self.pipeline.read_perception(person_detection)
                if perception is None:
                    logger.warning("Failed to read perception (no frame)")
                    time.sleep(0.1)
                    continue
                logger.debug(f"[Loop {loop_count}] Perception complete: distance={perception.distance_m:.2f}m, action={perception.current_action}" if perception.distance_m else f"[Loop {loop_count}] Perception complete: distance=N/A, action={perception.current_action}")
                
                # 2. DECISION: Compute comfort and select behavior
                logger.debug(f"[Loop {loop_count}] Computing comfort and behavior...")
                comfort, behavior = self.comfort_model.predict_behavior(perception)
                logger.debug(f"[Loop {loop_count}] Decision: comfort={comfort:.2f}, behavior={behavior.value}")
                
                # Visualize if enabled (after decision so we can show desired behavior)
                if self.args.visualize or self.args.save_images:
                    # Use the analyzed frame from perception to ensure alignment
                    frame = perception.frame if hasattr(perception, 'frame') and perception.frame is not None else None
                    if frame is not None:
                        key = visualize_pipeline_frame(
                            frame,
                            perception_data=perception,
                            person_detection=person_detection,
                            desired_behavior=behavior.value,  # Show the decided behavior
                            comfort_score=comfort,  # Show the comfort score
                            show=self.args.visualize,
                            save_dir=self.args.save_images,
                            iteration=loop_count
                        )
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            logger.info("User requested quit via visualization")
                            break
                
                # Log every 10 iterations to avoid spam
                if loop_count % 10 == 0:
                    logger.info(
                        f"[Loop {loop_count}] "
                        f"Comfort: {comfort:.2f} | "
                        f"Behavior: {behavior.value} | "
                        f"Pose: {perception.pose_label} | "
                        f"Emotion: {perception.emotion_label} | "
                        f"Face: {perception.face_label} | "
                        f"Gesture: {perception.gesture_label} | "
                        f"Action: {perception.current_action} | "
                        f"Distance: {perception.distance_m:.2f}m" if perception.distance_m else f"Distance: N/A"
                    )
                
                # 3. EXECUTION: Send command to robot (if enabled)
                if self.executor:
                    logger.debug(f"[Loop {loop_count}] Executing behavior: {behavior.value}")
                    self.executor.execute_behavior(behavior, perception=perception)
                else:
                    logger.debug(f"[Loop {loop_count}] No executor, skipping behavior execution")
                
                # Check once mode
                loop_count += 1
                if self.args.once:
                    logger.info(f"ONCE MODE: Completed iteration {loop_count}, exiting")
                    break
                
                # Rate limiting
                elapsed = time.time() - loop_start
                sleep_time = max(0, loop_period - elapsed)
                logger.debug(f"[Loop {loop_count}] Loop elapsed: {elapsed:.3f}s, sleep: {sleep_time:.3f}s")
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
        
        finally:
            # Release robot control
            if executor_context:
                try:
                    executor_context.__exit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error releasing robot control: {e}")
            
            self.cleanup()
    
    def cleanup(self):
        """Cleanup all resources."""
        logger.info("Cleaning up resources...")
        
        # Stop observer bridge
        if self.observer_bridge:
            logger.info("Stopping observer bridge...")
            self.observer_bridge.stop()
        
        # Close visualization windows
        if self.args.visualize:
            logger.info("Closing visualization windows...")
            close_all_windows()
        
        # Note: BehaviorExecutor uses context managers (ManagedLease, ManagedEstop)
        # which automatically handle cleanup, so no shutdown() method needed
        
        # Cleanup perception pipeline (closes MediaPipe, OpenCV)
        if self.pipeline:
            self.pipeline.cleanup()
        
        logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Friendly Spot: Integrated perception and behavior pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Robot connection
    robot_group = parser.add_mutually_exclusive_group(required=True)
    robot_group.add_argument(
        '--robot',
        help='Robot hostname or IP address'
    )
    robot_group.add_argument(
        '--webcam',
        action='store_true',
        help='Use local webcam instead of robot (development mode)'
    )
    
    parser.add_argument(
        '--user',
        help='Robot username (or set BOSDYN_CLIENT_USERNAME env var)'
    )
    parser.add_argument(
        '--password',
        help='Robot password (or set BOSDYN_CLIENT_PASSWORD env var)'
    )
    
    # Video source options
    parser.add_argument(
        '--webrtc',
        action='store_true',
        help='Use WebRTC streaming instead of ImageClient (TODO: not implemented)'
    )
    parser.add_argument(
        '--ptz-source',
        default='ptz',
        help='PTZ camera source name (default: ptz)'
    )
    
    # Pipeline options
    parser.add_argument(
        '--rate',
        type=float,
        default=5.0,
        help='Perception loop rate in Hz (default: 5.0)'
    )
    parser.add_argument(
        '--no-execute',
        action='store_true',
        help='Disable robot command execution (perception only)'
    )
    parser.add_argument(
        '--force-take-lease',
        action='store_true',
        help='Forcefully take lease from tablet or other clients (use if lease conflicts occur)'
    )
    parser.add_argument(
        '--enable-observer',
        action='store_true',
        help='Enable THREADED observer for continuous fast person tracking (advanced mode, default: integrated detection in main loop)'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run one perception cycle and exit (testing mode)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show live perception visualizations with OpenCV'
    )
    parser.add_argument(
        '--save-images',
        type=str,
        metavar='DIR',
        help='Save annotated frames to specified directory'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging for friendly_spot modules'
    )
    parser.add_argument(
        '--bosdyn-verbose',
        action='store_true',
        help='Enable verbose logging for Boston Dynamics SDK (very detailed)'
    )
    
    args = parser.parse_args()
    
    # Update logging level if verbose
    if args.verbose:
        # Set DEBUG for our modules only (not root logger)
        for module_name in ['__main__', 'robot_io', 'video_sources', 'run_pipeline', 
                            'behavior_planner', 'behavior_executor', 'observer_bridge',
                            'unified_visualization', 'people_observer']:
            logging.getLogger(module_name).setLevel(logging.DEBUG)
        # Ensure handlers can pass DEBUG messages
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("Verbose mode enabled for friendly_spot modules")
    
    # Bosdyn verbose is separate and affects only bosdyn.* loggers
    if args.bosdyn_verbose:
        logging.getLogger('bosdyn').setLevel(logging.DEBUG)
        logger.info("Boston Dynamics SDK verbose mode enabled")
    
    # Validate arguments
    if not args.webcam and not args.robot:
        parser.error("Must specify either --robot or --webcam")
    
    if args.webrtc and args.webcam:
        parser.error("Cannot use --webrtc with --webcam")
    
    # Create and run pipeline
    pipeline = FriendlySpotPipeline(args)
    pipeline.initialize()
    pipeline.run()


if __name__ == '__main__':
    main()
