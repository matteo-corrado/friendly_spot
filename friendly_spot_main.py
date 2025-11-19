# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/18/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Main entry point for integrated Friendly Spot pipeline combining perception,
# behavior planning, and robot command execution with PTZ camera streaming.
# Acknowledgements: Boston Dynamics Spot SDK for robot connection patterns

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

from robot_io import create_robot
from video_sources import create_video_source
from run_pipeline import PerceptionPipeline
from behavior_planner import ComfortModel
from behavior_executor import BehaviorExecutor
from observer_bridge import ObserverBridge, ObserverConfig
from unified_visualization import visualize_pipeline_frame, close_all_windows

# Configure logging (will be updated if --verbose is set)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FriendlySpotPipeline:
    """Main pipeline orchestrator for Friendly Spot."""
    
    def __init__(self, args):
        self.args = args
        self.robot = None
        self.video_source = None
        self.pipeline = None
        self.comfort_model = None
        self.executor = None
        self.observer_bridge = None
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
                verbose=self.args.verbose
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
        
        # Initialize behavior executor (only if robot connected and execution enabled)
        if self.robot and not self.args.no_execute:
            logger.info("Initializing behavior executor...")
            self.executor = BehaviorExecutor(self.robot)
            logger.debug("Behavior executor initialized")
            # Note: Lease/E-Stop managed automatically via context managers in execute_behavior()
        else:
            logger.debug(f"Skipping behavior executor: robot={self.robot is not None}, no_execute={self.args.no_execute}")
        
        # Initialize observer bridge if enabled
        if self.args.enable_observer and self.robot:
            logger.info("Initializing observer bridge...")
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
            logger.info("Observer bridge started")
        else:
            logger.debug(f"Observer bridge disabled: enable_observer={self.args.enable_observer}, robot={self.robot is not None}")
        
        logger.info("Initialization complete")
    
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
            return create_video_source(
                'imageclient',
                robot=self.robot,
                source_name=self.args.ptz_source,
                quality=75,
                include_depth=include_depth
            )
    
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
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Get person detection from observer if available
                person_detection = None
                if self.observer_bridge and self.observer_bridge.has_person():
                    person_detection = self.observer_bridge.get_person_detection()
                    logger.debug(f"[Loop {loop_count}] Using observer person detection")
                
                # 1. PERCEPTION: Read frame and extract perception data
                logger.debug(f"[Loop {loop_count}] Reading perception...")
                perception = self.pipeline.read_perception(person_detection)
                if perception is None:
                    logger.warning("Failed to read perception (no frame)")
                    time.sleep(0.1)
                    continue
                logger.debug(f"[Loop {loop_count}] Perception complete: distance={perception.distance_m:.2f}m" if perception.distance_m else f"[Loop {loop_count}] Perception complete: distance=N/A")
                
                # Visualize if enabled
                if self.args.visualize or self.args.save_images:
                    # Get current frame for visualization
                    ret, frame, _ = self.video_source.read()
                    if ret and frame is not None:
                        key = visualize_pipeline_frame(
                            frame,
                            perception_data=perception,
                            person_detection=person_detection,
                            show=self.args.visualize,
                            save_dir=self.args.save_images,
                            iteration=loop_count
                        )
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            logger.info("User requested quit via visualization")
                            break
                
                # 2. DECISION: Compute comfort and select behavior
                logger.debug(f"[Loop {loop_count}] Computing comfort and behavior...")
                comfort, behavior = self.comfort_model.predict_behavior(perception)
                logger.debug(f"[Loop {loop_count}] Decision: comfort={comfort:.2f}, behavior={behavior.value}")
                
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
                    self.executor.execute_behavior(behavior)
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
        
        # Shutdown executor (releases lease, E-Stop)
        if self.executor:
            self.executor.shutdown()
        
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
        '--enable-observer',
        action='store_true',
        help='Enable people_observer for person detection and PTZ tracking'
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
        help='Enable verbose debug logging'
    )
    
    args = parser.parse_args()
    
    # Update logging level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose mode enabled")
    
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
