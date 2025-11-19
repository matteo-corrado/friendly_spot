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
    --visualize            Show perception visualizations (TODO)
    --verbose              Enable verbose robot debugging output
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
        
        # Connect to robot if specified
        if not self.args.webcam:
            logger.info(f"Connecting to robot at {self.args.robot}...")
            self.robot = create_robot(
                hostname=self.args.robot,
                client_name='FriendlySpot',
                register_spot_cam=True,  # Enable PTZ/compositor clients
                verbose=self.args.verbose
            )
            logger.info("Robot connection established")
        
        # Create video source
        self.video_source = self._create_video_source()
        
        # Initialize perception pipeline
        logger.info("Initializing perception pipeline...")
        self.pipeline = PerceptionPipeline(self.video_source, robot=self.robot)
        
        # Initialize behavior planner
        logger.info("Initializing comfort model...")
        self.comfort_model = ComfortModel()
        
        # Initialize behavior executor (only if robot connected and execution enabled)
        if self.robot and not self.args.no_execute:
            logger.info("Initializing behavior executor...")
            self.executor = BehaviorExecutor(self.robot)
            # Note: Lease/E-Stop managed automatically via context managers in execute_behavior()
        
        logger.info("Initialization complete")
    
    def _create_video_source(self):
        """Create video source based on arguments."""
        if self.args.webcam:
            logger.info("Using webcam video source")
            return create_video_source('webcam', device=0)
        
        elif self.args.webrtc:
            logger.info("Using WebRTC video source")
            # TODO: WebRTC not yet implemented
            return create_video_source('webrtc', robot=self.robot)
        
        else:
            logger.info(f"Using ImageClient video source (camera: {self.args.ptz_source})")
            return create_video_source(
                'imageclient',
                robot=self.robot,
                source_name=self.args.ptz_source,
                quality=75
            )
    
    def run(self):
        """Main perception to decision to execution loop."""
        self.running = True
        logger.info(f"Starting pipeline at {self.args.rate} Hz...")
        logger.info("Press Ctrl+C to stop")
        
        loop_count = 0
        loop_period = 1.0 / self.args.rate
        
        try:
            while self.running:
                loop_start = time.time()
                
                # 1. PERCEPTION: Read frame and extract perception data
                perception = self.pipeline.read_perception()
                if perception is None:
                    logger.warning("Failed to read perception (no frame)")
                    time.sleep(0.1)
                    continue
                
                # 2. DECISION: Compute comfort and select behavior
                comfort, behavior = self.comfort_model.predict_behavior(perception)
                
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
                    self.executor.execute_behavior(behavior)
                
                # Rate limiting
                loop_count += 1
                elapsed = time.time() - loop_start
                sleep_time = max(0, loop_period - elapsed)
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
        '--visualize',
        action='store_true',
        help='Show perception visualizations (TODO: not implemented)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose robot debugging output'
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
