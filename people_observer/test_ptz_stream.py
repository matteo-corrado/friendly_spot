# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/18/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Standalone CLI test utility for PTZ WebRTC streaming with video recording,
# frame display, statistics tracking, and graceful shutdown for development and debugging
# Acknowledgements: OpenCV VideoWriter documentation for MP4 recording,
# aiortc examples for frame processing, Claude for CLI argument design and overlay rendering

"""PTZ WebRTC streaming test script.

Simple command-line tool to test PTZ video streaming independently of detection pipeline.

Usage:
    python -m people_observer.test_ptz_stream ROBOT_IP [--duration SECONDS] [--save-video PATH]

Examples:
    # Stream for 10 seconds, display in window
    python -m people_observer.test_ptz_stream 192.168.80.3 --duration 10
    
    # Stream for 30 seconds, save to MP4
    python -m people_observer.test_ptz_stream 192.168.80.3 --duration 30 --save-video ptz_stream.mp4
    
    # Stream indefinitely, press Ctrl+C to stop
    python -m people_observer.test_ptz_stream 192.168.80.3

Requirements:
    pip install aiortc av opencv-python

TODO: Once facial recognition models are ready, add --facial-recognition flag
"""
import argparse
import logging
import sys
import time

import cv2
import numpy as np

# Support both direct execution and module import
try:
    from .io_robot import connect, configure_stream
    from .ptz_stream import PtzStream, PtzStreamConfig
    from .config import RuntimeConfig
except ImportError:
    # Direct execution fallback
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from io_robot import connect, configure_stream
    from ptz_stream import PtzStream, PtzStreamConfig
    from config import RuntimeConfig

logger = logging.getLogger(__name__)


def main():
    """Main entry point for PTZ stream testing."""
    parser = argparse.ArgumentParser(
        description="Test PTZ WebRTC video streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("hostname", help="Spot robot IP address or hostname")
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Stream duration in seconds (0 = infinite, Ctrl+C to stop)"
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Save stream to MP4 file (e.g., ptz_stream.mp4)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video window display"
    )
    parser.add_argument(
        "--sdp-port",
        type=int,
        default=31102,
        help="SDP port for WebRTC (default: 31102)"
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=5.0,
        help="Print statistics every N seconds (default: 5.0)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info("=" * 60)
    logger.info("PTZ WebRTC Streaming Test")
    logger.info("=" * 60)
    logger.info(f"Robot: {args.hostname}")
    logger.info(f"Duration: {'infinite' if args.duration == 0 else f'{args.duration}s'}")
    logger.info(f"Display: {'disabled' if args.no_display else 'enabled'}")
    logger.info(f"Save video: {args.save_video or 'disabled'}")
    logger.info("=" * 60)
    
    # Check aiortc availability
    try:
        import aiortc
        logger.info(f"aiortc version: {aiortc.__version__}")
    except ImportError:
        logger.error("aiortc not installed. Install with: pip install aiortc av")
        return 1
    
    # Connect to robot
    try:
        logger.info("Connecting to robot...")
        robot = connect(args.hostname)
        logger.info("Robot connected successfully")
        
        # Configure stream (compositor + bitrate)
        cfg = RuntimeConfig()  # Use default config
        logger.info(f"Configuring stream: screen={cfg.ptz.compositor_screen}, bitrate={cfg.ptz.target_bitrate}")
        configure_stream(robot, cfg)
        
    except Exception as e:
        logger.error(f"Failed to connect to robot: {e}", exc_info=True)
        return 1
    
    # Create stream configuration
    stream_config = PtzStreamConfig(
        sdp_port=args.sdp_port,
        frame_queue_size=30,
        connection_timeout_sec=15.0,
    )
    
    # Create PTZ stream
    ptz_stream = PtzStream(robot, stream_config)
    
    # Setup video writer if saving
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Will initialize once we get first frame (to determine resolution)
        logger.info(f"Video will be saved to: {args.save_video}")
    
    try:
        # Start streaming
        logger.info("Starting PTZ stream...")
        if not ptz_stream.start():
            logger.error("Failed to start stream")
            return 1
        
        logger.info("Stream started! Receiving frames...")
        logger.info("Press Ctrl+C to stop")
        
        # Stream processing loop
        start_time = time.time()
        last_stats_time = start_time
        frame_count = 0
        
        while True:
            # Check duration limit
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                logger.info(f"Duration limit reached ({args.duration}s)")
                break
            
            # Get next frame
            frame = ptz_stream.get_frame(timeout=1.0)
            if frame is None:
                if not ptz_stream.is_running():
                    logger.warning("Stream stopped unexpectedly")
                    break
                continue
            
            frame_count += 1
            
            # Initialize video writer on first frame
            if args.save_video and video_writer is None:
                h, w = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    args.save_video,
                    fourcc,
                    30.0,  # FPS (approximate)
                    (w, h)
                )
                logger.info(f"Video writer initialized: {w}x{h}")
            
            # Save frame to video
            if video_writer:
                video_writer.write(frame)
            
            # Display frame
            if not args.no_display:
                # Add frame info overlay
                display_frame = frame.copy()
                h, w = display_frame.shape[:2]
                
                # Frame counter
                cv2.putText(
                    display_frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Elapsed time
                elapsed = time.time() - start_time
                cv2.putText(
                    display_frame,
                    f"Time: {elapsed:.1f}s",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Stream stats
                stats = ptz_stream.get_stats()
                cv2.putText(
                    display_frame,
                    f"FPS: {stats['fps']:.1f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow("PTZ Stream", display_frame)
                
                # Check for quit key
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("User requested quit")
                    break
            
            # Print statistics periodically
            if time.time() - last_stats_time >= args.stats_interval:
                stats = ptz_stream.get_stats()
                logger.info(
                    f"Stats: {stats['frames_received']} frames, "
                    f"{stats['fps']:.1f} fps, "
                    f"{stats['frames_dropped']} dropped, "
                    f"queue: {stats['queue_size']}/{stream_config.frame_queue_size}"
                )
                last_stats_time = time.time()
        
    except KeyboardInterrupt:
        logger.info("\nCtrl+C received, stopping stream...")
    
    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        ptz_stream.stop()
        
        if video_writer:
            video_writer.release()
            logger.info(f"Video saved to: {args.save_video}")
        
        if not args.no_display:
            cv2.destroyAllWindows()
        
        # Final statistics
        stats = ptz_stream.get_stats()
        logger.info("=" * 60)
        logger.info("Final Statistics:")
        logger.info(f"  Total frames: {stats['frames_received']}")
        logger.info(f"  Frames dropped: {stats['frames_dropped']}")
        logger.info(f"  Average FPS: {stats['fps']:.1f}")
        logger.info(f"  Duration: {stats['duration_sec']:.1f}s")
        logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
