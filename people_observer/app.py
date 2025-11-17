"""CLI entrypoint for People Observer.

Usage (auth via Activate.ps1):
    python -m people_observer.app <ROBOT_IP> [--mode bearing|transform] [--dry-run] [--visualize]

No user/password flags are required; env-based authentication is assumed.
"""
import argparse
import logging

import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.client.spot_cam.ptz import PtzClient

from .config import RuntimeConfig, TRANSFORM_MODE
from .io_robot import connect, ensure_clients, configure_stream
from .cameras import fetch_image_sources
from .tracker import run_loop

logger = logging.getLogger(__name__)


def main(argv=None):
    ap = argparse.ArgumentParser(description="YOLO-based person tracking with PTZ control")
    bosdyn.client.util.add_base_arguments(ap)  # Adds 'hostname' as positional arg
    ap.add_argument("--mode", choices=["bearing", "transform"], default=TRANSFORM_MODE,
                    help="Geometry mode: 'transform' uses full 3D transforms (default), 'bearing' uses simple projection")
    ap.add_argument("--once", action="store_true", 
                    help="Run one cycle and exit (dev)")
    ap.add_argument("--exit-on-detection", action="store_true", 
                    help="Exit after successfully detecting and commanding PTZ to a person")
    ap.add_argument("--dry-run", action="store_true", 
                    help="Skip PTZ commands, log detections only")
    ap.add_argument("--visualize", action="store_true", 
                    help="Show live detection visualization with OpenCV")
    ap.add_argument("--save-images", type=str, metavar="DIR",
                    help="Save annotated frames to specified directory")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = RuntimeConfig()
    cfg.observer_mode = args.mode
    cfg.dry_run = args.dry_run
    cfg.once = args.once
    cfg.exit_on_detection = args.exit_on_detection
    cfg.visualize = args.visualize
    cfg.save_images = args.save_images
    
    # Validate and create save_images directory if specified
    if cfg.save_images:
        import os
        abs_path = os.path.abspath(cfg.save_images)
        logger.info(f"Creating save_images directory: {abs_path}")
        os.makedirs(cfg.save_images, exist_ok=True)
        if os.path.exists(abs_path):
            logger.info(f"SAVE IMAGES MODE: Directory confirmed at {abs_path}")
        else:
            logger.error(f"Failed to create directory: {abs_path}")
    
    if cfg.dry_run:
        logger.info("DRY-RUN MODE: PTZ commands will be skipped")
    if cfg.once:
        logger.info("ONCE MODE: Will run single detection cycle and exit")
    if cfg.exit_on_detection:
        logger.info("EXIT-ON-DETECTION MODE: Will exit after detecting and commanding PTZ to a person")
    if cfg.visualize:
        logger.info("VISUALIZATION MODE: OpenCV windows will display detections")

    logger.info(f"Connecting to robot at {args.hostname}")
    robot = connect(args.hostname)
    image_client, compositor, stream_quality = ensure_clients(robot)
    configure_stream(robot, cfg)
    
    # Fetch camera intrinsics at startup
    logger.info("Fetching camera intrinsics from robot...")
    image_sources = fetch_image_sources(image_client)

    ptz_client = robot.ensure_client(PtzClient.default_service_name)

    try:
        run_loop(robot, image_client, ptz_client, cfg)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
