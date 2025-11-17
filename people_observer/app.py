"""CLI entrypoint for People Observer.

Usage (auth via Activate.ps1):
    python -m friendly_spot.people_observer.app --hostname <ROBOT_IP> [--mode bearing|transform] [--dry-run]

No user/password flags are required; env-based authentication is assumed.
"""
import argparse
import logging

import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.client.spot_cam.ptz import PtzClient

from .config import RuntimeConfig, ObserverMode
from .io_robot import connect, ensure_clients, configure_stream
from .cameras import ensure_available_sources
from .tracker import run_loop

logger = logging.getLogger(__name__)


def main(argv=None):
    ap = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(ap)
    ap.add_argument("--mode", choices=["bearing", "transform"], default="bearing")
    ap.add_argument("--once", action="store_true", help="Run one cycle and exit (dev)")
    ap.add_argument("--dry-run", action="store_true", help="Skip PTZ commands, log detections only")
    ap.add_argument("--visualize", action="store_true", help="Show live detection visualization with OpenCV")
    args = ap.parse_args(argv)

    cfg = RuntimeConfig()
    cfg.observer_mode = ObserverMode(args.mode)
    cfg.dry_run = args.dry_run
    cfg.once = args.once
    cfg.visualize = args.visualize
    
    if cfg.dry_run:
        logger.info("DRY-RUN MODE: PTZ commands will be skipped")
    if cfg.once:
        logger.info("ONCE MODE: Will run single detection cycle and exit")
    if cfg.visualize:
        logger.info("VISUALIZATION MODE: OpenCV windows will display detections")

    robot = connect(args.hostname)
    image_client, compositor, stream_quality = ensure_clients(robot)
    configure_stream(robot, cfg)
    cfg.sources = ensure_available_sources(image_client, cfg.sources)

    ptz_client = robot.ensure_client(PtzClient.default_service_name)

    # For now, --once is treated as a short run with a single iteration; tracker handles pacing
    try:
        run_loop(robot, image_client, ptz_client, cfg)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
