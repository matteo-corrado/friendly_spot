"""CLI entrypoint for People Observer.

Usage (auth via Activate.ps1):
    python -m friendly_spot.people_observer.app --hostname <ROBOT_IP> [--mode bearing|transform]

No user/password flags are required; env-based authentication is assumed.
"""
import argparse

import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.client.spot_cam.ptz import PtzClient

from .config import RuntimeConfig, ObserverMode
from .io_robot import connect, ensure_clients, configure_stream
from .cameras import ensure_available_sources
from .tracker import run_loop


def main(argv=None):
    ap = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(ap)
    ap.add_argument("--mode", choices=["bearing", "transform"], default="bearing")
    ap.add_argument("--once", action="store_true", help="Run one cycle and exit (dev)")
    args = ap.parse_args(argv)

    cfg = RuntimeConfig()
    cfg.observer_mode = ObserverMode(args.mode)

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
