"""Robot I/O helpers: SDK setup, authentication, clients, stream config.

References:
- bosdyn.client.util.authenticate
- bosdyn.client.spot_cam.compositor.CompositorClient
- bosdyn.client.spot_cam.streamquality.StreamQualityClient
- bosdyn.client.image.ImageClient

Auth is provided by venv Activate.ps1; do not expose credentials via CLI.

Functions
- connect(hostname) -> robot: create SDK, authenticate from env, time sync.
- ensure_clients(robot) -> (ImageClient, CompositorClient, StreamQualityClient)
- configure_stream(robot, cfg): set compositor screen and bitrate.
"""
from typing import Tuple

import bosdyn.client
import bosdyn.client.util
from bosdyn.client import create_standard_sdk, spot_cam
from bosdyn.client.image import ImageClient
from bosdyn.client.spot_cam.compositor import CompositorClient
from bosdyn.client.spot_cam.streamquality import StreamQualityClient

from .config import RuntimeConfig


def connect(hostname: str):
    """Create and authenticate a robot connection using env-based credentials.

    Input: hostname/IP string
    Output: robot instance with time sync performed.
    """
    sdk = create_standard_sdk("PeopleObserver")
    
    # Register all Spot CAM service clients (CompositorClient, StreamQualityClient, PtzClient, etc.)
    spot_cam.register_all_service_clients(sdk)
    
    robot = sdk.create_robot(hostname)
    # Rely on env-based auth (Activate.ps1), no user/pass CLI
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    return robot


def ensure_clients(robot) -> Tuple[ImageClient, CompositorClient, StreamQualityClient]:
    """Return common service clients used by this package."""
    image = robot.ensure_client(ImageClient.default_service_name)
    comp = robot.ensure_client(CompositorClient.default_service_name)
    sq = robot.ensure_client(StreamQualityClient.default_service_name)
    return image, comp, sq


def configure_stream(robot, cfg: RuntimeConfig):
    """Configure compositor screen and target bitrate for Spot CAM streaming."""
    # Compositor screen and WebRTC stream bitrate
    robot.ensure_client(CompositorClient.default_service_name).set_screen(cfg.ptz.compositor_screen)
    robot.ensure_client(StreamQualityClient.default_service_name).set_stream_params(
        target_bitrate=cfg.ptz.target_bitrate
    )
