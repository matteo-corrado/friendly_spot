"""Unified robot I/O module for Spot SDK operations.

Consolidates robot connection, authentication, client management, and command
utilities following Boston Dynamics SDK best practices. Provides a clean
interface for all modules that interact with Spot.

Usage:
    from robot_io import create_robot, RobotClients
    
    robot = create_robot(hostname, register_spot_cam=True)
    clients = RobotClients(robot)
    clients.command_client.robot_command(cmd)

References:
    - bosdyn.client.util (authenticate, setup_logging)
    - bosdyn.client.create_standard_sdk
    - SDK examples: hello_spot, estop, wasd
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import bosdyn.client
import bosdyn.client.util
from bosdyn.client import Robot, create_standard_sdk
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.power import PowerClient

# Spot CAM imports (optional, only if needed)
try:
    from bosdyn.client import spot_cam
    from bosdyn.client.spot_cam.compositor import CompositorClient
    from bosdyn.client.spot_cam.streamquality import StreamQualityClient
    from bosdyn.client.spot_cam.ptz import PtzClient
    SPOT_CAM_AVAILABLE = True
except ImportError:
    SPOT_CAM_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_robot(
    hostname: str,
    client_name: str = "FriendlySpot",
    register_spot_cam: bool = False,
    verbose: bool = False
) -> Robot:
    """Create and authenticate a robot connection.
    
    Follows SDK best practices:
    - Uses bosdyn.client.util.authenticate() for flexible auth
      (supports token, env vars, or interactive prompt)
    - Performs time sync (required for commands)
    - Optionally registers Spot CAM services
    
    Args:
        hostname: Robot IP address or DNS name
        client_name: Identifier for this client application
        register_spot_cam: Register Spot CAM service clients (PTZ, compositor, etc.)
        verbose: Enable debug logging
    
    Returns:
        Authenticated Robot instance with time sync completed
    
    Raises:
        RuntimeError: If authentication fails or time sync fails
    """
    # Setup logging
    bosdyn.client.util.setup_logging(verbose=verbose)
    
    # Create SDK instance
    sdk = create_standard_sdk(client_name)
    
    # Register Spot CAM services if requested
    if register_spot_cam:
        if not SPOT_CAM_AVAILABLE:
            logger.warning("Spot CAM services requested but not available")
        else:
            spot_cam.register_all_service_clients(sdk)
            logger.info("Spot CAM services registered")
    
    # Create robot connection
    robot = sdk.create_robot(hostname)
    
    # Authenticate (tries token, env vars, then interactive prompt)
    # Set credentials in environment variables to avoid interactive prompt
    try:
        bosdyn.client.util.authenticate(robot)
        logger.info(f"Authenticated with robot at {hostname}")
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise
    
    # Establish time sync (required for commands)
    try:
        robot.time_sync.wait_for_sync()
        logger.info("Time sync established")
    except Exception as e:
        logger.error(f"Time sync failed: {e}")
        raise
    
    return robot


@dataclass
class RobotClients:
    """Container for commonly used robot service clients.
    
    Lazily initializes clients on first access. Use robot.ensure_client()
    pattern from SDK to avoid creating unused clients.
    
    Attributes:
        robot: Robot instance
        command: RobotCommandClient (for movement, posture)
        state: RobotStateClient (for robot state queries)
        lease: LeaseClient (for lease management)
        estop: EstopClient (for E-Stop management)
        power: PowerClient (for power on/off)
        image: ImageClient (for camera images)
        compositor: CompositorClient (Spot CAM, if available)
        stream_quality: StreamQualityClient (Spot CAM, if available)
        ptz: PtzClient (Spot CAM, if available)
    """
    robot: Robot
    
    # Core clients (cached after first access)
    _command: Optional[RobotCommandClient] = None
    _state: Optional[RobotStateClient] = None
    _lease: Optional[LeaseClient] = None
    _estop: Optional[EstopClient] = None
    _power: Optional[PowerClient] = None
    _image: Optional[ImageClient] = None
    
    # Spot CAM clients (cached after first access)
    _compositor: Optional['CompositorClient'] = None
    _stream_quality: Optional['StreamQualityClient'] = None
    _ptz: Optional['PtzClient'] = None
    
    @property
    def command(self) -> RobotCommandClient:
        """Get RobotCommandClient (movement, posture commands)."""
        if self._command is None:
            self._command = self.robot.ensure_client(RobotCommandClient.default_service_name)
        return self._command
    
    @property
    def state(self) -> RobotStateClient:
        """Get RobotStateClient (robot state queries)."""
        if self._state is None:
            self._state = self.robot.ensure_client(RobotStateClient.default_service_name)
        return self._state
    
    @property
    def lease(self) -> LeaseClient:
        """Get LeaseClient (lease management)."""
        if self._lease is None:
            self._lease = self.robot.ensure_client(LeaseClient.default_service_name)
        return self._lease
    
    @property
    def estop(self) -> EstopClient:
        """Get EstopClient (E-Stop management)."""
        if self._estop is None:
            self._estop = self.robot.ensure_client(EstopClient.default_service_name)
        return self._estop
    
    @property
    def power(self) -> PowerClient:
        """Get PowerClient (power on/off)."""
        if self._power is None:
            self._power = self.robot.ensure_client(PowerClient.default_service_name)
        return self._power
    
    @property
    def image(self) -> ImageClient:
        """Get ImageClient (camera images)."""
        if self._image is None:
            self._image = self.robot.ensure_client(ImageClient.default_service_name)
        return self._image
    
    @property
    def compositor(self) -> 'CompositorClient':
        """Get CompositorClient (Spot CAM compositor control)."""
        if not SPOT_CAM_AVAILABLE:
            raise RuntimeError("Spot CAM services not available")
        if self._compositor is None:
            self._compositor = self.robot.ensure_client(CompositorClient.default_service_name)
        return self._compositor
    
    @property
    def stream_quality(self) -> 'StreamQualityClient':
        """Get StreamQualityClient (Spot CAM stream quality control)."""
        if not SPOT_CAM_AVAILABLE:
            raise RuntimeError("Spot CAM services not available")
        if self._stream_quality is None:
            self._stream_quality = self.robot.ensure_client(StreamQualityClient.default_service_name)
        return self._stream_quality
    
    @property
    def ptz(self) -> 'PtzClient':
        """Get PtzClient (Spot CAM PTZ control)."""
        if not SPOT_CAM_AVAILABLE:
            raise RuntimeError("Spot CAM services not available")
        if self._ptz is None:
            self._ptz = self.robot.ensure_client(PtzClient.default_service_name)
        return self._ptz


class ManagedLease:
    """Context manager for robot lease with automatic keep-alive.
    
    Follows SDK pattern from hello_spot.py and other examples.
    Automatically acquires lease on entry, maintains keep-alive during use,
    and returns lease on exit.
    
    Usage:
        with ManagedLease(robot) as lease_client:
            # lease is active here
            command_client.robot_command(cmd)
        # lease automatically returned
    
    Args:
        robot: Authenticated Robot instance
        must_acquire: If True, raises error if lease cannot be acquired
        return_at_exit: If True, returns lease on context exit
    """
    
    def __init__(
        self,
        robot: Robot,
        must_acquire: bool = True,
        return_at_exit: bool = True
    ):
        self.robot = robot
        self.must_acquire = must_acquire
        self.return_at_exit = return_at_exit
        self.lease_client = robot.ensure_client(LeaseClient.default_service_name)
        self.lease_keep_alive = None
    
    def __enter__(self) -> LeaseClient:
        """Acquire lease and start keep-alive."""
        self.lease_keep_alive = LeaseKeepAlive(
            self.lease_client,
            must_acquire=self.must_acquire,
            return_at_exit=self.return_at_exit
        )
        self.lease_keep_alive.__enter__()
        logger.info("Acquired robot lease")
        return self.lease_client
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Return lease and stop keep-alive."""
        if self.lease_keep_alive:
            self.lease_keep_alive.__exit__(exc_type, exc_val, exc_tb)
            logger.info("Released robot lease")


class ManagedEstop:
    """Context manager for software E-Stop with automatic keep-alive.
    
    Registers a software E-Stop endpoint and maintains keep-alive signals.
    Critical for safety - robot will E-Stop if keep-alive stops.
    
    Usage:
        with ManagedEstop(robot, name="MyApp"):
            # E-Stop is active here
            command_client.robot_command(cmd)
        # E-Stop automatically deregistered
    
    Args:
        robot: Authenticated Robot instance
        name: Name for this E-Stop endpoint
        timeout_sec: E-Stop timeout in seconds (default: 9.0)
    """
    
    def __init__(
        self,
        robot: Robot,
        name: str = "FriendlySpot",
        timeout_sec: float = 9.0
    ):
        self.robot = robot
        self.name = name
        self.timeout_sec = timeout_sec
        self.estop_client = None
        self.estop_endpoint = None
        self.estop_keep_alive = None
    
    def __enter__(self) -> EstopEndpoint:
        """Register E-Stop endpoint and start keep-alive."""
        self.estop_client = self.robot.ensure_client(EstopClient.default_service_name)
        self.estop_endpoint = EstopEndpoint(
            self.estop_client,
            self.name,
            self.timeout_sec
        )
        self.estop_endpoint.force_simple_setup()
        self.estop_keep_alive = EstopKeepAlive(self.estop_endpoint)
        logger.info(f"Software E-Stop registered: {self.name}")
        return self.estop_endpoint
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Deregister E-Stop and stop keep-alive."""
        if self.estop_keep_alive:
            self.estop_keep_alive.shutdown()
        logger.info("Software E-Stop deregistered")


def check_estop(robot: Robot) -> bool:
    """Check if robot is E-Stopped.
    
    Args:
        robot: Robot instance
    
    Returns:
        True if robot is E-Stopped, False otherwise
    """
    return robot.is_estopped()


def check_powered(robot: Robot) -> bool:
    """Check if robot is powered on.
    
    Args:
        robot: Robot instance
    
    Returns:
        True if robot is powered on, False otherwise
    """
    return robot.is_powered_on()


def configure_stream(
    robot: Robot,
    compositor_screen: str = "mech",
    target_bitrate: int = 5000000
):
    """Configure Spot CAM compositor screen and stream bitrate.
    
    Used by people_observer and other PTZ streaming applications.
    
    Args:
        robot: Robot instance
        compositor_screen: Compositor screen name (e.g., "mech", "ptz")
        target_bitrate: Target bitrate in bits/sec (default: 5Mbps)
    
    Raises:
        RuntimeError: If Spot CAM services not available
    
    Example:
        configure_stream(robot, compositor_screen="ptz", target_bitrate=4000000)
    """
    if not SPOT_CAM_AVAILABLE:
        raise RuntimeError("Spot CAM services not available")
    
    compositor = robot.ensure_client(CompositorClient.default_service_name)
    stream_quality = robot.ensure_client(StreamQualityClient.default_service_name)
    
    compositor.set_screen(compositor_screen)
    stream_quality.set_stream_params(target_bitrate=target_bitrate)
    logger.info(f"Stream configured: screen={compositor_screen}, bitrate={target_bitrate}")
