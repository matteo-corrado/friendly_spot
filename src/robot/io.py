# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Unified robot I/O module consolidating Spot SDK operations including robot connection,
# authentication, client management, and command utilities with comprehensive error handling
# Acknowledgements: Boston Dynamics Spot SDK (hello_spot, estop, wasd examples) for connection patterns,
# Claude for client management architecture and error handling design

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
from bosdyn.client.world_object import WorldObjectClient

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
    _world_object: Optional[WorldObjectClient] = None
    
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
    def world_object(self) -> WorldObjectClient:
        """Get WorldObjectClient (fiducial/object detection)."""
        if self._world_object is None:
            self._world_object = self.robot.ensure_client(WorldObjectClient.default_service_name)
        return self._world_object
    
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
        # Normal acquire (fails if another client has lease):
        with ManagedLease(robot) as lease_client:
            command_client.robot_command(cmd)
        
        # Force take lease from tablet/other clients:
        with ManagedLease(robot, force_take=True) as lease_client:
            command_client.robot_command(cmd)
    
    Args:
        robot: Authenticated Robot instance
        must_acquire: If True, raises error if lease cannot be acquired
        return_at_exit: If True, returns lease on context exit
        force_take: If True, uses take() to forcefully claim lease from other holders
                   (including tablet). Use with caution - interrupts other operations.
    """
    
    def __init__(
        self,
        robot: Robot,
        must_acquire: bool = True,
        return_at_exit: bool = True,
        force_take: bool = False
    ):
        self.robot = robot
        self.must_acquire = must_acquire
        self.return_at_exit = return_at_exit
        self.force_take = force_take
        self.lease_client = robot.ensure_client(LeaseClient.default_service_name)
        self.lease_keep_alive = None
    
    def __enter__(self) -> LeaseClient:
        """Acquire or take lease and start keep-alive."""
        # If force_take is enabled, forcefully take the lease from current holder
        if self.force_take:
            try:
                from bosdyn.client.lease import ResourceAlreadyClaimedError
                logger.info("Force-taking lease from current holder...")
                lease = self.lease_client.take()
                logger.info(f"Successfully took lease: {lease.lease_proto.resource}")
                # Add the taken lease to the wallet
                if lease and self.robot.lease_wallet:
                    self.robot.lease_wallet.add(lease)
            except Exception as e:
                logger.error(f"Failed to force-take lease: {e}")
                if self.must_acquire:
                    raise
        
        # Create LeaseKeepAlive with explicit wallet reference (SDK pattern)
        # The wallet is shared across all clients from the same Robot instance
        # If we already took the lease above, LeaseKeepAlive will use it from the wallet
        self.lease_keep_alive = LeaseKeepAlive(
            self.lease_client,
            lease_wallet=self.robot.lease_wallet,
            must_acquire=self.must_acquire and not self.force_take,  # Don't re-acquire if we took it
            return_at_exit=self.return_at_exit
        )
        self.lease_keep_alive.__enter__()
        logger.info("Robot lease acquired and keep-alive started")
        return self.lease_client
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Return lease and stop keep-alive."""
        if self.lease_keep_alive:
            try:
                self.lease_keep_alive.__exit__(exc_type, exc_val, exc_tb)
                logger.info("Robot lease returned and keep-alive stopped")
            except Exception as e:
                logger.warning(f"Error during lease cleanup: {e}")
                # Don't re-raise - cleanup errors shouldn't mask original exceptions


class ManagedEstop:
    """Context manager for software E-Stop with automatic keep-alive.
    
    Registers a software E-Stop endpoint and maintains keep-alive signals.
    Critical for safety - robot will E-Stop if keep-alive stops.
    
    Usage:
        with ManagedEstop(robot, name="MyApp"):
            # E-Stop is active here
            command_client.robot_command(cmd)
        # E-Stop automatically deregistered
        
        # Skip E-Stop registration when taking lease from tablet:
        with ManagedEstop(robot, skip_if_active=True):
            # Uses existing E-Stop from tablet
            command_client.robot_command(cmd)
    
    Args:
        robot: Authenticated Robot instance
        name: Name for this E-Stop endpoint
        timeout_sec: E-Stop timeout in seconds (default: 9.0)
        skip_if_active: If True, skip E-Stop registration if robot motors are on
                       (assumes existing E-Stop from tablet/other client)
    """
    
    def __init__(
        self,
        robot: Robot,
        name: str = "FriendlySpot",
        timeout_sec: float = 9.0,
        skip_if_active: bool = False
    ):
        self.robot = robot
        self.name = name
        self.timeout_sec = timeout_sec
        self.skip_if_active = skip_if_active
        self.estop_client = None
        self.estop_endpoint = None
        self.estop_keep_alive = None
        self.skipped = False
    
    def __enter__(self) -> EstopEndpoint:
        """Register E-Stop endpoint and start keep-alive."""
        # Check if we should skip E-Stop registration
        if self.skip_if_active:
            try:
                # Check if robot is already powered on (indicates active E-Stop)
                if self.robot.is_powered_on():
                    logger.info("Motors already on - skipping E-Stop registration (using existing E-Stop)")
                    self.skipped = True
                    return None
            except Exception as e:
                logger.warning(f"Could not check power state: {e} - attempting E-Stop registration")
        
        # Normal E-Stop registration
        try:
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
        except Exception as e:
            # If registration fails and skip_if_active is set, assume existing E-Stop is fine
            if self.skip_if_active and "motors are on" in str(e).lower():
                logger.warning(f"E-Stop registration failed (motors already on) - using existing E-Stop: {e}")
                self.skipped = True
                return None
            else:
                # Re-raise if this is an unexpected error
                logger.error(f"Failed to register E-Stop: {e}")
                raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Deregister E-Stop and stop keep-alive."""
        if self.skipped:
            logger.debug("E-Stop was not registered - no cleanup needed")
            return
        
        if self.estop_keep_alive:
            try:
                self.estop_keep_alive.shutdown()
                logger.info("Software E-Stop deregistered")
            except Exception as e:
                logger.warning(f"Error during E-Stop cleanup: {e}")


def take_lease(robot: Robot) -> bool:
    """Forcefully take the lease from the current holder (e.g., tablet).
    
    Use this when another client has the lease and you need to take control.
    The current lease holder will lose control immediately.
    
    Args:
        robot: Robot instance
    
    Returns:
        True if lease was successfully taken, False otherwise
    
    Example:
        if take_lease(robot):
            print("Lease acquired - you now have control")
        else:
            print("Failed to take lease")
    """
    try:
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease = lease_client.take()
        robot.lease_wallet.add(lease)
        logger.info(f"Successfully took lease: {lease.lease_proto.resource}")
        return True
    except Exception as e:
        logger.error(f"Failed to take lease: {e}")
        return False


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
