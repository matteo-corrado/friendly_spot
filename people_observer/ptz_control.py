# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/17/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: PTZ camera command helpers for absolute position control with validation,
# range clamping, and dry-run mode for testing
# Acknowledgements: Boston Dynamics Spot SDK PtzClient API documentation,
# SDK ptz_pb2 for PTZ command structure

"""PTZ command helpers.

Functions:
- set_ptz(ptz_client, ptz_name, pan_deg, tilt_deg, zoom, dry_run)
    Send an absolute PTZ position in degrees via Spot CAM API (async, non-blocking).
- is_ptz_ready()
    Check if PTZ is ready for a new command (previous command completed).
"""
import logging

import numpy as np
from bosdyn.client.spot_cam.ptz import PtzClient
from bosdyn.api.spot_cam import ptz_pb2

from .config import RuntimeConfig

logger = logging.getLogger(__name__)

# Global tracking of pending PTZ command
_pending_ptz_future = None
_last_ptz_command = None


def is_ptz_ready() -> bool:
    """Check if PTZ is ready to accept a new command.
    
    Returns:
        True if no command is pending or previous command completed, False otherwise
    """
    global _pending_ptz_future
    
    if _pending_ptz_future is None:
        return True
    
    # Check if the future is done
    if _pending_ptz_future.done():
        try:
            # Get result to check for exceptions
            _pending_ptz_future.result()
            logger.debug("Previous PTZ command completed successfully")
        except Exception as e:
            logger.warning(f"Previous PTZ command failed: {e}")
        _pending_ptz_future = None
        return True
    
    logger.debug("PTZ busy - previous command still executing")
    return False


def set_ptz(ptz_client: PtzClient, ptz_name: str, pan_deg: float, tilt_deg: float, zoom: float = 1.0, dry_run: bool = False, force: bool = False):
    """Set PTZ position asynchronously (non-blocking). If dry_run=True, log command instead of executing.
    
    This function sends async commands to avoid overloading the PTZ. If a previous command is still
    executing, the new command will be skipped unless force=True.
    
    Args:
        ptz_client: Spot CAM PTZ client
        ptz_name: PTZ device name ('mech' or 'digi')
        pan_deg: Pan angle in degrees [0, 360]
        tilt_deg: Tilt angle in degrees [-30, 100]
        zoom: Zoom level [1.0, 30.0] where 1.0 = no zoom
        dry_run: If True, log command without executing
        force: If True, send command even if previous command is pending
        
    Returns:
        True if command was sent, False if skipped due to pending command
    """
    global _pending_ptz_future, _last_ptz_command
    
    if dry_run:
        logger.info(f"[DRY-RUN] PTZ command: pan={pan_deg:.2f}deg, tilt={tilt_deg:.2f}deg, zoom={zoom:.2f}")
        return True
    
    # Check if PTZ is ready for new command
    if not force and not is_ptz_ready():
        logger.debug(f"Skipping PTZ command (pan={pan_deg:.2f}°, tilt={tilt_deg:.2f}°) - previous command still pending")
        return False
    
    # Create PTZ description (name of the PTZ - typically "mech" for mechanical PTZ)
    ptz_desc = ptz_pb2.PtzDescription(name=ptz_name)
    
    # Validate angles are in expected ranges
    if not (0.0 <= pan_deg <= 360.0):
        logger.error(f"Pan angle {pan_deg:.2f}° outside valid range [0, 360], clamping")
        pan_deg = np.clip(pan_deg, 0.0, 360.0)
    
    if not (-30.0 <= tilt_deg <= 100.0):
        logger.error(f"Tilt angle {tilt_deg:.2f}° outside valid range [-30, 100], clamping")
        tilt_deg = np.clip(tilt_deg, -30.0, 100.0)
        
    if not (1.0 <= zoom <= 30.0):
        logger.error(f"Zoom {zoom:.2f} outside valid range [1.0, 30.0], clamping")
        zoom = np.clip(zoom, 1.0, 30.0)
    
    # Send async command to robot
    try:
        logger.info(f"Commanding PTZ (async): pan={pan_deg:.2f}°, tilt={tilt_deg:.2f}°, zoom={zoom:.2f}")
        _pending_ptz_future = ptz_client.set_ptz_position_async(ptz_desc, pan_deg, tilt_deg, zoom)
        _last_ptz_command = {'pan': pan_deg, 'tilt': tilt_deg, 'zoom': zoom}
        logger.debug("PTZ command sent (async)")
        return True
    except Exception as e:
        logger.error(f"PTZ command failed: {type(e).__name__}: {e}")
        _pending_ptz_future = None
        return False
