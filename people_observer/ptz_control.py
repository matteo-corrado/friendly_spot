"""PTZ command helpers.

Functions:
- set_ptz(ptz_client, ptz_name, pan_deg, tilt_deg, zoom, dry_run)
    Send an absolute PTZ position in degrees via Spot CAM API.
"""
import logging

import numpy as np
from bosdyn.client.spot_cam.ptz import PtzClient
from bosdyn.api.spot_cam import ptz_pb2

from .config import RuntimeConfig

logger = logging.getLogger(__name__)


def set_ptz(ptz_client: PtzClient, ptz_name: str, pan_deg: float, tilt_deg: float, zoom: float = 1.0, dry_run: bool = False):
    """Set PTZ position. If dry_run=True, log command instead of executing.
    
    Args:
        ptz_client: Spot CAM PTZ client
        ptz_name: PTZ device name ('mech' or 'digi')
        pan_deg: Pan angle in degrees [0, 360]
        tilt_deg: Tilt angle in degrees [-30, 100]
        zoom: Zoom level [1.0, 30.0] where 1.0 = no zoom
        dry_run: If True, log command without executing
    """
    if dry_run:
        logger.info(f"[DRY-RUN] PTZ command: pan={pan_deg:.2f}deg, tilt={tilt_deg:.2f}deg, zoom={zoom:.2f}")
    else:
        # Create PTZ description (name of the PTZ - typically "mech" for mechanical PTZ)
        ptz_desc = ptz_pb2.PtzDescription(name=ptz_name)
        
        # Get current PTZ position to log state before commanding
        try:
            current_pos = ptz_client.get_ptz_position(ptz_desc)
            logger.info(f"Current PTZ state: pan={current_pos.pan.value:.2f}°, tilt={current_pos.tilt.value:.2f}°, zoom={current_pos.zoom.value:.2f}")
        except Exception as e:
            logger.warning(f"Could not query current PTZ position: {e}")
        
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
        
        # Send command to robot
        try:
            logger.info(f"Commanding PTZ: pan={pan_deg:.2f}°, tilt={tilt_deg:.2f}°, zoom={zoom:.2f}")
            ptz_client.set_ptz_position(ptz_desc, pan_deg, tilt_deg, zoom)
            logger.info("PTZ command successful")
        except Exception as e:
            logger.error(f"PTZ command failed: {type(e).__name__}: {e}")
