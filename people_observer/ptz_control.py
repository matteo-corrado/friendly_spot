"""PTZ command helpers with smoothing and deadbands.

Functions
- apply_deadband(target_deg, current_deg, deadband_deg) -> float
    Suppress small changes to reduce jitter.
- clamp_step(target_deg, current_deg, max_step_deg) -> float
    Limit max degrees per update step.
- set_ptz(ptz_client, ptz_name, pan_deg, tilt_deg, zoom=0.0, dry_run=False)
    Send an absolute PTZ position in radians via Spot CAM API.
"""
import logging
import math

from bosdyn.client.spot_cam.ptz import PtzClient, PtzPosition

from .config import RuntimeConfig

logger = logging.getLogger(__name__)


def apply_deadband(target_deg: float, current_deg: float, deadband_deg: float) -> float:
    if abs(target_deg - current_deg) < deadband_deg:
        return current_deg
    return target_deg


def clamp_step(target_deg: float, current_deg: float, max_step_deg: float) -> float:
    delta = target_deg - current_deg
    if abs(delta) > max_step_deg:
        target_deg = current_deg + math.copysign(max_step_deg, delta)
    return target_deg


def set_ptz(ptz_client: PtzClient, ptz_name: str, pan_deg: float, tilt_deg: float, zoom: float = 0.0, dry_run: bool = False):
    """Set PTZ position. If dry_run=True, log command instead of executing."""
    if dry_run:
        logger.info(f"[DRY-RUN] PTZ command: pan={pan_deg:.2f}deg, tilt={tilt_deg:.2f}deg, zoom={zoom:.2f}")
    else:
        pos = PtzPosition(pan=math.radians(pan_deg), tilt=math.radians(tilt_deg), zoom=zoom)
        ptz_client.set_ptz_position(ptz_name, pos, 0.0)
