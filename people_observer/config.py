"""Configuration for the people observer.

Accurate camera source names come from Boston Dynamics documentation and examples.
Avoid hardcoding HFOVs/yaws; prefer using transforms when possible. For bearing-only
mapping from pixel X to yaw, HFOVs are needed; keep them in one place and update
per official docs for your robot/camera firmware.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Surround cameras available on Spot (names per BD examples and Image service):
SURROUND_SOURCES: List[str] = [
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
    "back_fisheye_image",
]

# Optional: approximate HFOVs per camera for bearing-only mapping.
# WARNING: Replace with values from BD documentation for your hardware/firmware.
SURROUND_HFOV_DEG: Dict[str, float] = {
    "frontleft_fisheye_image": 133.0,
    "frontright_fisheye_image": 133.0,
    "left_fisheye_image": 133.0,
    "right_fisheye_image": 133.0,
    "back_fisheye_image": 133.0,
}

# PTZ/compositor/stream settings
PTZ_NAME = "ptz"
COMPOSITOR_SCREEN = "mech"
TARGET_BITRATE = 2_000_000

# Detection thresholds
PERSON_CLASS_ID = 0
MIN_CONFIDENCE = 0.30
MIN_AREA_PX = 600

# Loop pacing
LOOP_HZ = 7

# PTZ control policy
PAN_DEADBAND_DEG = 1.0
TILT_DEADBAND_DEG = 1.0
MAX_DEG_PER_STEP = 8.0
DEFAULT_TILT_DEG = -5.0

@dataclass
class ObserverMode:
    # "bearing" uses HFOV+yaw mapping; "transform" uses frame transforms and intrinsics when available.
    mode: str = "bearing"

@dataclass
class RuntimeConfig:
    sources: List[str] = field(default_factory=lambda: SURROUND_SOURCES)
    hfov_deg: Dict[str, float] = field(default_factory=lambda: SURROUND_HFOV_DEG)
    ptz_name: str = PTZ_NAME
    compositor_screen: str = COMPOSITOR_SCREEN
    target_bitrate: int = TARGET_BITRATE
    min_conf: float = MIN_CONFIDENCE
    min_area_px: int = MIN_AREA_PX
    loop_hz: int = LOOP_HZ
    pan_deadband_deg: float = PAN_DEADBAND_DEG
    tilt_deadband_deg: float = TILT_DEADBAND_DEG
    max_deg_per_step: float = MAX_DEG_PER_STEP
    default_tilt_deg: float = DEFAULT_TILT_DEG
    observer_mode: ObserverMode = field(default_factory=ObserverMode)
