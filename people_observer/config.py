"""Configuration for the people observer.

Accurate camera source names come from Boston Dynamics documentation and examples.
Avoid hardcoding HFOVs/yaws; prefer using transforms when possible. For bearing-only
mapping from pixel X to yaw, HFOVs are needed; keep them in one place and update
per official docs for your robot/camera firmware.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

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
FALLBACK_HFOV_DEG = 133.0  # Default HFOV if camera-specific value not available
SURROUND_HFOV_DEG: Dict[str, float] = {
    "frontleft_fisheye_image": 133.0,
    "frontright_fisheye_image": 133.0,
    "left_fisheye_image": 133.0,
    "right_fisheye_image": 133.0,
    "back_fisheye_image": 133.0,
}

# Camera yaw offsets in robot frame (degrees, clockwise from front)
# WARNING: Update these based on your robot's actual camera mounting angles
SURROUND_YAW_DEG: Dict[str, float] = {
    "frontleft_fisheye_image": 50.0,
    "frontright_fisheye_image": -50.0,
    "left_fisheye_image": 90.0,
    "right_fisheye_image": -90.0,
    "back_fisheye_image": 180.0,
}

# PTZ/compositor/stream settings
PTZ_NAME = "ptz"
COMPOSITOR_SCREEN = "mech"
TARGET_BITRATE = 2_000_000

# Detection thresholds
PERSON_CLASS_ID = 0
MIN_CONFIDENCE = 0.30
MIN_AREA_PX = 600
YOLO_IOU_THRESHOLD = 0.5  # IOU threshold for YOLO NMS

# Loop pacing
LOOP_HZ = 7

# PTZ control policy
PAN_DEADBAND_DEG = 1.0
TILT_DEADBAND_DEG = 1.0
MAX_DEG_PER_STEP = 8.0
DEFAULT_TILT_DEG = -5.0

# YOLO model settings
DEFAULT_YOLO_MODEL = "yolov8x.pt"  # extra large model for accuracy
YOLO_IMG_SIZE = 640
YOLO_DEVICE = "cuda"  # or "cuda" if GPU available

# Connection and retry settings
ROBOT_CONNECT_TIMEOUT_SEC = 10.0
TIME_SYNC_TIMEOUT_SEC = 5.0
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SEC = 2.0

# Logging
LOG_LEVEL = os.getenv("PEOPLE_OBSERVER_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

@dataclass
class ObserverMode:
    """Observation mode: 'bearing' uses HFOV+yaw, 'transform' uses frame transforms."""
    mode: str = "bearing"
    
    def __post_init__(self):
        if self.mode not in ["bearing", "transform"]:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'bearing' or 'transform'.")

@dataclass
class YOLOConfig:
    """YOLO detection model configuration."""
    model_path: str = DEFAULT_YOLO_MODEL
    img_size: int = YOLO_IMG_SIZE
    device: str = YOLO_DEVICE
    iou_threshold: float = YOLO_IOU_THRESHOLD
    person_class_id: int = PERSON_CLASS_ID
    min_confidence: float = MIN_CONFIDENCE
    min_area_px: int = MIN_AREA_PX
    
    def __post_init__(self):
        if self.min_confidence < 0.0 or self.min_confidence > 1.0:
            raise ValueError(f"min_confidence must be in [0.0, 1.0], got {self.min_confidence}")
        if self.min_area_px < 0:
            raise ValueError(f"min_area_px must be non-negative, got {self.min_area_px}")
        if self.img_size <= 0 or self.img_size % 32 != 0:
            raise ValueError(f"img_size must be positive and divisible by 32, got {self.img_size}")

@dataclass
class ConnectionConfig:
    """Robot connection and retry settings."""
    timeout_sec: float = ROBOT_CONNECT_TIMEOUT_SEC
    time_sync_timeout_sec: float = TIME_SYNC_TIMEOUT_SEC
    max_retries: int = MAX_RETRY_ATTEMPTS
    retry_delay_sec: float = RETRY_DELAY_SEC
    
    def __post_init__(self):
        if self.timeout_sec <= 0:
            raise ValueError(f"timeout_sec must be positive, got {self.timeout_sec}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")

@dataclass
class PTZConfig:
    """PTZ camera control settings."""
    name: str = PTZ_NAME
    compositor_screen: str = COMPOSITOR_SCREEN  # Use the module constant
    target_bitrate: int = TARGET_BITRATE
    pan_deadband_deg: float = PAN_DEADBAND_DEG
    tilt_deadband_deg: float = TILT_DEADBAND_DEG
    max_deg_per_step: float = MAX_DEG_PER_STEP
    default_tilt_deg: float = DEFAULT_TILT_DEG
    
    def __post_init__(self):
        if self.target_bitrate <= 0:
            raise ValueError(f"target_bitrate must be positive, got {self.target_bitrate}")
        if self.pan_deadband_deg < 0 or self.tilt_deadband_deg < 0:
            raise ValueError("Deadband values must be non-negative")
        if self.max_deg_per_step <= 0 or self.max_deg_per_step > 180:
            raise ValueError(f"max_deg_per_step must be in (0, 180], got {self.max_deg_per_step}")

@dataclass
class RuntimeConfig:
    """Main runtime configuration aggregating all subsystems."""
    sources: List[str] = field(default_factory=lambda: SURROUND_SOURCES)
    hfov_deg: Dict[str, float] = field(default_factory=lambda: SURROUND_HFOV_DEG)
    yaw_deg: Dict[str, float] = field(default_factory=lambda: SURROUND_YAW_DEG)
    loop_hz: int = LOOP_HZ
    observer_mode: ObserverMode = field(default_factory=ObserverMode)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    ptz: PTZConfig = field(default_factory=PTZConfig)
    log_level: str = LOG_LEVEL
    log_format: str = LOG_FORMAT
    dry_run: bool = False  # Skip PTZ commands, log detections only
    once: bool = False  # Run only one detection cycle
    visualize: bool = False  # Show live detection visualization with OpenCV
    
    def __post_init__(self):
        if self.loop_hz <= 0:
            raise ValueError(f"loop_hz must be positive, got {self.loop_hz}")
        # Validate that all sources have HFOV and yaw entries
        for src in self.sources:
            if src not in self.hfov_deg:
                raise ValueError(f"Source '{src}' missing HFOV definition")
            if src not in self.yaw_deg:
                raise ValueError(f"Source '{src}' missing yaw definition")
    
    @classmethod
    def from_env(cls, **overrides) -> "RuntimeConfig":
        """Create config with environment variable overrides."""
        config = cls()
        # Allow environment overrides for key settings
        if "YOLO_MODEL" in os.environ:
            config.yolo.model_path = os.environ["YOLO_MODEL"]
        if "LOOP_HZ" in os.environ:
            config.loop_hz = int(os.environ["LOOP_HZ"])
        if "MIN_CONFIDENCE" in os.environ:
            config.yolo.min_confidence = float(os.environ["MIN_CONFIDENCE"])
        # Apply any passed overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def validate_camera_source(self, source: str) -> bool:
        """Check if a camera source is valid and configured."""
        return source in self.sources and source in self.hfov_deg and source in self.yaw_deg
