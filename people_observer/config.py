# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Centralized configuration management with constants and dataclasses for camera sources,
# YOLO detection parameters, PTZ control settings, depth processing, and rotation angles
# Acknowledgements: Boston Dynamics SDK documentation for camera rotation angles and PTZ parameters,
# Ultralytics YOLO documentation for model and detection threshold settings

"""Configuration for the people observer.

Camera intrinsics (focal length, distortion coefficients) are now queried dynamically
from the Spot SDK's ImageSource proto at runtime. This supports both:
- Kannala-Brandt fisheye model (5 surround cameras with k1-k4 distortion)
- Pinhole model (PTZ, hand camera)

The values below serve as fallbacks only if intrinsics are unavailable.
Prefer SDK-provided intrinsics via cameras.fetch_image_sources() for accuracy.
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

# Fallback HFOV (used only if SDK intrinsics unavailable)
# Actual HFOV calculated from focal length via cameras.calculate_hfov_from_intrinsics()
FALLBACK_HFOV_DEG = 133.0  # Empirical fisheye HFOV fallback
SURROUND_HFOV_DEG: Dict[str, float] = {
    "frontleft_fisheye_image": 133.0,
    "frontright_fisheye_image": 133.0,
    "left_fisheye_image": 133.0,
    "right_fisheye_image": 133.0,
    "back_fisheye_image": 133.0,
}

# Camera yaw offsets in robot frame (degrees, clockwise from front)
# These are geometric constants based on Spot's camera mounting positions
# Could be computed from frame transforms, but hardcoded for simplicity
CAM_YAW_DEG: Dict[str, float] = {
    "frontleft_fisheye_image": 50.0,
    "frontright_fisheye_image": -50.0,
    "left_fisheye_image": 90.0,
    "right_fisheye_image": -90.0,
    "back_fisheye_image": 180.0,
}

# Physical camera rotation angles (degrees, counterclockwise)
# Spot's fisheye cameras are physically rotated relative to their mounting frames.
# These rotation angles from the SDK are critical for accurate pixel to ray transforms.
# Source: boston-dynamics/spot-sdk examples (get_image.py, stitch_front_images.py)
ROTATION_ANGLE: Dict[str, float] = {
    "frontleft_fisheye_image": -78.0,
    "frontright_fisheye_image": -102.0,
    "left_fisheye_image": 0.0,
    "right_fisheye_image": 180.0,
    "back_fisheye_image": 0.0,
}

# PTZ/compositor/stream settings
# Spot CAM uses separate 'spot-cam-image' service (not standard 'image' service)
# Discovery from test_list_image_sources.py showed available sources:
# - Service: 'spot-cam-image'
#   Sources: 'ptz' (1920x1080), 'pano' (360°), 'c0'-'c4' (ring cameras), 
#            'stream' (compositor), 'projected-ring', 'tiled-ring'

# Image service names
SPOT_CAM_IMAGE_SERVICE = 'spot-cam-image'  # Spot CAM cameras (PTZ, 360, ring)
SURROUND_IMAGE_SERVICE = 'image'            # Standard fisheye cameras

# PTZ source configuration
PTZ_SOURCE_NAME = 'ptz'                     # PTZ camera source in spot-cam-image service
PTZ_NAME = "mech"                           # PTZ device name for control ('mech' or 'digi')
COMPOSITOR_SCREEN = "mech_full"             # Full mechanical PTZ view in compositor
TARGET_BITRATE = 2000000                    # Target bitrate for streaming

# Additional Spot CAM sources (for reference)
PANO_SOURCE_NAME = 'pano'                   # 360-degree panoramic camera
STREAM_SOURCE_NAME = 'stream'               # Compositor stream (multi-camera view)
RING_CAMERA_SOURCES = ['c0', 'c1', 'c2', 'c3', 'c4']  # Individual ring cameras

# PTZ hardware mounting offset (degrees)
# The mechanical PTZ is physically mounted with a 35° offset to the right of the body frame.
# When we command pan=0°, the PTZ actually points 35° to the right of forward.
# To point forward, we must command pan=325° (360° - 35°).
PTZ_OFFSET_DEG = 35.0

# Loop pacing
LOOP_HZ = 7

# PTZ control policy
PAN_DEADBAND_DEG = 1.0
TILT_DEADBAND_DEG = 1.0
MAX_DEG_PER_STEP = 8.0
DEFAULT_TILT_DEG = -5.0
DEFAULT_ZOOM = 1.0  # Zoom range [1.0, 30.0], 1.0 = no zoom
TRANSFORM_MODE = "transform"  # or "bearing"

# YOLO model settings
DEFAULT_YOLO_MODELNAME = "yolo11m-seg.pt"  # medium model for accuracy
# Models should be in people_observer/ directory (same as this config file)
YOLO_MODELS_DIR = Path(__file__).parent
DEFAULT_YOLO_MODEL = str(YOLO_MODELS_DIR / DEFAULT_YOLO_MODELNAME)  # Uses defined path
YOLO_IMG_SIZE = 640  # YOLO input size (images auto-resized to this for inference)
                      # Note: PTZ camera is 1920x1080, surround cameras are 640x480
                      # YOLO automatically handles resizing internally
YOLO_DEVICE = "cuda"
YOLO_HALF = True if YOLO_DEVICE == "cuda" else False  # Use half-precision FP16 for inference on CUDA
YOLO_VERBOSE = False  # Set to True for detailed model inference logging
DEFAULT_INCLUDE_DEPTH = True  # Use depth data if available

# Detection thresholds
PERSON_CLASS_ID = 0
MIN_CONFIDENCE = 0.40
MIN_AREA_PX = 600
YOLO_IOU_THRESHOLD = 0.5  # IOU threshold for YOLO NMS

# Connection and retry settings
ROBOT_CONNECT_TIMEOUT_SEC = 10.0
TIME_SYNC_TIMEOUT_SEC = 5.0
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SEC = 2.0

# Logging
LOG_LEVEL = os.getenv("PEOPLE_OBSERVER_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

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
    half: bool = YOLO_HALF
    verbose: bool = YOLO_VERBOSE
    
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
    # Service and source configuration
    image_service: str = SPOT_CAM_IMAGE_SERVICE  # Image service name ('spot-cam-image')
    source_name: str = PTZ_SOURCE_NAME           # Camera source name ('ptz')
    
    # PTZ control settings
    name: str = PTZ_NAME                         # PTZ device name ('mech' or 'digi')
    compositor_screen: str = COMPOSITOR_SCREEN   # Compositor screen name
    target_bitrate: int = TARGET_BITRATE         # Streaming bitrate
    
    # Movement control
    pan_deadband_deg: float = PAN_DEADBAND_DEG
    tilt_deadband_deg: float = TILT_DEADBAND_DEG
    max_deg_per_step: float = MAX_DEG_PER_STEP
    default_tilt_deg: float = DEFAULT_TILT_DEG
    default_zoom: float = DEFAULT_ZOOM
    
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
    yaw_deg: Dict[str, float] = field(default_factory=lambda: CAM_YAW_DEG)
    loop_hz: int = LOOP_HZ
    observer_mode: str = TRANSFORM_MODE  # 'transform' or 'bearing'
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    ptz: PTZConfig = field(default_factory=PTZConfig)
    log_level: str = LOG_LEVEL
    log_format: str = LOG_FORMAT
    dry_run: bool = False  # Skip PTZ commands, log detections only
    once: bool = False  # Run only one detection cycle
    exit_on_detection: bool = False  # Exit after successfully detecting and commanding PTZ to a person
    visualize: bool = False  # Show live detection visualization with OpenCV
    save_images: Optional[str] = None  # Directory path to save annotated frames (None = disabled)
    
    def __post_init__(self):
        if self.loop_hz <= 0:
            raise ValueError(f"loop_hz must be positive, got {self.loop_hz}")
        if self.observer_mode not in ["bearing", "transform"]:
            raise ValueError(f"Invalid observer_mode '{self.observer_mode}'. Must be 'bearing' or 'transform'.")
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
