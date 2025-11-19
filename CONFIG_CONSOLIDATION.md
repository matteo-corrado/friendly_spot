# Configuration Consolidation Analysis

## Current State (November 19, 2025)

### Configuration Locations

We currently have configuration spread across multiple files:

1. **`people_observer/config.py`** (Primary config)
   - Camera sources (surround fisheye cameras)
   - YOLO detection parameters
   - PTZ control settings (movement, deadbands)
   - Connection/retry settings
   - Dataclasses: `YOLOConfig`, `ConnectionConfig`, `PTZConfig`, `RuntimeConfig`
   - **NEW**: Added Spot CAM service/source names

2. **`video_sources.py`** (Video source config)
   - Spot CAM image service name (`SPOT_CAM_IMAGE_SERVICE = 'spot-cam-image'`)
   - PTZ source name (`SPOT_CAM_PTZ_SOURCE = 'ptz'`)
   - 360/ring/stream source names
   - Surround camera sources (duplicated from config.py)
   - Image quality settings

3. **`friendly_spot_main.py`** (Pipeline config)
   - Default loop rate (5 Hz)
   - Command-line argument defaults
   - No explicit constants section

4. **`perception_pipeline.py`** (Perception config)
   - MediaPipe model complexity settings
   - Face detection confidence thresholds
   - Embedded in code, not externalized

## Recent Discovery: Spot CAM Architecture

Investigation with `test_list_image_sources.py` revealed:

- **Spot CAM uses separate service**: `'spot-cam-image'` (not `'image'`)
- **PTZ camera source**: `'ptz'` (resolution: 1920x1080)
- **Other sources**: `'pano'` (360°), `'c0'-'c4'` (ring), `'stream'` (compositor)

This required updating both:
- `video_sources.py`: Global constants for video source selection
- `people_observer/config.py`: PTZConfig with service/source fields

## Consolidation Recommendation: Keep Current Structure ✅

**Decision: Do NOT consolidate into single global config**

### Rationale

1. **Modular Architecture**
   - `people_observer/` is a standalone module (could be separate package)
   - `video_sources.py` is a reusable abstraction layer
   - Each module should be independently configurable

2. **Different Configuration Purposes**
   - `people_observer/config.py`: Detection/tracking algorithm parameters
   - `video_sources.py`: Video I/O and hardware interface settings
   - `friendly_spot_main.py`: Pipeline orchestration and CLI defaults

3. **Clear Ownership**
   - Observer team owns `people_observer/config.py`
   - Video team owns `video_sources.py`
   - Integration team owns `friendly_spot_main.py`

4. **Avoid Circular Dependencies**
   - `people_observer/` imports nothing from parent directory
   - `video_sources.py` is independent of observer
   - Single global config would create tight coupling

### What We DID Fix ✅

1. **Added Spot CAM discovery to config.py**:
   ```python
   SPOT_CAM_IMAGE_SERVICE = 'spot-cam-image'
   SURROUND_IMAGE_SERVICE = 'image'
   PTZ_SOURCE_NAME = 'ptz'
   PTZ_NAME = 'mech'
   ```

2. **Updated PTZConfig dataclass**:
   ```python
   @dataclass
   class PTZConfig:
       image_service: str = SPOT_CAM_IMAGE_SERVICE  # NEW
       source_name: str = PTZ_SOURCE_NAME           # NEW
       name: str = PTZ_NAME
       # ... rest of fields
   ```

3. **Documented constants in both files**:
   - `video_sources.py`: Global config section at top with comments
   - `config.py`: Added service/source documentation

### Configuration Access Pattern

```python
# For video source selection (video_sources.py)
from video_sources import create_video_source, SPOT_CAM_PTZ_SOURCE
source = create_video_source('imageclient', robot=robot, source_name=SPOT_CAM_PTZ_SOURCE)

# For detection/tracking (people_observer/config.py)
from people_observer.config import RuntimeConfig
cfg = RuntimeConfig()
ptz_service = cfg.ptz.image_service  # 'spot-cam-image'
ptz_source = cfg.ptz.source_name     # 'ptz'

# For pipeline orchestration (friendly_spot_main.py)
# Use command-line args (argparse) with sensible defaults
```

## Future Improvements (If Needed)

If configuration management becomes unwieldy, consider:

1. **Environment Variables**
   - `SPOT_CAM_SERVICE`, `PTZ_SOURCE`, `YOLO_MODEL_PATH`
   - Already supported in `RuntimeConfig.from_env()`

2. **Config File (YAML/TOML)**
   - For deployment-specific overrides
   - Keep code defaults, allow file-based overrides
   - Example: `friendly_spot_config.yaml`

3. **Dataclass Composition**
   - Use nested dataclasses for logical grouping
   - Already done: `RuntimeConfig` contains `PTZConfig`, `YOLOConfig`

## Current Status: ✅ COMPLETE

- [x] Discovered Spot CAM service/source architecture
- [x] Updated `video_sources.py` with global constants
- [x] Updated `people_observer/config.py` with service/source fields
- [x] Added PTZConfig.image_service and PTZConfig.source_name
- [x] Documented configuration in both files
- [x] Fixed `friendly_spot_main.py` to use cfg.ptz.name
- [x] Analyzed consolidation trade-offs
- [x] Decided to keep modular structure

## Recommendation

**Keep configuration distributed** across modules with clear ownership and documentation. Current structure supports:
- Module independence
- Clear separation of concerns
- No circular dependencies
- Easy testing and reuse

Only consolidate if you find yourself copy-pasting the same constants across 3+ files.
