# Friendly Spot Codebase Refactor Plan

**Date**: November 19, 2025  
**Branch**: matteodev_pipelinefull  
**Status**: In Progress

## Overview

Reorganizing codebase into logical module structure with proper documentation standards.

## New Directory Structure

```
friendly_spot/
├── src/                          # Source code modules
│   ├── perception/               # Perception pipeline
│   ├── behavior/                 # Behavior planning and execution
│   ├── robot/                    # Robot I/O and control
│   ├── video/                    # Video source abstraction
│   ├── visualization/            # Visualization system
│   └── utils/                    # Shared utilities
├── tests/                        # Test scripts
├── data/                         # Data files
│   ├── models/                   # YOLO and other models
│   └── outputs/                  # Generated outputs
├── docs/                         # Documentation markdown files
├── deprecated/                   # Old/redundant files
├── friendly_spot_main.py         # Main entry point
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## File Relocation Mapping

### Perception Module (`src/perception/`)
- `run_pipeline.py` → `src/perception/pipeline.py`
- `detection_types.py` → `src/perception/detection_types.py`
- `people_observer/detection.py` → `src/perception/yolo_detector.py`
- `people_observer/cameras.py` → `src/perception/cameras.py`
- `people_observer/geometry.py` → `src/perception/geometry.py`
- `people_observer/tracker.py` → `src/perception/tracker.py`
- `people_observer/config.py` → `src/perception/config.py`
- `EmotionRecognition/*` → `src/perception/emotion/` (review and integrate)

### Behavior Module (`src/behavior/`)
- `behavior_planner.py` → `src/behavior/planner.py`
- `behavior_executor.py` → `src/behavior/executor.py`

### Robot Module (`src/robot/`)
- `robot_io.py` → `src/robot/io.py`
- `robot_action_monitor.py` → `src/robot/action_monitor.py`
- `observer_bridge.py` → `src/robot/observer_bridge.py`
- `people_observer/ptz_control.py` → `src/robot/ptz_control.py`

### Video Module (`src/video/`)
- `video_sources.py` → `src/video/sources.py`
- `people_observer/ptz_stream.py` → `src/video/ptz_stream.py`
- `people_observer/ptz_webrtc_client.py` → `src/video/webrtc_client.py`

### Visualization Module (`src/visualization/`)
- `unified_visualization.py` → `src/visualization/overlay.py`
- `people_observer/visualization.py` → `src/visualization/helpers.py`

### Utils Module (`src/utils/`)
- (Create utility helpers as needed during refactor)

### Tests (`tests/`)
- `test_list_image_sources.py` → `tests/test_image_sources.py`
- `test_ptz_convention.py` → `tests/test_ptz_convention.py`
- `test_ptz_image_client.py` → `tests/test_ptz_image_client.py`
- `run_behavior_demo.py` → `tests/demo_behavior.py`
- `people_observer/test_*.py` → `tests/`
- `print_camera_transforms.py` → `tests/debug_transforms.py`

### Data (`data/`)
- `people_observer/*.pt` (YOLO models) → `data/models/`
- `test_output/` → `data/outputs/test_output/`
- `images/` → `data/images/`
- `dataset/` → `data/dataset/`
- `Records/` → `data/records/`

### Documentation (`docs/`)
- `CONFIG_CONSOLIDATION.md` → `docs/config_consolidation.md`
- `INTEGRATED_SYSTEM_GUIDE.md` → `docs/system_guide.md`
- `PTZ_CAMERA_FIX_GUIDE.md` → `docs/ptz_troubleshooting.md`
- `ROBOT_MOVEMENT_IMPLEMENTATION_PLAN.md` → `docs/movement_implementation.md`
- `QUICK_START.md` → Keep in root, update
- `people_observer/*.md` → `docs/people_observer/`

### Deprecated (`deprecated/`)
- `Facial Recognition/` → `deprecated/facial_recognition/`
- `people_observer/geometry_old.py` → `deprecated/geometry_old.py`
- `people_observer/app.py` → Review - may be obsolete
- Redundant requirements files → `deprecated/requirements/`

## Implementation Steps

### Phase 1: Structure Setup ✅
- [x] Create new directory structure
- [x] Document relocation plan

### Phase 2: Copy and Update Core Files
- [ ] Copy files to new locations (preserve originals initially)
- [ ] Update imports in copied files
- [ ] Create `__init__.py` for each module
- [ ] Test imports work correctly

### Phase 3: Update Main Entry Point
- [ ] Update `friendly_spot_main.py` imports
- [ ] Test full pipeline execution
- [ ] Verify all functionality works

### Phase 4: Documentation
- [ ] Standardize docstrings across all modules
- [ ] Create module-level READMEs
- [ ] Create comprehensive main README
- [ ] Move old docs to `docs/`

### Phase 5: Cleanup
- [ ] Delete original files after verification
- [ ] Archive deprecated files
- [ ] Update .gitignore
- [ ] Clean up redundant files

### Phase 6: Testing
- [ ] Run all test scripts
- [ ] Verify robot connection and execution
- [ ] Test webcam fallback mode
- [ ] Document any breaking changes

## Import Path Strategy

Using relative imports within modules and absolute imports from `src`:

```python
# Old style (flat)
from robot_io import RobotClients
from behavior_planner import ComfortModel

# New style (structured)
from src.robot.io import RobotClients
from src.behavior.planner import ComfortModel

# Within module (relative)
from .detection_types import PersonDetection
from ..robot.io import RobotClients
```

**PYTHONPATH approach**: Add `friendly_spot/` to PYTHONPATH, then imports work naturally.

## Breaking Changes

### Import Paths
- All imports must be updated to new module structure
- External scripts importing these modules will need updates

### File Locations
- Models now in `data/models/`
- Outputs now in `data/outputs/`
- Config files consolidated

### Configuration
- Plan to consolidate scattered config into single source
- Document migration path from old config locations

## Rollback Plan

- Keep original files until Phase 5
- Git branch allows easy rollback
- Test thoroughly before deleting originals
- Document any issues encountered

## Notes

- Maintain backward compatibility where possible
- Update all README files to reflect new structure
- Ensure all tests pass before considering complete
- Update .gitignore to handle new structure
