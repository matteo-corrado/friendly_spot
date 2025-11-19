# Tests

Test suite for Friendly Spot pipeline components.

## Overview

This directory contains unit tests, integration tests, and validation scripts for the Friendly Spot perception and behavior pipeline.

## Structure

```
tests/
├── test_imports.py           # Validate module imports
├── test_image_sources.py     # Camera source tests
├── test_ptz_convention.py    # PTZ angle convention tests
├── test_yolo_pipeline.py     # YOLO detection pipeline tests
└── test_unified_viz.py       # Visualization tests
```

## Running Tests

### All Tests

```powershell
# Run all tests
python -m pytest tests/

# With verbose output
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

### Individual Test Files

```powershell
# Import validation
python tests/test_imports.py

# Image source tests
python tests/test_image_sources.py

# PTZ convention tests
python tests/test_ptz_convention.py
```

### Robot Tests (Require Hardware)

Some tests require connection to Spot robot:

```powershell
# Set robot hostname
$env:SPOT_HOSTNAME = "192.168.80.3"

# Run robot-dependent tests
python -m pytest tests/ -m robot
```

## Test Categories

### Unit Tests
Test individual functions/classes in isolation:
- `test_imports.py`: Module loading and exports
- `test_ptz_convention.py`: Angle calculations

### Integration Tests
Test component interactions:
- `test_yolo_pipeline.py`: Detection → tracking → visualization
- `test_image_sources.py`: Camera → decoder → display

### Hardware Tests
Require robot connection (marked with `@pytest.mark.robot`):
- PTZ command execution
- Camera frame acquisition
- Behavior command execution

## Writing Tests

### Test Structure

```python
import pytest
from src.perception import YoloDetector

class TestYoloDetector:
    def test_model_loading(self):
        """Test YOLO model loads successfully."""
        detector = YoloDetector(model_path="yolov8n.pt")
        assert detector.model is not None
    
    def test_person_detection(self):
        """Test person detection on sample image."""
        detector = YoloDetector()
        frame = cv2.imread("tests/fixtures/person.jpg")
        detections = detector.predict_batch([frame])[0]
        assert len(detections) > 0
        assert all(det.conf > 0.4 for det in detections)
```

### Fixtures

Common test data in `tests/fixtures/`:

```
fixtures/
├── person.jpg             # Sample image with person
├── multi_person.jpg       # Multiple people
├── empty_room.jpg         # No detections expected
└── depth_image.npy        # Sample depth map
```

### Mocking Robot Connection

Use mocks for robot-dependent tests:

```python
from unittest.mock import Mock, patch

@patch('src.robot.io.create_robot')
def test_behavior_executor(mock_create_robot):
    """Test executor without real robot."""
    mock_robot = Mock()
    mock_create_robot.return_value = mock_robot
    
    # Test behavior logic...
```

## Test Coverage

Target coverage goals:
- **Core modules** (perception, behavior, robot): >80%
- **Video/visualization**: >60%
- **Overall**: >70%

Check coverage:

```powershell
python -m pytest tests/ --cov=src --cov-report=term-missing
```

## Continuous Integration

Tests run automatically on:
- Pre-commit hooks (fast tests only)
- Pull requests (all tests except hardware)
- Nightly builds (including hardware tests)

## Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'src'`  
**Solution**: Run from workspace root or set `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "$PWD"
python tests/test_imports.py
```

### Missing Test Fixtures
**Problem**: `FileNotFoundError: tests/fixtures/person.jpg`  
**Solution**: Download test fixtures:

```powershell
# Download from project assets
curl -L https://example.com/test_fixtures.zip -o fixtures.zip
Expand-Archive fixtures.zip tests/fixtures/
```

### Slow Tests
**Problem**: Tests take too long  
**Solution**: Run fast tests only:

```powershell
python -m pytest tests/ -m "not slow and not robot"
```

## Dependencies

- `pytest` >= 7.4.0: Test framework
- `pytest-cov` >= 4.1.0: Coverage reports
- `pytest-mock` >= 3.11.0: Mocking utilities

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
