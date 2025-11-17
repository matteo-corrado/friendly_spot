"""People observer package: detect humans in surround fisheye streams and aim Spot CAM PTZ.

Modules
- app: CLI entrypoint
- io_robot: SDK setup and clients
- cameras: image capture utilities for surround cameras
- detection: YOLO detector wrapper and detection dataclass
- geometry: bearing-only mapping and transform-based pan/tilt math
- ptz_control: PTZ command helpers with smoothing
- tracker: main orchestrator (frames -> detections -> target -> PTZ)
- config: constants and defaults (camera lists, loop params)

Auth is provided via venv.
"""
