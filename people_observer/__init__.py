# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/18/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Package initialization for people_observer - real-time person detection and tracking
# with PTZ camera control, depth integration, and WebRTC streaming for facial recognition
# Acknowledgements: Boston Dynamics Spot SDK for robot API foundation

"""People observer package: detect humans in surround fisheye streams and aim Spot CAM PTZ.

Modules:
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
