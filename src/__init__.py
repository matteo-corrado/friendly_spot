# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Top-level package initialization for modular friendly_spot architecture organizing
# perception, behavior, robot control, video sources, and visualization into coherent subsystems
# Acknowledgements: Claude for modular architecture design and package structure

"""Friendly Spot Source Code.

Modular architecture for social robot behavior based on human perception.

Modules:
- perception: Person detection, pose, face, emotion, gesture recognition
- behavior: Comfort model and behavior decision logic
- robot: Robot I/O, client management, lease/estop handling
- video: Video source abstraction (webcam, PTZ, WebRTC)
- visualization: Real-time perception overlays
- utils: Shared utilities and helpers
"""

__version__ = '2.0.0'

# Make key components easily importable
from .robot import create_robot, RobotClients
from .perception import PerceptionPipeline, PersonDetection
from .behavior import ComfortModel, BehaviorExecutor, BehaviorLabel
from .video import create_video_source
from .visualization import visualize_pipeline_frame

__all__ = [
    'create_robot',
    'RobotClients',
    'PerceptionPipeline',
    'PersonDetection',
    'ComfortModel',
    'BehaviorExecutor',
    'BehaviorLabel',
    'create_video_source',
    'visualize_pipeline_frame',
]
