# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Perception module package initialization exposing pipeline, detection types, and configuration
# Acknowledgements: Claude for module organization

"""Perception pipeline module.

Integrates pose, face, emotion, gesture detection with person tracking.

Key Components:
- PerceptionPipeline: Main pipeline orchestrator
- PersonDetection: Unified detection data structure
- YoloDetector: Person detection using YOLO
- Geometric helpers: Camera transforms and PTZ calculations
- Config: Configuration constants and settings

Usage:
    >>> from src.perception import PerceptionPipeline, PersonDetection
    >>> pipeline = PerceptionPipeline()
    >>> perception = pipeline.read_perception(person_detection)
"""

from .detection_types import PersonDetection, validate_depth_against_heuristic, estimate_distance_from_bbox
from .pipeline import PerceptionPipeline
from .yolo_detector import YoloDetector, Detection
from .cameras import fetch_image_sources
from .geometry import pixel_to_ptz_angles_simple, pixel_to_ptz_angles_transform

__all__ = [
    'PersonDetection',
    'PerceptionPipeline',
    'YoloDetector',
    'Detection',
    'fetch_image_sources',
    'pixel_to_ptz_angles_simple',
    'pixel_to_ptz_angles_transform',
    'validate_depth_against_heuristic',
    'estimate_distance_from_bbox',
]
