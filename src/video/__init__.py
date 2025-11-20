# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Video module package initialization exposing unified video source interfaces
# Acknowledgements: Claude for module organization

"""Video source abstraction module.

Provides unified interface for different video sources (webcam, PTZ, WebRTC).

Key Components:
- VideoSource: Abstract base class for video sources
- WebcamSource: Local USB/built-in camera
- SpotPTZImageClient: Spot PTZ via ImageClient (recommended)
- SpotPTZWebRTC: Spot PTZ via WebRTC streaming
- PtzStream: WebRTC streaming helper
- create_video_source: Factory function

Usage:
    >>> from src.video import create_video_source
    >>> # Webcam for development
    >>> source = create_video_source('webcam', device=0)
    >>> # Robot PTZ camera
    >>> source = create_video_source('imageclient', robot=robot)
"""

from .sources import (
    VideoSource,
    WebcamSource,
    SpotPTZImageClient,
    SpotPTZWebRTC,
    create_video_source,
)

__all__ = [
    'VideoSource',
    'WebcamSource',
    'SpotPTZImageClient',
    'SpotPTZWebRTC',
    'create_video_source',
]
