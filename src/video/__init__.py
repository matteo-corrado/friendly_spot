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
