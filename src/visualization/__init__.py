"""Visualization and overlay module.

Provides real-time visualization of perception results on video frames.

Key Components:
- visualize_pipeline_frame: Main visualization function
- draw_perception_results: Draw overlays on frame
- close_all_windows: Cleanup function

Usage:
    >>> from src.visualization import visualize_pipeline_frame
    >>> visualize_pipeline_frame(
    ...     frame, perception, behavior, comfort_score,
    ...     save_path='output.jpg'
    ... )
"""

from .overlay import (
    visualize_pipeline_frame,
    draw_perception_results,
    close_all_windows,
)

__all__ = [
    'visualize_pipeline_frame',
    'draw_perception_results',
    'close_all_windows',
]
