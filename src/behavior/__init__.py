# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Behavior module package initialization exposing comfort model and behavior planning utilities
# Acknowledgements: Claude for module organization

"""Behavior planning and execution module.

Translates perception data into comfort scores and robot behaviors.

Key Components:
- ComfortModel: Computes comfort scores from perception inputs
- BehaviorPlanner: Maps comfort to behavior decisions
- BehaviorExecutor: Executes robot commands based on behavior labels
- BehaviorLabel: Enum of possible behaviors (GO_CLOSE, BACK_AWAY, etc.)

Usage:
    >>> from src.behavior import ComfortModel, BehaviorExecutor, BehaviorLabel
    >>> model = ComfortModel()
    >>> comfort = model.compute_comfort(perception)
    >>> behavior = model.decide_behavior(comfort)
"""

from .planner import (
    ComfortModel,
    BehaviorLabel,
    PerceptionInput
)

from .executor import BehaviorExecutor

__all__ = [
    'ComfortModel',
    'BehaviorLabel',
    'PerceptionInput',
    'BehaviorExecutor',
]
