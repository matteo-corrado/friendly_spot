"""Robot control and I/O module.

Provides robot connection, client management, lease handling, and E-Stop functionality.

Key Components:
- RobotClients: Unified client manager for all Boston Dynamics SDK clients
- ManagedLease: Context manager for lease acquisition and release
- ManagedEstop: Context manager for E-Stop configuration
- create_robot: Helper to create and authenticate robot connection
- PTZ control: PTZ camera positioning and tracking

Usage:
    >>> from src.robot.io import create_robot, RobotClients
    >>> robot = create_robot(hostname='192.168.80.3')
    >>> clients = RobotClients(robot)
    >>> print(clients.state.get_robot_state())
"""

from .io import (
    RobotClients,
    ManagedLease,
    ManagedEstop,
    create_robot
)

from .action_monitor import RobotActionMonitor
from .observer_bridge import ObserverBridge, ObserverConfig

__all__ = [
    'RobotClients',
    'ManagedLease', 
    'ManagedEstop',
    'create_robot',
    'RobotActionMonitor',
    'ObserverBridge',
    'ObserverConfig',
]
