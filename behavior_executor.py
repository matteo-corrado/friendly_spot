# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/18/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Behavior executor translating high-level behavior decisions into Spot robot commands.
# Maps BehaviorLabel enum from behavior_planner to SDK RobotCommand API calls.
# Acknowledgements: Boston Dynamics Spot SDK robot_command examples for command patterns

"""Behavior command executor for Spot robot.

Translates BehaviorLabel decisions from ComfortModel into actual robot commands
using Boston Dynamics SDK. Uses robot_io module for unified client management.

TODO sections mark where full command implementation is needed.
"""

import time
import logging
from typing import Optional

from bosdyn.client import Robot
from bosdyn.client.robot_command import RobotCommandBuilder, blocking_stand
from bosdyn.client.lease import ResourceAlreadyClaimedError

from behavior_planner import BehaviorLabel
from robot_io import RobotClients, ManagedLease, ManagedEstop

logger = logging.getLogger(__name__)


class BehaviorExecutor:
    """Execute robot behaviors based on comfort model decisions.
    
    Uses robot_io context managers for automatic lease and E-Stop management.
    Clients are lazily initialized via RobotClients container.
    
    Args:
        robot: Authenticated Robot instance (from robot_io.create_robot())
    """
    
    def __init__(self, robot: Robot):
        self.robot = robot
        
        # Initialize lazy client container
        self.clients = RobotClients(robot)
        
        # Track last executed behavior to avoid redundant commands
        self.last_behavior: Optional[BehaviorLabel] = None
        
        logger.info("BehaviorExecutor initialized")
    
    def execute_behavior(self, behavior: BehaviorLabel) -> bool:
        """Execute robot command based on behavior label.
        
        Uses context managers to automatically handle lease acquisition/return
        and E-Stop registration/deregistration for each command.
        
        Args:
            behavior: Desired behavior from ComfortModel
        
        Returns:
            True if command executed successfully, False otherwise
        
        Note: Skips execution if behavior matches last_behavior to avoid
              redundant commands (e.g., repeated SIT commands).
        """
        # Skip if same as last behavior (avoid redundant commands)
        if behavior == self.last_behavior:
            logger.debug(f"Skipping redundant behavior: {behavior.value}")
            return True
        
        logger.info(f"Executing behavior: {behavior.value}")
        
        try:
            # Context managers automatically handle lease and E-Stop
            with ManagedLease(self.robot), ManagedEstop(self.robot, name="BehaviorExecutor"):
                success = False
                
                if behavior == BehaviorLabel.GO_CLOSE:
                    success = self._walk_forward(distance_m=0.5, speed=1.0)
                
                elif behavior == BehaviorLabel.GO_CLOSE_SLOWLY:
                    success = self._walk_forward(distance_m=0.3, speed=0.5)
                
                elif behavior == BehaviorLabel.BACK_AWAY:
                    success = self._walk_backward(distance_m=0.5, speed=1.0)
                
                elif behavior == BehaviorLabel.BACK_AWAY_SLOWLY:
                    success = self._walk_backward(distance_m=0.3, speed=0.5)
                
                elif behavior == BehaviorLabel.SIT:
                    success = self._sit()
                
                elif behavior == BehaviorLabel.STAY:
                    success = self._stand()
                
                else:
                    logger.warning(f"Unknown behavior: {behavior}")
                    return False
                
                if success:
                    self.last_behavior = behavior
                
                return success
            
        except ResourceAlreadyClaimedError as e:
            logger.error(f"Lease already claimed by another client: {e}")
            logger.error("Run with --no-execute to disable behavior execution, or ensure no other clients are using the robot")
            logger.error("You may need to manually return the lease using the robot's tablet or admin console")
            return False
        except Exception as e:
            logger.error(f"Failed to execute behavior {behavior.value}: {e}")
            return False
    
    def _walk_forward(self, distance_m: float, speed: float) -> bool:
        """Walk forward at specified speed.
        
        TODO: Implement full locomotion command with distance control.
        Current implementation is a stub placeholder.
        
        Args:
            distance_m: Distance to walk in meters
            speed: Forward velocity in m/s
        
        Returns:
            True if command sent successfully
        """
        # TODO: Implement using RobotCommandBuilder.synchro_velocity_command()
        # Duration should be calculated as distance_m / speed
        # Use blocking command or async command with feedback
        
        logger.info(f"TODO: Walk forward {distance_m}m at {speed} m/s")
        
        # Placeholder: just stand for now
        return self._stand()
    
    def _walk_backward(self, distance_m: float, speed: float) -> bool:
        """Walk backward at specified speed.
        
        TODO: Implement full locomotion command with distance control.
        
        Args:
            distance_m: Distance to walk backward in meters (positive value)
            speed: Backward velocity in m/s (positive value)
        
        Returns:
            True if command sent successfully
        """
        # TODO: Same as _walk_forward but with negative velocity
        logger.info(f"TODO: Walk backward {distance_m}m at {speed} m/s")
        return self._stand()
    
    def _sit(self) -> bool:
        """Command robot to sit.
        
        TODO: Verify sit command works correctly and add error handling.
        """
        try:
            logger.info("Commanding robot to sit")
            cmd = RobotCommandBuilder.synchro_sit_command()
            self.clients.command.robot_command(cmd)
            # TODO: Add blocking wait or feedback check to verify sit completed
            time.sleep(2.0)  # Placeholder wait
            return True
        except Exception as e:
            logger.error(f"Sit command failed: {e}")
            return False
    
    def _stand(self) -> bool:
        """Command robot to stand.
        
        Uses blocking_stand() helper for reliable standing.
        """
        try:
            logger.info("Commanding robot to stand")
            blocking_stand(self.clients.command, timeout_sec=10)
            return True
        except Exception as e:
            logger.error(f"Stand command failed: {e}")
            return False
