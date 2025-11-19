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
    
    Acquires lease and E-Stop at initialization and holds them for entire session.
    Use as context manager for proper cleanup.
    
    Usage:
        with BehaviorExecutor(robot) as executor:
            executor.execute_behavior(BehaviorLabel.GO_CLOSE)
    
    Args:
        robot: Authenticated Robot instance (from robot_io.create_robot())
        force_take_lease: If True, forcefully takes lease from tablet/other clients.
                         Use when lease conflicts prevent operation.
    """
    
    def __init__(self, robot: Robot, force_take_lease: bool = False):
        self.robot = robot
        self.force_take_lease = force_take_lease
        
        # Initialize lazy client container
        self.clients = RobotClients(robot)
        
        # Track last executed behavior to avoid redundant commands
        self.last_behavior: Optional[BehaviorLabel] = None
        
        # Lease and E-Stop context managers (initialized in __enter__)
        self._lease_manager = None
        self._estop_manager = None
        self._lease_active = False
        
        if force_take_lease:
            logger.warning("BehaviorExecutor initialized with force_take_lease=True - will take lease from current holders")
        else:
            logger.info("BehaviorExecutor initialized")
    
    def __enter__(self):
        """Acquire lease and E-Stop for the session."""
        logger.info("Acquiring robot control (lease + E-Stop)...")
        
        # Acquire lease
        self._lease_manager = ManagedLease(self.robot, force_take=self.force_take_lease)
        self._lease_manager.__enter__()
        
        # Acquire E-Stop (skip if force_take_lease and motors already on)
        self._estop_manager = ManagedEstop(self.robot, name="BehaviorExecutor", 
                                          skip_if_active=self.force_take_lease)
        self._estop_manager.__enter__()
        
        self._lease_active = True
        logger.info("Robot control acquired - ready to execute behaviors")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lease and E-Stop."""
        logger.info("Releasing robot control...")
        self._lease_active = False
        
        if self._estop_manager:
            try:
                self._estop_manager.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning(f"Error releasing E-Stop: {e}")
        
        if self._lease_manager:
            try:
                self._lease_manager.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning(f"Error releasing lease: {e}")
        
        logger.info("Robot control released")
    
    def get_battery_state(self) -> dict:
        """Get current battery state and metrics.
        
        Returns:
            Dictionary with battery percentage, charge status, voltage, and temperature
        """
        try:
            state = self.clients.state.get_robot_state()
            battery = state.power_state.locomotion_charge_percentage
            shore_power = state.power_state.shore_power_state
            
            return {
                'percentage': battery.value,
                'charging': shore_power == 1,
                'low_battery': battery.value < 20.0
            }
        except Exception as e:
            logger.error(f"Failed to get battery state: {e}")
            return {'percentage': 0, 'charging': False, 'low_battery': True}
    
    def get_system_metrics(self) -> dict:
        """Get robot system metrics (temperature, errors, etc.).
        
        Returns:
            Dictionary with system health metrics
        """
        try:
            metrics = self.clients.state.get_robot_metrics()
            return {
                'timestamp': metrics.timestamp.seconds,
                'metrics': str(metrics)  # Full metrics proto for detailed inspection
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {'error': str(e)}
    
    def check_battery_before_action(self, action_name: str) -> bool:
        """Check battery level before executing movement actions.
        
        Args:
            action_name: Name of the action being attempted
        
        Returns:
            True if battery is sufficient, False if too low
        """
        battery_state = self.get_battery_state()
        battery_pct = battery_state.get('percentage', 0)
        
        if battery_pct < 10.0:
            logger.error(f"Battery critically low ({battery_pct:.1f}%) - refusing {action_name}")
            return False
        elif battery_pct < 20.0:
            logger.warning(f"Battery low ({battery_pct:.1f}%) - {action_name} may fail")
        
        return True
    
    def execute_behavior(self, behavior: BehaviorLabel) -> bool:
        """Execute robot command based on behavior label.
        
        Requires lease and E-Stop to already be acquired (via context manager).
        Checks robot power and stand state before movement commands.
        
        Args:
            behavior: Desired behavior from ComfortModel
        
        Returns:
            True if command executed successfully, False otherwise
        
        Note: Skips execution if behavior matches last_behavior to avoid
              redundant commands (e.g., repeated SIT commands).
        """
        # Check lease is active
        if not self._lease_active:
            logger.error("Cannot execute behavior - lease not active. Use BehaviorExecutor as context manager.")
            return False
        
        # Skip if same as last behavior (avoid redundant commands)
        if behavior == self.last_behavior:
            logger.debug(f"Skipping redundant behavior: {behavior.value}")
            return True
        
        logger.info(f"Executing behavior: {behavior.value}")
        
        try:
            # Check battery before movement commands
            movement_behaviors = [
                BehaviorLabel.GO_CLOSE,
                BehaviorLabel.GO_CLOSE_SLOWLY,
                BehaviorLabel.BACK_AWAY,
                BehaviorLabel.BACK_AWAY_SLOWLY
            ]
            if behavior in movement_behaviors:
                if not self.check_battery_before_action(behavior.value):
                    logger.error("Aborting behavior due to low battery")
                    return False
            
            # Check robot is powered on before movement commands
            if behavior in [BehaviorLabel.GO_CLOSE, BehaviorLabel.GO_CLOSE_SLOWLY,
                          BehaviorLabel.BACK_AWAY, BehaviorLabel.BACK_AWAY_SLOWLY,
                          BehaviorLabel.STAY]:
                if not self.robot.is_powered_on():
                    logger.warning("Robot not powered on - standing up first...")
                    if not self._stand():
                        logger.error("Failed to power on/stand robot")
                        return False
            
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
            logger.error("This shouldn't happen if using context manager correctly")
            return False
        except Exception as e:
            logger.error(f"Failed to execute behavior {behavior.value}: {e}")
            return False
    
    def _walk_forward(self, distance_m: float, speed: float) -> bool:
        """Walk forward using time-based velocity control.
        
        Sends repeated short-lived velocity commands (0.6s each) until estimated
        distance traveled. Follows SDK safety pattern: commands auto-expire if
        client fails, robot stops automatically.
        
        Args:
            distance_m: Distance to walk in meters
            speed: Forward velocity in m/s (recommended: 0.3-0.8)
        
        Returns:
            True if command sent successfully
        
        Note: Actual distance may vary ±20% due to slip, obstacles, etc.
              This is open-loop control (no odometry feedback).
        """
        try:
            # SDK safety pattern constants (from wasd.py)
            VELOCITY_CMD_DURATION = 0.6  # seconds (command auto-expires)
            COMMAND_INTERVAL = 0.5  # seconds (resend before expiration)
            
            # Calculate total time needed
            total_time = distance_m / speed
            
            logger.info(f"Walking forward {distance_m}m at {speed}m/s (estimated {total_time:.1f}s)")
            
            command_client = self.clients.command
            
            # Send repeated velocity commands until time elapsed
            start_time = time.time()
            while (time.time() - start_time) < total_time:
                # Build velocity command (v_x positive = forward)
                cmd = RobotCommandBuilder.synchro_velocity_command(
                    v_x=speed,   # forward velocity
                    v_y=0.0,     # no lateral movement
                    v_rot=0.0    # no rotation
                )
                
                # Send command with auto-expire timeout
                end_time = time.time() + VELOCITY_CMD_DURATION
                command_client.robot_command(command=cmd, end_time_secs=end_time)
                
                # Wait before next command (allows overlap for smooth motion)
                time.sleep(COMMAND_INTERVAL)
            
            # Stop after movement complete
            stop_cmd = RobotCommandBuilder.stop_command()
            command_client.robot_command(command=stop_cmd)
            
            logger.info("Forward walk complete")
            return True
            
        except Exception as e:
            logger.error(f"Walk forward failed: {e}")
            # Try to stop robot as safety measure
            try:
                stop_cmd = RobotCommandBuilder.stop_command()
                self.clients.command.robot_command(command=stop_cmd)
            except:
                pass  # Best effort stop
            return False
    
    def _walk_backward(self, distance_m: float, speed: float) -> bool:
        """Walk backward using time-based velocity control.
        
        Uses negative v_x velocity to move backward. Follows same pattern as
        _walk_forward with repeated short-lived commands for safety.
        
        Args:
            distance_m: Distance to walk backward in meters (positive value)
            speed: Backward velocity magnitude in m/s (positive value, 0.2-0.5 recommended)
        
        Returns:
            True if command sent successfully
        
        Note: Backward movement typically slower/more cautious than forward.
              Actual distance may vary ±20% (open-loop control).
        """
        try:
            VELOCITY_CMD_DURATION = 0.6
            COMMAND_INTERVAL = 0.5
            
            total_time = distance_m / speed
            
            logger.info(f"Walking backward {distance_m}m at {speed}m/s (estimated {total_time:.1f}s)")
            
            command_client = self.clients.command
            
            start_time = time.time()
            while (time.time() - start_time) < total_time:
                # Negative v_x = backward movement
                cmd = RobotCommandBuilder.synchro_velocity_command(
                    v_x=-speed,  # negative = backward
                    v_y=0.0,
                    v_rot=0.0
                )
                
                end_time = time.time() + VELOCITY_CMD_DURATION
                command_client.robot_command(command=cmd, end_time_secs=end_time)
                
                time.sleep(COMMAND_INTERVAL)
            
            # Stop
            stop_cmd = RobotCommandBuilder.stop_command()
            command_client.robot_command(command=stop_cmd)
            
            logger.info("Backward walk complete")
            return True
            
        except Exception as e:
            logger.error(f"Walk backward failed: {e}")
            try:
                stop_cmd = RobotCommandBuilder.stop_command()
                self.clients.command.robot_command(command=stop_cmd)
            except:
                pass
            return False
    
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
