"""Monitor robot state and classify current action for comfort model.

Uses RobotStateClient to poll the robot's velocity, locomotion state, and other
indicators to classify the robot's current action into categories that affect
human comfort:
- "idle": Robot is stationary and not commanding any motion
- "moving": Robot is actively locomoting (walking, turning, etc.)
- "waiting": Robot is stationary but possibly ready/oriented toward person
- "interacting": Robot is close to person and adjusting position/orientation
- "searching": Robot is moving head/body to look for person
- "sit": Robot is sitting (motors powered but not standing)
- "stopped": Robot motors are powered off
"""

import logging
import numpy as np
from typing import Optional
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.api import robot_state_pb2

logger = logging.getLogger(__name__)


class RobotActionMonitor:
    """Monitors robot state and classifies current action."""
    
    # Velocity thresholds for action classification
    LINEAR_VELOCITY_THRESHOLD = 0.05  # m/s - below this is considered stationary
    ANGULAR_VELOCITY_THRESHOLD = 0.1  # rad/s - below this is considered stationary
    INTERACTION_DISTANCE_THRESHOLD = 2.0  # meters - distance for "interacting" classification
    
    def __init__(self, robot):
        """Initialize the action monitor.
        
        Args:
            robot: Authenticated robot object from bosdyn SDK
        """
        self.robot = robot
        self.state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._last_action = "idle"
        logger.info("RobotActionMonitor initialized")
    
    def get_current_action(self, person_distance: Optional[float] = None) -> str:
        """Determine the robot's current action based on state.
        
        Args:
            person_distance: Optional distance to detected person in meters
        
        Returns:
            Action string: "idle", "moving", "waiting", "interacting", "searching", "sit", or "stopped"
        """
        try:
            # Get current robot state
            logger.debug("Polling robot state...")
            state = self.state_client.get_robot_state()
            logger.debug("Robot state received successfully")
            
            # Check power state first
            motor_power_state = state.power_state.motor_power_state
            motor_power_name = robot_state_pb2.PowerState.MotorPowerState.Name(motor_power_state)
            logger.debug(f"Motor power state: {motor_power_name}")
            
            # If motors are off, robot is stopped
            if motor_power_state == robot_state_pb2.PowerState.MOTOR_POWER_STATE_OFF:
                action = "stopped"
                if action != self._last_action:
                    logger.info(f"Robot action changed: {self._last_action} -> {action} (motors powered off)")
                    self._last_action = action
                return action
            
            # Check behavior state for sitting/standing/stepping
            behavior_state = state.behavior_state.state
            behavior_state_name = robot_state_pb2.BehaviorState.State.Name(behavior_state)
            logger.debug(f"Behavior state: {behavior_state_name}")
            
            # Robot is sitting if not in standing/stepping/transition states
            if behavior_state == robot_state_pb2.BehaviorState.STATE_NOT_READY:
                # NOT_READY typically means sitting
                action = "sit"
                if action != self._last_action:
                    logger.info(f"Robot action changed: {self._last_action} -> {action} (behavior_state={behavior_state_name})")
                    self._last_action = action
                return action
            
            # Extract velocity from kinematic state
            if not state.kinematic_state or not state.kinematic_state.velocity_of_body_in_odom:
                logger.warning("No kinematic state or velocity available, defaulting to idle")
                return "idle"
            
            velocity = state.kinematic_state.velocity_of_body_in_odom
            
            # Compute velocity magnitudes
            linear_vel = velocity.linear
            angular_vel = velocity.angular
            
            linear_speed = np.sqrt(linear_vel.x**2 + linear_vel.y**2 + linear_vel.z**2)
            angular_speed = np.sqrt(angular_vel.x**2 + angular_vel.y**2 + angular_vel.z**2)
            
            logger.debug(f"Velocity: linear={linear_speed:.3f} m/s, angular={angular_speed:.3f} rad/s, "
                        f"person_distance={person_distance:.2f}m" if person_distance else 
                        f"Velocity: linear={linear_speed:.3f} m/s, angular={angular_speed:.3f} rad/s")
            
            # Classify based on motion and context
            is_moving = (linear_speed > self.LINEAR_VELOCITY_THRESHOLD or 
                        angular_speed > self.ANGULAR_VELOCITY_THRESHOLD)
            
            logger.debug(f"Is moving: {is_moving} (linear_threshold={self.LINEAR_VELOCITY_THRESHOLD}, "
                        f"angular_threshold={self.ANGULAR_VELOCITY_THRESHOLD})")
            
            if is_moving:
                # Robot is actively moving
                if person_distance is not None and person_distance < self.INTERACTION_DISTANCE_THRESHOLD:
                    # Moving close to person - likely adjusting position for interaction
                    action = "interacting"
                    logger.debug(f"Action: interacting (distance {person_distance:.2f}m < {self.INTERACTION_DISTANCE_THRESHOLD}m)")
                elif angular_speed > linear_speed * 2:
                    # Primarily rotating - likely searching/scanning
                    action = "searching"
                    logger.debug(f"Action: searching (angular {angular_speed:.3f} > linear {linear_speed:.3f} * 2)")
                else:
                    # General locomotion
                    action = "moving"
                    logger.debug("Action: moving (general locomotion)")
            else:
                # Robot is stationary
                if person_distance is not None and person_distance < self.INTERACTION_DISTANCE_THRESHOLD:
                    # Close to person but stationary - ready to interact
                    action = "waiting"
                    logger.debug(f"Action: waiting (stationary, distance {person_distance:.2f}m < {self.INTERACTION_DISTANCE_THRESHOLD}m)")
                else:
                    # Stationary and not close to person
                    action = "idle"
                    logger.debug("Action: idle (stationary)")
            
            # Log significant action changes
            if action != self._last_action:
                logger.info(f"Robot action changed: {self._last_action} -> {action} "
                          f"(behavior={behavior_state_name}, power={motor_power_name}, "
                          f"linear_vel={linear_speed:.3f} m/s, angular_vel={angular_speed:.3f} rad/s, "
                          f"distance={person_distance:.2f}m)" if person_distance else 
                          f"(behavior={behavior_state_name}, power={motor_power_name}, "
                          f"linear_vel={linear_speed:.3f} m/s, angular_vel={angular_speed:.3f} rad/s)")
                self._last_action = action
            
            return action
            
        except Exception as e:
            logger.error(f"Failed to get robot action: {e}", exc_info=True)
            logger.debug(f"Returning last known action: {self._last_action}")
            return self._last_action  # Return last known action on error
    
    def get_detailed_state(self) -> dict:
        """Get detailed robot state information for debugging.
        
        Returns:
            Dictionary with velocity, power state, battery, and other metrics
        """
        try:
            state = self.state_client.get_robot_state()
            
            velocity = state.kinematic_state.velocity_of_body_in_odom
            linear_vel = velocity.linear
            angular_vel = velocity.angular
            
            # Battery state
            battery_pct = state.power_state.locomotion_charge_percentage.value
            shore_power = state.power_state.shore_power_state
            
            # E-Stop state
            estop_states = []
            for estop in state.estop_states:
                estop_states.append({
                    'name': estop.name,
                    'type': robot_state_pb2.EStopState.Type.Name(estop.type),
                    'state': robot_state_pb2.EStopState.State.Name(estop.state)
                })
            
            return {
                'linear_velocity': {
                    'x': linear_vel.x,
                    'y': linear_vel.y,
                    'z': linear_vel.z,
                    'magnitude': np.sqrt(linear_vel.x**2 + linear_vel.y**2 + linear_vel.z**2)
                },
                'angular_velocity': {
                    'x': angular_vel.x,
                    'y': angular_vel.y,
                    'z': angular_vel.z,
                    'magnitude': np.sqrt(angular_vel.x**2 + angular_vel.y**2 + angular_vel.z**2)
                },
                'power_state': robot_state_pb2.PowerState.MotorPowerState.Name(
                    state.power_state.motor_power_state
                ),
                'battery': {
                    'percentage': battery_pct,
                    'shore_power_connected': shore_power == 1,
                    'low_battery': battery_pct < 20.0
                },
                'estop_states': estop_states,
                'action': self._last_action
            }
        except Exception as e:
            logger.error(f"Failed to get detailed state: {e}")
            return {'error': str(e)}
    
    def get_battery_percentage(self) -> float:
        """Get current battery percentage.
        
        Returns:
            Battery percentage (0-100), or 0 on error
        """
        try:
            state = self.state_client.get_robot_state()
            return state.power_state.locomotion_charge_percentage.value
        except Exception as e:
            logger.error(f"Failed to get battery percentage: {e}")
            return 0.0
    
    def is_battery_low(self, threshold: float = 20.0) -> bool:
        """Check if battery is below threshold.
        
        Args:
            threshold: Battery percentage threshold (default 20%)
        
        Returns:
            True if battery below threshold
        """
        return self.get_battery_percentage() < threshold
    
    def get_estop_states(self) -> list:
        """Get all E-Stop states.
        
        Returns:
            List of E-Stop state dictionaries
        """
        try:
            state = self.state_client.get_robot_state()
            estop_states = []
            for estop in state.estop_states:
                estop_states.append({
                    'name': estop.name,
                    'type': robot_state_pb2.EStopState.Type.Name(estop.type),
                    'state': robot_state_pb2.EStopState.State.Name(estop.state),
                    'is_stopped': estop.state != robot_state_pb2.EStopState.STATE_NOT_ESTOPPED
                })
            return estop_states
        except Exception as e:
            logger.error(f"Failed to get E-Stop states: {e}")
            return []

    def get_battery_percentage(self) -> float:
        """Get current battery percentage.
        
        Returns:
            Battery percentage (0-100), or 0 on error
        """
        try:
            state = self.state_client.get_robot_state()
            return state.power_state.locomotion_charge_percentage.value
        except Exception as e:
            logger.error(f"Failed to get battery percentage: {e}")
            return 0.0
    
    def is_battery_low(self, threshold: float = 20.0) -> bool:
        """Check if battery is below threshold.
        
        Args:
            threshold: Battery percentage threshold (default 20%)
        
        Returns:
            True if battery below threshold
        """
        return self.get_battery_percentage() < threshold
