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
import math
from typing import Optional

from bosdyn.client import Robot
from bosdyn.client.robot_command import RobotCommandBuilder, blocking_stand, block_for_trajectory_cmd
from bosdyn.client.lease import ResourceAlreadyClaimedError
from bosdyn.client import frame_helpers, math_helpers
from bosdyn.api import world_object_pb2, geometry_pb2
import bosdyn.geometry
import numpy as np

from .planner import BehaviorLabel
from ..robot.io import RobotClients, ManagedLease, ManagedEstop

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
    
    def _get_obstacles_in_path(self, max_distance: float = 2.0) -> list:
        """Get world objects that might be obstacles in the robot's path.
        
        Args:
            max_distance: Maximum distance (meters) to consider obstacles
        
        Returns:
            List of (distance, object) tuples for nearby objects
        """
        try:
            # Get robot state for transforms
            robot_state = self.clients.state.get_robot_state()
            transforms = robot_state.kinematic_state.transforms_snapshot
            
            # List all world objects
            world_objects = self.clients.world_object.list_world_objects().world_objects
            
            if not world_objects:
                logger.debug("No world objects detected")
                return []
            
            obstacles = []
            for obj in world_objects:
                try:
                    # Get object position relative to robot body
                    obj_snapshot = obj.transforms_snapshot
                    if not obj_snapshot or not obj_snapshot.child_to_parent_edge_map:
                        continue
                    
                    # Try to get object frame name (usually the object's name)
                    obj_frame = obj.name if obj.name else str(obj.id)
                    
                    # Get transform from body to object
                    body_tform_obj = frame_helpers.get_a_tform_b(
                        obj_snapshot,
                        frame_helpers.BODY_FRAME_NAME,
                        obj_frame,
                        validate=False
                    )
                    
                    if body_tform_obj:
                        # Calculate distance (x-y plane, ignore height)
                        x = body_tform_obj.x
                        y = body_tform_obj.y
                        distance = (x**2 + y**2)**0.5
                        
                        # Check if in front of robot and within range
                        if x > 0 and distance < max_distance:
                            obstacles.append((distance, obj, x, y))
                            logger.debug(f"Obstacle '{obj.name}' at {distance:.2f}m (x={x:.2f}, y={y:.2f})")
                
                except Exception as e:
                    logger.debug(f"Could not process world object {obj.id}: {e}")
                    continue
            
            # Sort by distance (closest first)
            obstacles.sort(key=lambda o: o[0])
            logger.info(f"Detected {len(obstacles)} obstacles in path")
            return obstacles
            
        except Exception as e:
            logger.warning(f"Failed to get obstacles: {e}")
            return []
    
    def _check_path_clear(self, distance_m: float, bearing_deg: float = 0.0) -> bool:
        """Check if path is clear of obstacles.
        
        Args:
            distance_m: Planned movement distance
            bearing_deg: Direction of movement (0=forward)
        
        Returns:
            True if path appears clear, False if obstacles detected
        """
        obstacles = self._get_obstacles_in_path(max_distance=distance_m + 0.5)
        
        if not obstacles:
            return True
        
        # Check if any obstacles are directly in path
        import math
        bearing_rad = math.radians(bearing_deg)
        
        for dist, obj, x, y in obstacles:
            # Calculate angle to obstacle
            angle_to_obj = math.atan2(y, x)
            angle_diff = abs(angle_to_obj - bearing_rad)
            
            # If obstacle is within 30 degrees of path and closer than planned distance
            if angle_diff < math.radians(30) and dist < distance_m:
                logger.warning(f"Obstacle '{obj.name}' detected in path at {dist:.2f}m")
                return False
        
        return True
    
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
    
    def execute_behavior(self, behavior: BehaviorLabel, perception=None) -> bool:
        """Execute robot command based on behavior label.
        
        Requires lease and E-Stop to already be acquired (via context manager).
        Checks robot power and stand state before movement commands.
        
        Args:
            behavior: Desired behavior from ComfortModel
            perception: Optional PerceptionInput with PTZ bearing for directional movement
        
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
                # Walk to person's location using SE2 trajectory
                if perception and hasattr(perception, 'distance') and perception.distance is not None:
                    success = self._walk_to_person(perception, target_distance_m=1.0, speed=0.8)
                else:
                    success = self._walk_forward(distance_m=0.5, speed=1.0)
            
            elif behavior == BehaviorLabel.GO_CLOSE_SLOWLY:
                # Walk to person's location slowly using SE2 trajectory
                if perception and hasattr(perception, 'distance') and perception.distance is not None:
                    success = self._walk_to_person(perception, target_distance_m=1.5, speed=0.5)
                else:
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
            
            # Check for obstacles in path
            if not self._check_path_clear(distance_m, bearing_deg=0.0):
                logger.warning(f"Path blocked by obstacles - reducing distance")
                distance_m = distance_m * 0.5  # Move half the distance
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
    
    def _walk_to_person(self, perception, target_distance_m: float, speed: float) -> bool:
        """Walk to person's location using SE2 trajectory command.
        
        Computes person's position in vision frame from PTZ angles and depth,
        then commands robot to move to a position target_distance_m away from person.
        
        Args:
            perception: PerceptionInput with distance, ptz_pan, ptz_tilt
            target_distance_m: Desired final distance from person in meters
            speed: Maximum linear velocity in m/s
        
        Returns:
            True if command sent successfully and reached goal
        
        Note: Uses synchro_se2_trajectory_command which blocks until goal reached.
        """
        try:
            # Extract perception data
            distance_to_person = perception.distance
            ptz_pan = getattr(perception, 'ptz_pan', None)
            ptz_tilt = getattr(perception, 'ptz_tilt', None)
            
            if distance_to_person is None or distance_to_person <= 0:
                logger.warning("No valid distance to person - falling back to forward walk")
                return self._walk_forward(distance_m=0.5, speed=speed)
            
            # Get robot state for transforms
            robot_state = self.clients.state.get_robot_state()
            transforms = robot_state.kinematic_state.transforms_snapshot
            
            # Get vision_tform_body (robot's current position in vision frame)
            vision_tform_body = frame_helpers.get_a_tform_b(
                transforms, 
                frame_helpers.VISION_FRAME_NAME, 
                frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME
            )
            
            # Convert PTZ angles to person position in vision frame
            # PTZ pan/tilt give us direction from robot body frame
            # We need to compute: vision_tform_person
            
            if ptz_pan is not None and ptz_tilt is not None:
                # Person is at (distance * cos(tilt), 0, distance * sin(tilt)) in PTZ/body frame
                # Then rotated by pan angle around Z axis
                pan_rad = math.radians(ptz_pan)
                tilt_rad = math.radians(ptz_tilt)
                
                # Position in body frame (X forward, Y left, Z up)
                # Horizontal distance component
                horiz_dist = distance_to_person * math.cos(tilt_rad)
                x_body = horiz_dist * math.cos(pan_rad)  # Forward component
                y_body = horiz_dist * math.sin(pan_rad)  # Left component
                z_body = distance_to_person * math.sin(tilt_rad)  # Vertical component
                
                logger.debug(f"Person position in body frame: x={x_body:.2f}, y={y_body:.2f}, z={z_body:.2f}")
            else:
                # No PTZ data - assume person is straight ahead
                logger.warning("No PTZ angles available - assuming person straight ahead")
                x_body = distance_to_person
                y_body = 0.0
                z_body = 0.0
            
            # Transform person position to vision frame
            # vision_tform_person = vision_tform_body * body_tform_person
            body_tform_person = math_helpers.SE3Pose(x=x_body, y=y_body, z=z_body, rot=math_helpers.Quat())
            vision_tform_person = vision_tform_body * body_tform_person
            
            logger.info(f"Person location in vision frame: ({vision_tform_person.x:.2f}, {vision_tform_person.y:.2f}, {vision_tform_person.z:.2f})")
            
            # Compute where robot should stand (similar to fetch.py compute_stand_location_and_yaw)
            # Vector from person to robot using math_helpers
            robot_rt_person_ewrt_vision = [
                vision_tform_body.x - vision_tform_person.x,
                vision_tform_body.y - vision_tform_person.y,
                vision_tform_body.z - vision_tform_person.z
            ]
            
            # Compute unit vector
            dist = math_helpers.Vec3(
                robot_rt_person_ewrt_vision[0],
                robot_rt_person_ewrt_vision[1],
                robot_rt_person_ewrt_vision[2]
            ).length()
            
            if dist < 0.01:
                logger.warning("Robot and person positions too close - using default direction")
                robot_rt_person_ewrt_vision_hat = [1.0, 0.0, 0.0]
            else:
                robot_rt_person_ewrt_vision_hat = [
                    robot_rt_person_ewrt_vision[0] / dist,
                    robot_rt_person_ewrt_vision[1] / dist,
                    robot_rt_person_ewrt_vision[2] / dist
                ]
            
            # Target position: start at person, back up target_distance_m along unit vector
            goal_x = vision_tform_person.x + robot_rt_person_ewrt_vision_hat[0] * target_distance_m
            goal_y = vision_tform_person.y + robot_rt_person_ewrt_vision_hat[1] * target_distance_m
            
            # Compute heading to face person (from fetch.py pattern)
            # X axis should point from robot to person
            xhat = [-robot_rt_person_ewrt_vision_hat[0], 
                    -robot_rt_person_ewrt_vision_hat[1], 
                    -robot_rt_person_ewrt_vision_hat[2]]
            zhat = [0.0, 0.0, 1.0]
            yhat = math_helpers.Vec3(zhat[0], zhat[1], zhat[2]).cross(
                math_helpers.Vec3(xhat[0], xhat[1], xhat[2])
            )
            
            # Build rotation matrix and get yaw
            mat = np.matrix([xhat, [yhat.x, yhat.y, yhat.z], zhat]).transpose()
            goal_heading = math_helpers.Quat.from_matrix(mat).to_yaw()
            
            logger.info(f"Walking to position ({goal_x:.2f}, {goal_y:.2f}) with heading {math.degrees(goal_heading):.1f}°")
            logger.info(f"Target distance from person: {target_distance_m:.1f}m, speed: {speed:.1f}m/s")
            
            # Check if path is clear (use bearing to person for obstacle check)
            bearing_to_person = math.degrees(math.atan2(y_body, x_body))
            move_distance = dist - target_distance_m
            if move_distance < 0:
                logger.warning(f"Already closer than target distance ({dist:.2f}m < {target_distance_m:.1f}m)")
                return True
            
            if not self._check_path_clear(move_distance, bearing_deg=bearing_to_person):
                logger.warning("Path to person blocked - adjusting target distance")
                target_distance_m = target_distance_m * 1.5  # Stay farther away
                goal_x = vision_tform_person.x + robot_rt_person_ewrt_vision_hat[0] * target_distance_m
                goal_y = vision_tform_person.y + robot_rt_person_ewrt_vision_hat[1] * target_distance_m
            
            # Build SE2 trajectory command
            se2_pose = geometry_pb2.SE2Pose(
                position=geometry_pb2.Vec2(x=goal_x, y=goal_y),
                angle=goal_heading
            )
            
            # Set velocity limits
            max_vel_linear = geometry_pb2.Vec2(x=speed, y=speed)
            max_vel_se2 = geometry_pb2.SE2Velocity(linear=max_vel_linear, angular=0.8)
            vel_limit = geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2)
            params = RobotCommandBuilder.mobility_params()
            params.vel_limit.CopyFrom(vel_limit)
            
            # Send trajectory command
            move_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
                se2_pose,
                frame_name=frame_helpers.VISION_FRAME_NAME,
                params=params
            )
            
            # Command timeout based on distance and speed
            timeout = max(move_distance / speed * 1.5, 5.0)
            cmd_id = self.clients.command.robot_command(
                command=move_cmd,
                end_time_secs=time.time() + timeout
            )
            
            logger.debug(f"Waiting for trajectory to complete (timeout={timeout:.1f}s)...")
            
            # Wait for trajectory to complete
            success = block_for_trajectory_cmd(
                self.clients.command, 
                cmd_id, 
                timeout_sec=timeout,
                feedback_interval_secs=0.2
            )
            
            if success:
                logger.info("Successfully reached person's location")
            else:
                logger.warning("Trajectory did not complete within timeout")
            
            return success
            
        except Exception as e:
            logger.error(f"Walk to person failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            try:
                stop_cmd = RobotCommandBuilder.stop_command()
                self.clients.command.robot_command(command=stop_cmd)
            except:
                pass
            return False
    
    def _sit(self) -> bool:
        """Stationary pose with head tilted down (looking at ground).
        
        Instead of sitting, adjusts body pitch to look downward.
        Uses synchro_stand_command with pitch adjustment.
        """
        try:
            logger.info("Adjusting body pose: looking down (submissive/calm pose)")
            
            # Create body orientation with negative pitch (look down)
            # pitch: negative = look down, positive = look up
            # Typical range: -0.5 to 0.5 radians (~28 degrees)
            footprint_R_body = bosdyn.geometry.EulerZXY(yaw=0.0, roll=0.0, pitch=-0.4)
            
            cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body)
            self.clients.command.robot_command(cmd)
            
            logger.info("Body pose adjusted: head down")
            time.sleep(1.0)  # Allow pose to stabilize
            return True
            
        except Exception as e:
            logger.error(f"Body pose adjustment (look down) failed: {e}")
            return False
    
    def _stand(self) -> bool:
        """Stand with head tilted up (alert/observant pose).
        
        Adjusts body pitch to look upward while remaining standing.
        Uses synchro_stand_command with pitch adjustment.
        """
        try:
            logger.info("Adjusting body pose: looking up (alert/observant pose)")
            
            # Create body orientation with positive pitch (look up)
            # pitch: positive = look up, negative = look down
            # Typical range: -0.5 to 0.5 radians (~28 degrees)
            footprint_R_body = bosdyn.geometry.EulerZXY(yaw=0.0, roll=0.0, pitch=0.4)
            
            cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body)
            self.clients.command.robot_command(cmd)
            
            logger.info("Body pose adjusted: head up")
            time.sleep(1.0)  # Allow pose to stabilize
            return True
            
        except Exception as e:
            logger.error(f"Body pose adjustment (look up) failed: {e}")
            # Fallback to neutral stand if pose adjustment fails
            try:
                logger.warning("Falling back to neutral stand")
                blocking_stand(self.clients.command, timeout_sec=10)
                return True
            except Exception as e2:
                logger.error(f"Fallback stand also failed: {e2}")
                return False
