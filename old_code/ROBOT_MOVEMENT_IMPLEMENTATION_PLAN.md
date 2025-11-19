# Robot Movement Implementation Plan

## Executive Summary
Plan for implementing `_walk_forward()` and `_walk_backward()` methods in `behavior_executor.py` based on Boston Dynamics Spot SDK patterns discovered in `wasd.py` example.

## Key SDK Discoveries

### 1. Velocity Command Pattern (from wasd.py)
```python
# Constants
VELOCITY_BASE_SPEED = 0.5  # m/s
VELOCITY_BASE_ANGULAR = 0.8  # rad/sec
VELOCITY_CMD_DURATION = 0.6  # seconds
COMMAND_INPUT_RATE = 0.1  # seconds between commands

# Usage pattern
def _velocity_cmd_helper(self, desc='', v_x=0.0, v_y=0.0, v_rot=0.0):
    self._start_robot_command(
        desc, 
        RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot),
        end_time_secs=time.time() + VELOCITY_CMD_DURATION
    )

# Sending command
def _start_robot_command(self, desc, command_proto, end_time_secs=None):
    def _start_command():
        self._robot_command_client.robot_command(
            command=command_proto,
            end_time_secs=end_time_secs
        )
    self._try_grpc(desc, _start_command)
```

### 2. Movement Examples from wasd.py
```python
def _move_forward(self):
    self._velocity_cmd_helper('move_forward', v_x=VELOCITY_BASE_SPEED)

def _move_backward(self):
    self._velocity_cmd_helper('move_backward', v_x=-VELOCITY_BASE_SPEED)

def _strafe_left(self):
    self._velocity_cmd_helper('strafe_left', v_y=VELOCITY_BASE_SPEED)

def _strafe_right(self):
    self._velocity_cmd_helper('strafe_right', v_y=-VELOCITY_BASE_SPEED)

def _turn_left(self):
    self._velocity_cmd_helper('turn_left', v_rot=VELOCITY_BASE_ANGULAR)

def _turn_right(self):
    self._velocity_cmd_helper('turn_right', v_rot=-VELOCITY_BASE_ANGULAR)

def _stop(self):
    self._start_robot_command('stop', RobotCommandBuilder.stop_command())
```

### 3. SDK Safety Recommendations
From robot services documentation:
- **"Send short-lived commands and continuously resend them"** - robot stops if client crashes
- For longer commands: cache `CommandResponse` id, poll `CommandFeedbackRequest` for status
- Velocity commands use `end_time_secs` parameter for automatic timeout

## Implementation Strategy

### Option A: Simple Time-Based (Recommended for v1)
**Pros:**
- Simple, predictable code
- Follows wasd.py pattern exactly
- Inherently safe (short-lived commands with automatic timeout)

**Cons:**
- Less accurate (slip, obstacles affect actual distance)
- No feedback on actual distance traveled

**Implementation:**
```python
def _walk_forward(self, distance_m: float = 1.0, speed: float = 0.5):
    """Walk forward for specified distance at given speed.
    
    Uses time-based open-loop control: sends repeated velocity commands
    until calculated time elapsed. Follows SDK safety pattern of 
    short-lived commands (0.6s each) that auto-expire.
    
    Args:
        distance_m: Target distance in meters (default 1.0)
        speed: Forward velocity in m/s (default 0.5, max ~1.0)
    
    Note: Actual distance may vary due to slip, obstacles, etc.
          Consider this a best-effort command, not precision navigation.
    """
    VELOCITY_CMD_DURATION = 0.6  # seconds (command auto-expires)
    COMMAND_INTERVAL = 0.5  # seconds (resend rate, < cmd_duration for overlap)
    
    # Calculate total time needed
    total_time = distance_m / speed
    
    # Get command client
    command_client = self.clients.robot_command
    
    logger.info(f"Walking forward {distance_m}m at {speed}m/s (estimated {total_time:.1f}s)")
    
    # Send repeated velocity commands until time elapsed
    start_time = time.time()
    while (time.time() - start_time) < total_time:
        # Build velocity command
        cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=speed,  # forward velocity
            v_y=0.0,    # no lateral movement
            v_rot=0.0   # no rotation
        )
        
        # Send command with timeout
        end_time = time.time() + VELOCITY_CMD_DURATION
        command_client.robot_command(command=cmd, end_time_secs=end_time)
        
        # Wait before next command (allow some command overlap for smoothness)
        time.sleep(COMMAND_INTERVAL)
    
    # Stop after movement complete
    stop_cmd = RobotCommandBuilder.stop_command()
    command_client.robot_command(command=stop_cmd)
    
    logger.info("Forward walk complete")
```

### Option B: Odometry Feedback Loop (v2 - More Accurate)
**Pros:**
- Accurate distance measurement
- Can detect if robot gets stuck or blocked
- Handles slip/obstacles gracefully

**Cons:**
- More complex code
- Requires polling robot_state for odometry
- Need to handle frame transforms (ODOM_FRAME_NAME)

**Implementation Sketch:**
```python
def _walk_forward_precise(self, distance_m: float = 1.0, speed: float = 0.5):
    """Walk forward for EXACT distance using odometry feedback.
    
    Uses closed-loop control: polls robot odometry, sends velocity commands
    until actual distance traveled matches target.
    """
    from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_se2_a_tform_b
    
    # Get starting position from robot state
    state_client = self.clients.robot_state
    start_state = state_client.get_robot_state()
    start_pos = get_se2_a_tform_b(
        start_state.kinematic_state.transforms_snapshot,
        ODOM_FRAME_NAME,
        'body'
    )
    start_x = start_pos.position.x
    start_y = start_pos.position.y
    
    command_client = self.clients.robot_command
    VELOCITY_CMD_DURATION = 0.6
    COMMAND_INTERVAL = 0.3
    
    distance_traveled = 0.0
    while distance_traveled < distance_m:
        # Send velocity command
        cmd = RobotCommandBuilder.synchro_velocity_command(v_x=speed, v_y=0.0, v_rot=0.0)
        command_client.robot_command(command=cmd, end_time_secs=time.time() + VELOCITY_CMD_DURATION)
        
        time.sleep(COMMAND_INTERVAL)
        
        # Poll current position
        current_state = state_client.get_robot_state()
        current_pos = get_se2_a_tform_b(
            current_state.kinematic_state.transforms_snapshot,
            ODOM_FRAME_NAME,
            'body'
        )
        
        # Calculate distance traveled
        dx = current_pos.position.x - start_x
        dy = current_pos.position.y - start_y
        distance_traveled = math.sqrt(dx**2 + dy**2)
        
        logger.debug(f"Distance: {distance_traveled:.2f}/{distance_m:.2f}m")
    
    # Stop
    stop_cmd = RobotCommandBuilder.stop_command()
    command_client.robot_command(command=stop_cmd)
```

### Option C: SE2 Trajectory Command (SDK Goal-Based)
**Pros:**
- SDK handles path planning and execution
- Single command (no loop needed)
- Can specify exact goal pose (position + heading)

**Cons:**
- More complex (need to specify goal in ODOM_FRAME)
- May be overkill for simple forward/backward
- Harder to interrupt mid-trajectory

**Implementation Reference (from wasd.py):**
```python
def _return_to_origin(self):
    self._start_robot_command(
        'fwd_and_rotate',
        RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=0.0, 
            goal_y=0.0, 
            goal_heading=0.0, 
            frame_name=ODOM_FRAME_NAME, 
            params=None,
            body_height=0.0, 
            locomotion_hint=spot_command_pb2.HINT_SPEED_SELECT_TROT
        ),
        end_time_secs=time.time() + 20
    )
```

## Recommended Implementation Path

### Phase 1: Time-Based (Immediate)
1. Implement `_walk_forward()` and `_walk_backward()` using **Option A** (time-based)
2. Use constants: `speed=0.5 m/s`, `VELOCITY_CMD_DURATION=0.6s`, `COMMAND_INTERVAL=0.5s`
3. Add `stop_command()` at end of movement
4. Test with person-following scenarios

### Phase 2: Odometry Enhancement (Future)
1. Add odometry feedback version as `_walk_forward_precise()`
2. Compare accuracy between time-based and odometry-based
3. Consider switching default if precision is critical

### Phase 3: Advanced Features (Optional)
1. Add obstacle detection during movement (use local_grid service?)
2. Add rotation commands (`_turn_left()`, `_turn_right()`) for heading adjustment
3. Consider trajectory commands for more complex paths

## Integration Points

### behavior_executor.py Current State
```python
def _walk_forward(self, distance_m: float = 1.0, speed: float = 0.5):
    """TODO: Implement forward locomotion command.
    
    Should use RobotCommandBuilder.synchro_velocity_command()
    to move robot forward by distance_m at given speed.
    Consider using feedback loop for accurate distance control.
    """
    logger.warning("_walk_forward not yet implemented")
    # TODO: Implementation here
```

### Required Imports (already present)
```python
import time
from bosdyn.client.robot_command import RobotCommandBuilder
```

### Client Access (via robot_io.RobotClients)
```python
self.clients.robot_command  # RobotCommandClient
self.clients.robot_state     # RobotStateClient (for odometry)
```

### Context Managers (already in execute_behavior)
```python
with ManagedLease(self.clients.lease) as lease:
    with ManagedEstop(self.clients.estop) as estop:
        # Commands here have lease + estop protection
        self._walk_forward(distance_m, speed)
```

## Speed and Distance Guidelines

### Safe Speed Ranges
- **Slow/Careful**: 0.2-0.3 m/s (around people, tight spaces)
- **Normal**: 0.5 m/s (default, comfortable for following)
- **Fast**: 0.8-1.0 m/s (max safe speed indoors)
- **Max**: ~1.5 m/s (outdoor use, requires clear space)

### Distance Parameters for Friendly Spot
Based on person-following use case:
- **Approach**: 0.3-0.5m (close distance when person far)
- **Maintain**: 0.1-0.2m (small adjustments to hold distance)
- **Retreat**: 0.3-0.5m (back away if too close)

### Command Timing
- `VELOCITY_CMD_DURATION`: 0.6s (command expires after this)
- `COMMAND_INTERVAL`: 0.5s (send new command before old expires)
- Overlap ensures smooth motion without stops

## Error Handling

### Expected Issues
1. **Lease conflicts**: Already handled in `execute_behavior()` with `ResourceAlreadyClaimedError`
2. **Robot not powered**: Check `robot.is_powered_on()` before movement
3. **E-Stop active**: ManagedEstop context should prevent this
4. **Command rejected**: Catch `RpcError`, `ResponseError` from SDK
5. **Robot stuck**: Odometry version can detect (distance not increasing)

### Recommended Error Strategy
```python
try:
    self._walk_forward(distance_m, speed)
except (RpcError, ResponseError) as e:
    logger.error(f"Movement command failed: {e}")
    # Send stop command as safety measure
    try:
        stop_cmd = RobotCommandBuilder.stop_command()
        self.clients.robot_command.robot_command(command=stop_cmd)
    except:
        pass  # Best effort stop
    return False
```

## Testing Plan

### Test 1: Basic Movement
1. Ensure Spot powered on and standing
2. Call `_walk_forward(distance_m=1.0, speed=0.3)` (slow, safe)
3. Observe: Robot should move ~1m forward, then stop
4. Verify: No errors, smooth motion, clean stop

### Test 2: Distance Accuracy
1. Mark starting position
2. Call `_walk_forward(distance_m=2.0, speed=0.5)`
3. Measure actual distance traveled
4. Acceptable: ±20% error for time-based version

### Test 3: Backward Movement
1. Call `_walk_backward(distance_m=0.5, speed=0.3)`
2. Verify: Robot moves backward, stops cleanly

### Test 4: Integration with Friendly Spot
1. Run full `friendly_spot_main.py` with implemented movements
2. Test person-following scenario
3. Verify: Robot maintains distance, adjusts position smoothly
4. Check: No PTZ conflicts (async control should prevent)

## Additional SDK Services to Explore

### world-object Service
- **Use**: Fiducial detection (AprilTags), can provide more accurate person tracking
- **Integration**: Could enhance perception with fixed reference points

### local-grid Service
- **Use**: Terrain awareness, obstacle detection
- **Integration**: Could make movement safer by avoiding obstacles during locomotion

### manipulation-api (if arm equipped)
- **Use**: Arm control for "friendly" gestures
- **Integration**: Could add waving, pointing behaviors to enhance friendliness

## References
- **SDK Docs**: `spot-sdk/docs/concepts/robot_services.md` (robot-command service)
- **Example**: `spot-sdk/python/examples/wasd/wasd.py` (velocity commands, lines 428-452)
- **Example**: `spot-sdk/python/examples/hello_spot/hello_spot.py` (trajectory commands, stand/sit)
- **Client API**: `spot-sdk/python/bosdyn-client/src/bosdyn/client/robot_command.py` (blocking helpers)

## Implementation Priority

**Priority 1 - Immediate (Required for person-following):**
- [x] `_walk_forward()` - Time-based implementation ✅ COMPLETE
- [x] `_walk_backward()` - Time-based implementation ✅ COMPLETE
- [x] Error handling and stop command ✅ COMPLETE
- [ ] Integration testing with person-following

**Priority 2 - Enhancement (Improves accuracy):**
- [ ] Odometry feedback version
- [ ] Distance accuracy testing/comparison
- [ ] Stuck detection (robot not moving despite command)

**Priority 3 - Advanced (Optional features):**
- [ ] Rotation commands for heading adjustment
- [ ] Obstacle avoidance integration
- [ ] Trajectory command version for complex paths
- [ ] world-object service integration

## Implementation Summary (COMPLETED)

**Date Implemented:** November 19, 2025

**What Was Built:**
- Time-based velocity control for forward/backward movement
- Follows wasd.py SDK pattern: repeated 0.6s commands with 0.5s intervals
- Automatic stop command after movement completes
- Error handling with best-effort emergency stop
- Open-loop control (±20% accuracy expected)

**Key Implementation Details:**
```python
# Pattern: Send velocity commands in loop until time elapsed
VELOCITY_CMD_DURATION = 0.6  # Command auto-expires (safety)
COMMAND_INTERVAL = 0.5       # Resend before expiration (smooth motion)
total_time = distance_m / speed

while (time.time() - start_time) < total_time:
    cmd = RobotCommandBuilder.synchro_velocity_command(v_x=speed, v_y=0, v_rot=0)
    command_client.robot_command(cmd, end_time_secs=time.time() + 0.6)
    time.sleep(0.5)

stop_cmd = RobotCommandBuilder.stop_command()
command_client.robot_command(stop_cmd)
```

**Testing Status:**
- [ ] Basic movement test (powered robot, safe space)
- [ ] Distance accuracy measurement
- [ ] Integration with person-following in friendly_spot_main.py

## Notes
- Current `behavior_executor.py` already has proper structure with context managers
- All required clients are available via `self.clients` (robot_io module)
- Lease and E-Stop management is automatic (no manual handling needed)
- Consider adding `max_distance` parameter (safety limit: e.g., 3.0m max per command)
- RobotCommandBuilder found in `spot-sdk/python/bosdyn-client/src/bosdyn/client/robot_command.py`
  - `synchro_velocity_command()` at line 900
  - `stop_command()` at line 651
  - `synchro_sit_command()` at line 983
  - `synchro_stand_command()` at line 947
