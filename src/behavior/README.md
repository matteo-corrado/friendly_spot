# Behavior Module

Comfort-based behavior planning and robot command execution.

## Overview

The behavior module implements a proximity-aware "comfort model" that decides how Spot should respond to detected persons based on distance and social interaction rules. It translates high-level behaviors (GO_CLOSE, MAINTAIN_DISTANCE, etc.) into Boston Dynamics SDK robot commands.

## Components

### Behavior Planning
- **`planner.py`**: Comfort model decision logic
  - `BehaviorLabel`: Enum defining all possible behaviors (GO_CLOSE, MAINTAIN_DISTANCE, GO_AWAY, EXPLORE, STOP)
  - `ComfortModel`: State machine that maps person distance → behavior
  - Distance thresholds: comfortable (1.5-2.5m), too close (<1.5m), too far (>2.5m)
  - Debouncing to prevent rapid behavior oscillation

### Command Execution
- **`executor.py`**: Robot command implementation
  - `BehaviorExecutor`: Context manager for lease/estop management and command execution
  - Implements each BehaviorLabel as SDK RobotCommand:
    - GO_CLOSE: SE2 trajectory to approach person
    - MAINTAIN_DISTANCE: Stand in place
    - GO_AWAY: Backup trajectory
    - EXPLORE: Rotation scan
    - STOP: Stand and hold
  - Uses vision frame for stable world coordinates
  - Proper blocking for trajectory completion

## Usage

### Basic Behavior Loop

```python
from src.robot import create_robot, RobotClients
from src.behavior import ComfortModel, BehaviorExecutor
from src.perception import Detection

# Setup
robot = create_robot("192.168.80.3", register_spot_cam=True)

# Initialize comfort model
comfort = ComfortModel(
    comfortable_distance=2.0,  # Target distance in meters
    too_close_threshold=1.2,   # Backup if closer
    too_far_threshold=3.5      # Approach if farther
)

# Execute behaviors
with BehaviorExecutor(robot) as executor:
    while True:
        # Get detection from perception module
        detection = get_current_target_person()
        
        # Plan behavior
        behavior = comfort.update(detection)
        
        # Execute
        executor.execute_behavior(behavior)
```

### Custom Behavior Extension

Add new behaviors to the system:

```python
# In planner.py, extend BehaviorLabel enum:
class BehaviorLabel(Enum):
    GO_CLOSE = "go_close"
    MAINTAIN_DISTANCE = "maintain_distance"
    GO_AWAY = "go_away"
    EXPLORE = "explore"
    STOP = "stop"
    WAVE = "wave"  # NEW: Wave gesture

# In planner.py, add decision logic:
def update(self, detection: Optional[Detection]) -> BehaviorLabel:
    if some_condition:
        return BehaviorLabel.WAVE
    # ... existing logic

# In executor.py, implement command:
def execute_behavior(self, behavior: BehaviorLabel):
    if behavior == BehaviorLabel.WAVE:
        self._wave_gesture()
    # ... existing cases

def _wave_gesture(self):
    """Wave arm at person."""
    # Use ArmCommand to wave
    arm_command = RobotCommandBuilder.arm_pose_command(...)
    self.clients.command.robot_command(arm_command)
```

### Adjusting Comfort Zones

Tune distance thresholds for different interaction styles:

```python
# Friendly/approachable robot
comfort = ComfortModel(
    comfortable_distance=1.5,  # Get closer
    too_close_threshold=0.8,   # Allow very close approach
    too_far_threshold=2.5
)

# Cautious/respectful robot
comfort = ComfortModel(
    comfortable_distance=3.0,  # Keep more distance
    too_close_threshold=2.0,
    too_far_threshold=5.0
)
```

### Force Control in Multi-Client Environment

When other clients (tablet, SDK scripts) hold the lease:

```python
# Take lease from tablet/other holders
with BehaviorExecutor(robot, force_take_lease=True) as executor:
    executor.execute_behavior(BehaviorLabel.GO_CLOSE)
```

## Comfort Model Theory

The comfort model implements proxemics (social distance theory):

### Distance Zones
1. **Intimate Zone** (<1.2m): Too close, robot backs away
2. **Personal Zone** (1.2-2.5m): Comfortable, robot maintains position
3. **Social Zone** (2.5-4m): Acceptable but distant, robot approaches
4. **Public Zone** (>4m): Too far, robot explores or approaches

### State Machine

```
     NO PERSON                PERSON DETECTED
         │                          │
         ▼                          ▼
     EXPLORE  ─────────────────► CLASSIFY DISTANCE
         ▲                          │
         │                          ├─ < too_close_threshold → GO_AWAY
         │                          ├─ < comfortable_distance → MAINTAIN_DISTANCE
         │                          ├─ > too_far_threshold → GO_CLOSE
         │                          └─ else → MAINTAIN_DISTANCE
         │                          
         └──────────────────────────┘
              (person lost)
```

### Debouncing

Prevents rapid behavior changes when person is at zone boundary:

```python
BEHAVIOR_CHANGE_COOLDOWN = 2.0  # seconds

# Only change behavior if:
# 1. Different from last behavior
# 2. Cooldown timer expired
if new_behavior != self.last_behavior and time_since_change > COOLDOWN:
    self.last_behavior = new_behavior
    return new_behavior
else:
    return self.last_behavior  # Hold current behavior
```

## Command Implementation Details

### GO_CLOSE: SE2 Trajectory to Person

Uses vision frame for stable goal positions:

```python
# Get person position from detection (pan, tilt, distance)
pan_rad, tilt_rad = detection.ptz_angles
distance_m = detection.distance

# Compute 3D position in body frame (spherical → Cartesian)
person_body = Vec3(
    x=distance_m * cos(tilt_rad) * cos(pan_rad),
    y=distance_m * cos(tilt_rad) * sin(pan_rad),
    z=distance_m * sin(tilt_rad)
)

# Transform to vision frame (world coordinates)
vision_tform_body = frame_helpers.get_a_tform_b(...)
person_vision = vision_tform_body * person_body

# Compute goal position (back away from person by target_distance)
approach_vector = person_vision.normalize()
goal_position = person_vision - approach_vector * target_distance_m

# Build SE2 trajectory command
goal_se2 = SE2Pose(x=goal_position.x, y=goal_position.y, angle=yaw_to_person)
cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
    goal_se2, frame_name=VISION_FRAME_NAME, ...
)
self.clients.command.robot_command(cmd)
block_for_trajectory_cmd(self.clients.command, cmd_id)
```

### MAINTAIN_DISTANCE: Stand

Simply holds current pose:

```python
blocking_stand(self.clients.command, timeout_sec=10)
```

### GO_AWAY: Backup Trajectory

Moves backward from current position:

```python
body_tform_goal = SE2Pose(x=-1.0, y=0, angle=0)  # 1m backward
cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
    body_tform_goal, frame_name=BODY_FRAME_NAME, ...
)
```

### EXPLORE: Rotation Scan

Rotates in place to search for people:

```python
body_tform_goal = SE2Pose(x=0, y=0, angle=math.radians(90))  # 90° turn
cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
    body_tform_goal, frame_name=BODY_FRAME_NAME, ...
)
```

## Lease and E-Stop Management

`BehaviorExecutor` uses context managers for safe resource handling:

```python
with BehaviorExecutor(robot) as executor:
    # Lease and E-Stop acquired
    executor.execute_behavior(behavior)
    # Lease and E-Stop automatically released on exit
```

Equivalent to:

```python
executor = BehaviorExecutor(robot)
try:
    executor.__enter__()  # Acquire lease + estop
    executor.execute_behavior(behavior)
finally:
    executor.__exit__()  # Release lease + estop
```

### Handling ResourceAlreadyClaimedError

If tablet or another client holds the lease:

```python
try:
    with BehaviorExecutor(robot) as executor:
        ...
except ResourceAlreadyClaimedError:
    # Option 1: Force take (will interrupt tablet)
    with BehaviorExecutor(robot, force_take_lease=True) as executor:
        ...
    
    # Option 2: Return lease from tablet, then retry
```

## Configuration

Behavior parameters in `executor.py`:

```python
# Trajectory velocity limits
MAX_LINEAR_VELOCITY = 0.5  # m/s
MAX_ANGULAR_VELOCITY = 1.0  # rad/s

# Distance thresholds (used in GO_CLOSE)
TARGET_APPROACH_DISTANCE = 2.0  # meters from person

# Timeouts
TRAJECTORY_TIMEOUT = 10.0  # seconds
STAND_TIMEOUT = 3.0  # seconds
```

Comfort model parameters in `planner.py`:

```python
COMFORTABLE_DISTANCE = 2.0  # meters
TOO_CLOSE_THRESHOLD = 1.5  # meters
TOO_FAR_THRESHOLD = 3.0  # meters
BEHAVIOR_CHANGE_COOLDOWN = 2.0  # seconds
```

## Troubleshooting

### Robot Doesn't Move
- **No lease**: Check executor entered context manager (`with BehaviorExecutor(...)`)
- **E-Stop active**: Press E-Stop button or use `skip_if_active=True`
- **Motors off**: Call `robot.power_on()` before executing behaviors
- **Trajectory timeout**: Increase `TRAJECTORY_TIMEOUT` for slower robots

### Behavior Oscillates Rapidly
- **Too close to threshold**: Increase `BEHAVIOR_CHANGE_COOLDOWN` in ComfortModel
- **Noisy distance estimates**: Add low-pass filter to detection distance
- **Hysteresis needed**: Use different thresholds for entering/exiting behaviors

### Person Not in View After GO_CLOSE
- **Overshoot**: Reduce `TARGET_APPROACH_DISTANCE` or add stopping margin
- **Frame transform error**: Verify time sync (`robot.time_sync.wait_for_sync()`)
- **Vision frame drift**: Use odometry frame for shorter trajectories

## Dependencies

- `bosdyn-client` == 5.0.1.2: Spot SDK robot commands
- `numpy` >= 1.24.0: Vector math

## References

- [Spot SDK Robot Command](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/robot_command)
- [SE2 Trajectory Commands](https://dev.bostondynamics.com/docs/concepts/geometry_and_frames)
- [Frame Helpers](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/frame_helpers)
- [Proxemics Theory](https://en.wikipedia.org/wiki/Proxemics) by Edward T. Hall
- Boston Dynamics Fetch Example: `spot-sdk/python/examples/fetch/fetch.py`
