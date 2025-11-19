# Robot Module

Robot connection, authentication, client management, and control utilities for Spot SDK.

## Overview

The robot module provides a clean interface for all Spot SDK operations, consolidating connection management, client initialization, lease/estop handling, and PTZ camera control. Follows Boston Dynamics SDK best practices.

## Components

### Connection & Clients
- **`io.py`**: Robot connection and client management
  - `create_robot()`: Authenticate and time-sync robot connection
  - `RobotClients`: Lazy-loading container for SDK service clients (command, state, image, etc.)
  - `ManagedLease`: Context manager for lease acquisition/release
  - `ManagedEstop`: Context manager for E-Stop registration/deregistration
  - Utilities: `ensure_motors_on()`, `safe_power_off()`

### Action Monitoring
- **`action_monitor.py`**: Async command status tracking
  - Monitor long-running commands (trajectories, manipulation) without blocking
  - Callback system for command completion/failure
  - Used by behavior executor for parallel operations

### Perception Bridge
- **`observer_bridge.py`**: Bridge detection events to robot behaviors
  - Converts perception detections → behavior inputs
  - Manages state between perception and behavior modules
  - Handles multi-person tracking and target selection

### PTZ Control
- **`ptz_control.py`**: Spot CAM PTZ camera control
  - `set_ptz()`: Command pan, tilt, zoom angles
  - Angle convention: pan ∈ [-180°, 180°], tilt ∈ [-90°, 90°], zoom ∈ [1.0, 30.0]
  - Rate limiting to avoid overwhelming PTZ service

## Usage

### Basic Robot Connection

```python
from src.robot import create_robot, RobotClients

# Connect and authenticate
robot = create_robot(
    hostname="192.168.80.3",  # Robot IP
    register_spot_cam=True,    # Enable PTZ/compositor
    verbose=False
)

# Get service clients
clients = RobotClients(robot)

# Use clients (lazy initialization)
state = clients.state.get_robot_state()
print(f"Battery: {state.power_state.locomotion_charge_percentage.value}%")
```

### Lease Management

Acquire control for commanding robot:

```python
from src.robot import ManagedLease, ensure_motors_on

# Acquire lease (returns control to prior holder on exit)
with ManagedLease(robot) as lease:
    ensure_motors_on(robot)
    # Send commands...
    cmd = RobotCommandBuilder.synchro_stand_command()
    clients.command.robot_command(cmd)
# Lease automatically released

# Force take lease from tablet/other clients
with ManagedLease(robot, force_take=True) as lease:
    # Commands execute even if tablet was controlling
    pass
```

### E-Stop Registration

Required for safety before motor commands:

```python
from src.robot import ManagedEstop

# Register E-Stop endpoint
with ManagedEstop(robot, name="MyController") as estop:
    # E-Stop registered and active
    # Can now send motion commands
    pass
# E-Stop deregistered

# Skip registration if E-Stop already active (for multi-client setups)
with ManagedEstop(robot, skip_if_active=True) as estop:
    pass
```

### PTZ Camera Control

Command Spot CAM PTZ:

```python
from src.robot import set_ptz

# Point camera at person (pan=0°, tilt=-20°, zoom=2x)
set_ptz(clients.ptz, pan_deg=0.0, tilt_deg=-20.0, zoom=2.0)

# Scan area (multiple angles)
for pan in [-45, 0, 45]:
    set_ptz(clients.ptz, pan_deg=pan, tilt_deg=0.0)
    time.sleep(1)
```

### Full Control Session

Combined lease + estop + motors:

```python
from src.robot import create_robot, RobotClients, ManagedLease, ManagedEstop, ensure_motors_on

robot = create_robot("192.168.80.3", register_spot_cam=True)
clients = RobotClients(robot)

with ManagedLease(robot) as lease:
    with ManagedEstop(robot, name="Session") as estop:
        ensure_motors_on(robot)
        
        # Stand up
        blocking_stand(clients.command, timeout_sec=10)
        
        # Walk forward 1m
        goal = SE2Pose(x=1.0, y=0, angle=0)
        cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
            goal, frame_name=BODY_FRAME_NAME
        )
        clients.command.robot_command(cmd)
```

## Client Types

`RobotClients` provides lazy-loaded clients for:

| Client | Purpose | Service Name |
|--------|---------|--------------|
| `command` | Motion commands (stand, walk, sit) | `robot-command` |
| `state` | Robot state (battery, poses, joints) | `robot-state` |
| `lease` | Lease management | `lease` |
| `estop` | E-Stop registration | `estop` |
| `power` | Motor power on/off | `power` |
| `image` | Camera image acquisition | `image` |
| `world_object` | Fiducial/object detection | `world-object` |
| `compositor` | Spot CAM video compositor | `spot-cam-compositor` (optional) |
| `stream_quality` | Spot CAM stream settings | `spot-cam-stream-quality` (optional) |
| `ptz` | Spot CAM PTZ control | `spot-cam-ptz` (optional) |

Clients are initialized on first access:

```python
clients = RobotClients(robot)
clients.command  # First access: creates RobotCommandClient
clients.command  # Subsequent: returns cached instance
```

## Authentication

`create_robot()` uses `bosdyn.client.util.authenticate()` which tries:

1. **Stored token**: Previous session token from `~/.bosdyn/user_tokens`
2. **Environment variables**: `BOSDYN_CLIENT_USERNAME`, `BOSDYN_CLIENT_PASSWORD`
3. **Interactive prompt**: Asks for username/password

Set environment variables to avoid prompts:

```powershell
# PowerShell
$env:BOSDYN_CLIENT_USERNAME = "user"
$env:BOSDYN_CLIENT_PASSWORD = "password"
python script.py
```

```bash
# Bash
export BOSDYN_CLIENT_USERNAME="user"
export BOSDYN_CLIENT_PASSWORD="password"
python script.py
```

## PTZ Angle Conventions

### Coordinate System

PTZ angles are in **body frame** (robot-relative):

- **Pan**: Rotation around Z-axis (yaw)
  - 0° = straight ahead (robot's forward direction)
  - +90° = left
  - -90° = right
  - Range: [-180°, 180°]

- **Tilt**: Rotation around Y-axis (pitch)
  - 0° = horizontal
  - +45° = up
  - -45° = down
  - Range: [-90°, 90°] (hardware limited)

- **Zoom**: Optical zoom factor
  - 1.0 = wide angle
  - 30.0 = maximum zoom
  - Range: [1.0, 30.0]

### Example Angles

```python
# Look straight ahead
set_ptz(ptz_client, pan_deg=0, tilt_deg=0, zoom=1.0)

# Look up-left
set_ptz(ptz_client, pan_deg=45, tilt_deg=30, zoom=2.0)

# Look down-right
set_ptz(ptz_client, pan_deg=-45, tilt_deg=-30, zoom=1.0)

# Pan left, level horizon
set_ptz(ptz_client, pan_deg=90, tilt_deg=0, zoom=1.0)
```

## Error Handling

### Common Errors

**`RpcError: ServiceUnavailableError`**
- **Cause**: Service not running (e.g., no Spot CAM on base Spot)
- **Solution**: Check `register_spot_cam=True` only used on robots with Spot CAM, or catch exception and fallback

**`LeaseUseError: ResourceAlreadyClaimedError`**
- **Cause**: Another client (tablet, SDK script) holds the lease
- **Solution**: Use `ManagedLease(robot, force_take=True)` or return lease from other client

**`EstopError`**
- **Cause**: E-Stop not registered or deactivated
- **Solution**: Use `ManagedEstop` context manager before sending commands

**`PowerError: MotorsNotPowered`**
- **Cause**: Robot motors are off
- **Solution**: Call `ensure_motors_on(robot)` after acquiring lease

### Graceful Shutdown

Always release resources on exit:

```python
import atexit

robot = create_robot("192.168.80.3")
clients = RobotClients(robot)

def cleanup():
    try:
        safe_power_off(robot)  # Sits robot and powers off motors
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

atexit.register(cleanup)
```

## Configuration

### Client Initialization

Clients are lazy-loaded and cached:

```python
# Fast: Only creates requested clients
clients = RobotClients(robot)
cmd = clients.command  # Creates RobotCommandClient
img = clients.image    # Creates ImageClient
# Other clients not created

# vs. eager initialization (not recommended)
cmd_client = robot.ensure_client(RobotCommandClient.default_service_name)
img_client = robot.ensure_client(ImageClient.default_service_name)
# ... manually create each client
```

### Spot CAM Availability

Check before using PTZ/compositor:

```python
from src.robot.io import SPOT_CAM_AVAILABLE

if SPOT_CAM_AVAILABLE:
    robot = create_robot(hostname, register_spot_cam=True)
    clients.ptz.set_ptz_position(...)
else:
    logger.warning("Spot CAM not available, skipping PTZ")
```

### Time Sync

Time sync is **required** for robot commands (trajectory goals are timestamped). `create_robot()` performs sync automatically, but verify:

```python
if not robot.time_sync.has_established_time_sync:
    robot.time_sync.wait_for_sync(timeout_sec=10)
```

## Troubleshooting

### Robot Connection Fails
- **Network**: Ping robot IP (`ping 192.168.80.3`)
- **Credentials**: Set `BOSDYN_CLIENT_USERNAME` and `BOSDYN_CLIENT_PASSWORD` env vars
- **Firewall**: Ensure ports 443 (HTTPS) and 50051 (gRPC) are open

### Lease Not Acquired
- **Tablet holding lease**: Return lease from tablet or use `force_take=True`
- **Another script**: Check `robot.list_leases()` to see lease holders

### E-Stop Not Registered
- **Multiple endpoints**: Only one E-Stop endpoint can be active per client
- **Keepalive timeout**: Ensure keepalive thread is running (automatic in `ManagedEstop`)

### PTZ Commands Ignored
- **Spot CAM not registered**: Use `register_spot_cam=True` in `create_robot()`
- **PTZ service unavailable**: Check robot has Spot CAM hardware
- **Rate limit**: Reduce PTZ command frequency (<10 Hz)

## Dependencies

- `bosdyn-client` == 5.0.1.2: Spot SDK core
- `bosdyn-api` == 5.0.1.2: Protobuf definitions

## References

- [Spot SDK Quickstart](https://dev.bostondynamics.com/docs/python/quickstart)
- [Robot Command Service](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/robot_command)
- [Lease Management](https://dev.bostondynamics.com/docs/concepts/lease)
- [E-Stop Service](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/estop)
- [Spot CAM PTZ](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/spot_cam/ptz)
