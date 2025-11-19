# Robot I/O Migration Guide

## Overview
All Spot SDK robot connection and client management has been unified into `robot_io.py` following Boston Dynamics SDK best practices.

## Changes

### New Unified Module: `robot_io.py`
Located at: `friendly_spot/robot_io.py`

**Key Components:**
- `create_robot()`: Standard connection pattern (SDK → authenticate → time sync)
- `RobotClients`: Lazy-loading container for common service clients
- `ManagedLease`: Context manager for automatic lease acquisition/return
- `ManagedEstop`: Context manager for automatic E-Stop registration/deregistration
- `configure_stream()`: Spot CAM stream configuration helper

### Updated Files

#### 1. `behavior_executor.py`
**Before:**
```python
class BehaviorExecutor:
    def __init__(self, robot, enable_estop=True):
        self.command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self.lease_client = robot.ensure_client(LeaseClient.default_service_name)
        # Manual lease/estop management
    
    def acquire_lease(self):
        # Manual lease acquisition
    
    def shutdown(self):
        # Manual cleanup
```

**After:**
```python
from robot_io import RobotClients, ManagedLease, ManagedEstop

class BehaviorExecutor:
    def __init__(self, robot):
        self.clients = RobotClients(robot)  # Lazy client loading
    
    def execute_behavior(self, behavior):
        # Context managers handle lease and E-Stop automatically
        with ManagedLease(self.robot), ManagedEstop(self.robot):
            self.clients.command.robot_command(cmd)
```

**Benefits:**
- No manual `acquire_lease()` needed
- No manual `shutdown()` needed
- Automatic resource cleanup via context managers
- Cleaner separation of concerns
- Secure credential handling via SDK authentication

#### 2. `friendly_spot_main.py`
**Before:**
```python
from bosdyn.client import create_standard_sdk
from bosdyn.client.util import authenticate

def _connect_robot(self):
    sdk = create_standard_sdk('FriendlySpot')
    robot = sdk.create_robot(self.args.robot)
    authenticate(robot)
    robot.time_sync.wait_for_sync()
    return robot

# Later:
self.executor = BehaviorExecutor(self.robot)
self.executor.acquire_lease()
```

**After:**
```python
from robot_io import create_robot

def _connect_robot(self):
    return create_robot(
        hostname=self.args.robot,
        client_name='FriendlySpot',
        register_spot_cam=True,
        verbose=False
    )

# Later:
self.executor = BehaviorExecutor(self.robot)
# No acquire_lease() call needed
```

**Benefits:**
- Single function call for full robot setup
- Automatically registers Spot CAM services
- Supports env var authentication (`BOSDYN_CLIENT_USERNAME`, `BOSDYN_CLIENT_PASSWORD`)
- No manual cleanup in shutdown

#### 3. `people_observer/io_robot.py` → **DELETED**
**Before:**
```python
def connect(hostname):
    sdk = create_standard_sdk("PeopleObserver")
    spot_cam.register_all_service_clients(sdk)
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    return robot

def ensure_clients(robot):
    image = robot.ensure_client(ImageClient.default_service_name)
    comp = robot.ensure_client(CompositorClient.default_service_name)
    sq = robot.ensure_client(StreamQualityClient.default_service_name)
    return image, comp, sq

def configure_stream(robot, cfg):
    robot.ensure_client(CompositorClient.default_service_name).set_screen(cfg.ptz.compositor_screen)
    robot.ensure_client(StreamQualityClient.default_service_name).set_stream_params(
        target_bitrate=cfg.ptz.target_bitrate
    )
```

**After:**
File deleted - functionality moved to parent `robot_io.py`. Only had wrapper functions with no unique logic.

**Benefits:**
- Eliminates unnecessary file and code duplication
- One less file to maintain
- Direct import from unified `robot_io.py`

#### 4. `people_observer/app.py`
**Before:**
```python
from .io_robot import connect, ensure_clients, configure_stream

robot = connect(args.hostname)
image_client, compositor, stream_quality = ensure_clients(robot)
configure_stream(robot, cfg)
```

**After:**
```python
from robot_io import create_robot, configure_stream

robot = create_robot(
    hostname=args.hostname,
    client_name="PeopleObserver",
    register_spot_cam=True,
    verbose=False
)

# Get service clients directly
image_client = robot.ensure_client(ImageClient.default_service_name)
compositor = robot.ensure_client(CompositorClient.default_service_name)
stream_quality = robot.ensure_client(StreamQualityClient.default_service_name)

# Configure PTZ stream
configure_stream(
    robot,
    compositor_screen=cfg.ptz.compositor_screen,
    target_bitrate=cfg.ptz.target_bitrate
)
```

**Benefits:**
- Direct usage of unified `robot_io` module
- No unnecessary wrapper layer
- Clearer code flow

## Usage Patterns

### Basic Robot Connection
```python
from robot_io import create_robot

# With Spot CAM services (for PTZ, streaming)
robot = create_robot("192.168.80.3", register_spot_cam=True)

# Without Spot CAM (for basic movement)
robot = create_robot("192.168.80.3", register_spot_cam=False)
```

### Using Lazy Clients
```python
from robot_io import RobotClients

clients = RobotClients(robot)

# Clients are created only when accessed
image = clients.image.get_image(...)
clients.command.robot_command(cmd)
state = clients.state.get_robot_state()
```

### Executing Commands with Lease/E-Stop
```python
from robot_io import ManagedLease, ManagedEstop

# Context managers handle all resource management
with ManagedLease(robot), ManagedEstop(robot, name="MyApp"):
    clients = RobotClients(robot)
    clients.command.robot_command(cmd)
# Lease and E-Stop automatically cleaned up
```

### Configuring PTZ Stream
```python
from robot_io import configure_stream

configure_stream(
    robot,
    compositor_screen="ptz",
    target_bitrate=5000000  # 5 Mbps
)
```

## Authentication Methods

The `create_robot()` function supports multiple authentication methods (in priority order):

1. **Existing token** (if robot already authenticated)
2. **Environment variables** (recommended): Set in your shell activation script
3. **Interactive prompt** (fallback if no env vars)

**Security Note:** Never hardcode credentials in source code. Use environment variables set outside of version control.

## Migration Checklist

**Completed:**
- `robot_io.py` created with unified connection logic
- `behavior_executor.py` updated to use `RobotClients` and context managers
- `friendly_spot_main.py` updated to use `create_robot()`
- `people_observer/io_robot.py` **DELETED** (was only wrappers, no unique logic)
- `people_observer/app.py` updated to import directly from `robot_io`
- All files pass lint validation

## Testing

1. **Test robot connection:**
   ```powershell
   python -c "from robot_io import create_robot; robot = create_robot('ROBOT_IP', register_spot_cam=True); print('Connected:', robot.is_powered_on())"
   ```

2. **Test people_observer (backward compatibility):**
   ```powershell
   python -m people_observer.app ROBOT_IP --once
   ```

3. **Test friendly_spot pipeline:**
   ```powershell
   python friendly_spot_main.py --robot ROBOT_IP --rate 5
   ```

## Benefits Summary

1. **Code Reuse**: Single source of truth for robot connection logic
2. **SDK Compliance**: Follows Boston Dynamics best practices from examples
3. **Automatic Cleanup**: Context managers prevent resource leaks
4. **Lazy Loading**: Clients created only when needed (performance)
5. **Backward Compatibility**: Existing code continues to work
6. **Maintainability**: Changes to connection logic only need to happen in one place
7. **Security**: Supports environment variable authentication (no hardcoded credentials)

## References

- `spot-sdk/python/examples/hello_spot/hello_spot.py`: Standard connection pattern
- `spot-sdk/python/bosdyn-client/src/bosdyn/client/util.py`: Authentication utilities
- `friendly_spot/robot_io.py`: Unified implementation
