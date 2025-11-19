"""Test PTZ pan convention to verify geometry assumptions.

This script commands the PTZ to 4 cardinal directions and waits for user confirmation
at each position to verify the pan convention matches our geometry calculations.

Expected behavior if convention is correct:
- Pan 0°: PTZ points forward (same as robot body +X)
- Pan 90°: PTZ points right (body -Y)
- Pan 180°: PTZ points backward (body -X)
- Pan 270°: PTZ points left (body +Y)

Usage:
    python test_ptz_convention.py ROBOT_IP
"""
import argparse
import sys
import time

import bosdyn.client
import bosdyn.client.util
from bosdyn.client import spot_cam
from bosdyn.client.spot_cam.ptz import PtzClient
from bosdyn.api.spot_cam import ptz_pb2


def test_ptz_convention(hostname: str):
    """Test PTZ pan convention by commanding 4 cardinal directions."""
    
    # PTZ mechanical offset: PTZ hardware is offset 45° to the right of body frame
    # When we command 0°, the PTZ points at 45° (robot's front-right)
    # To point forward (body +X), we need to command -45°
    PTZ_OFFSET_DEG = 35.0
    
    # Connect to robot
    sdk = bosdyn.client.create_standard_sdk('PTZConventionTest')
    
    # Register all Spot CAM service clients
    spot_cam.register_all_service_clients(sdk)
    
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()
    
    ptz_client = robot.ensure_client(PtzClient.default_service_name)
    ptz_desc = ptz_pb2.PtzDescription(name="mech")
    
    # Test positions: (desired_body_bearing_deg, expected_direction_description)
    # We'll apply the offset to get the actual PTZ command
    test_positions = [
        (0, "FORWARD (body +X, straight ahead)"),
        (90, "RIGHT (body -Y, robot's right side)"),
        (180, "BACKWARD (body -X, behind robot)"),
        (270, "LEFT (body +Y, robot's left side)"),
        (0, "FORWARD again (returning to start)")
    ]
    
    tilt = 0.0  # Level horizon
    zoom = 1.0  # No zoom
    
    print("=" * 70)
    print("PTZ PAN CONVENTION TEST (with 45° offset correction)")
    print("=" * 70)
    print("\nThis test will move the PTZ to 4 cardinal directions.")
    print("Please observe which direction the PTZ camera points at each position.")
    print(f"\nApplying PTZ hardware offset: {PTZ_OFFSET_DEG}° to the right")
    print("(PTZ is mechanically mounted 45° offset from body frame)")
    print("\nExpected convention (after offset correction):")
    print("  Body   0°: PTZ points FORWARD (same as robot facing direction)")
    print("  Body  90°: PTZ points RIGHT (robot's right side)")
    print("  Body 180°: PTZ points BACKWARD (behind robot)")
    print("  Body 270°: PTZ points LEFT (robot's left side)")
    print("\n" + "=" * 70)
    
    input("\nPress ENTER to start test...")
    
    for i, (body_bearing_deg, expected_desc) in enumerate(test_positions, 1):
        # Apply offset correction: subtract offset because PTZ is rotated right
        # If PTZ hardware points 45° right of commanded angle, we command (desired - 45)
        ptz_command_deg = (body_bearing_deg - PTZ_OFFSET_DEG) % 360
        
        print(f"\n[{i}/{len(test_positions)}] Desired body bearing: {body_bearing_deg}°")
        print(f"    PTZ command (with offset): {ptz_command_deg}°, tilt={tilt}°, zoom={zoom}")
        print(f"    Expected direction: {expected_desc}")
        
        try:
            # Command PTZ position with offset correction
            ptz_client.set_ptz_position(ptz_desc, ptz_command_deg, tilt, zoom)
            print("    ✓ Command sent successfully")
            
            # Wait for PTZ to move
            time.sleep(2.0)
            
            # Query actual position
            current_pos = ptz_client.get_ptz_position(ptz_desc)
            actual_pan = current_pos.pan.value
            actual_tilt = current_pos.tilt.value
            print(f"    Actual PTZ position: pan={actual_pan:.1f}°, tilt={actual_tilt:.1f}°")
            
            # Wait for user confirmation
            response = input(f"\n    Does PTZ point {expected_desc}? (y/n/skip): ").strip().lower()
            
            if response == 'y':
                print("    ✓ PASS: Convention matches expectation")
            elif response == 'n':
                print("    ✗ FAIL: Convention does NOT match expectation")
                print("\n" + "!" * 70)
                print("CONVENTION MISMATCH DETECTED!")
                print("The geometry calculations may need adjustment.")
                print("!" * 70)
                cont = input("\nContinue test anyway? (y/n): ").strip().lower()
                if cont != 'y':
                    print("Test aborted by user.")
                    return False
            elif response == 'skip':
                print("    - SKIPPED by user")
            else:
                print("    ? Unknown response, continuing...")
                
        except Exception as e:
            print(f"    ✗ ERROR: {type(e).__name__}: {e}")
            cont = input("\nContinue despite error? (y/n): ").strip().lower()
            if cont != 'y':
                print("Test aborted by user.")
                return False
    
    print("\n" + "=" * 70)
    print("PTZ CONVENTION TEST COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print("- If all positions matched expectations: Offset correction is CORRECT ✓")
    print(f"- PTZ hardware offset applied: {PTZ_OFFSET_DEG}°")
    print("- If positions still off: Adjust PTZ_OFFSET_DEG in geometry.py")
    print("\nNext steps:")
    print("  1. If test passed, update people_observer/geometry.py")
    print("  2. Add PTZ_OFFSET_DEG constant to config.py")
    print("  3. Apply offset in pixel_to_ptz_angles_transform()")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    bosdyn.client.util.add_base_arguments(parser)
    args = parser.parse_args()
    
    try:
        success = test_ptz_convention(args.hostname)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user (Ctrl+C)")
        return 130
    except Exception as e:
        print(f"\n\nUnexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
