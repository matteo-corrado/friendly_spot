#!/usr/bin/env python3
"""Print camera frame transforms from Spot robot.

Fetches image sources and prints the transform information for each camera,
including rotation matrices and position vectors in the body frame.

Usage:
    python print_camera_transforms.py --robot ROBOT_IP
    python print_camera_transforms.py --robot ROBOT_IP --user USER --password PASS
"""

import argparse
import logging
import sys
from bosdyn.client import create_standard_sdk
from bosdyn.client.image import ImageClient
from bosdyn.client import frame_helpers
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_rotation_matrix(rot, indent="  "):
    """Print a rotation matrix in readable format."""
    # Extract rotation matrix as 3x3 numpy array
    # Transform basis vectors to get columns
    x_axis = rot.transform_point(1, 0, 0)
    y_axis = rot.transform_point(0, 1, 0)
    z_axis = rot.transform_point(0, 0, 1)
    
    matrix = np.array([
        [x_axis[0], y_axis[0], z_axis[0]],
        [x_axis[1], y_axis[1], z_axis[1]],
        [x_axis[2], y_axis[2], z_axis[2]]
    ])
    
    print(f"{indent}Rotation Matrix:")
    for row in matrix:
        print(f"{indent}  [{row[0]:7.4f}, {row[1]:7.4f}, {row[2]:7.4f}]")


def analyze_camera_orientation(rot):
    """Analyze camera orientation from rotation matrix."""
    # Get camera axes in body frame
    cam_x = rot.transform_point(1, 0, 0)  # Camera +X (right)
    cam_y = rot.transform_point(0, 1, 0)  # Camera +Y (down)
    cam_z = rot.transform_point(0, 0, 1)  # Camera +Z (forward/optical axis)
    
    # Find primary alignment for optical axis (Z)
    z_components = [abs(cam_z[0]), abs(cam_z[1]), abs(cam_z[2])]
    primary_axis = ['X', 'Y', 'Z'][z_components.index(max(z_components))]
    direction = '+' if cam_z[z_components.index(max(z_components))] > 0 else '-'
    
    return f"Optical axis (cam +Z) points toward body {direction}{primary_axis}"


def print_camera_transforms(robot):
    """Fetch and print camera frame transforms."""
    logger.info("Fetching image sources...")
    
    # Get image clients for both standard and Spot CAM services
    image_client = robot.ensure_client(ImageClient.default_service_name)
    sources = image_client.list_image_sources()
    
    # Try to get Spot CAM sources (PTZ, pano, etc.)
    try:
        spot_cam_client = robot.ensure_client('spot-cam-image')
        spot_cam_sources = spot_cam_client.list_image_sources()
        sources.extend(spot_cam_sources)
        logger.info(f"Found {len(spot_cam_sources)} Spot CAM sources")
    except Exception as e:
        logger.warning(f"Spot CAM service not available: {e}")
    
    print("\n" + "="*80)
    print("SPOT CAMERA FRAME TRANSFORMS")
    print(f"Total cameras: {len(sources)}")
    print("="*80)
    
    for src in sources:
        print(f"\n Camera: {src.name}")
        print(f"   Resolution: {src.cols}x{src.rows}")
        
        # Check camera model
        if src.HasField('pinhole'):
            model = "PINHOLE"
            fx = src.pinhole.intrinsics.focal_length.x
            fy = src.pinhole.intrinsics.focal_length.y
            cx = src.pinhole.intrinsics.principal_point.x
            cy = src.pinhole.intrinsics.principal_point.y
            print(f"   Model: {model} (fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f})")
        elif src.HasField('kannala_brandt'):
            model = "KANNALA-BRANDT (Fisheye)"
            fx = src.kannala_brandt.intrinsics.focal_length.x
            fy = src.kannala_brandt.intrinsics.focal_length.y
            k1 = src.kannala_brandt.k1
            k2 = src.kannala_brandt.k2
            print(f"   Model: {model} (fx={fx:.1f}, fy={fy:.1f}, k1={k1:.4f}, k2={k2:.4f})")
        else:
            print(f"   Model: UNKNOWN")
        
        # Fetch a single image to get transform snapshot
        from bosdyn.api import image_pb2
        request = [image_pb2.ImageRequest(
            image_source_name=src.name,
            quality_percent=10,  # Low quality, we just need transforms
            resize_ratio=0.25     # Small size
        )]
        
        try:
            # Use the appropriate client based on source name
            # PTZ, pano, c0-c4 are from spot-cam-image service
            if src.name in ['ptz', 'pano'] or src.name.startswith('c'):
                client = spot_cam_client if 'spot_cam_client' in locals() else image_client
            else:
                client = image_client
            
            responses = client.get_image(request)
            if not responses:
                print(f" No image response")
                continue
                
            resp = responses[0]
            shot = resp.shot
            
            if not shot.HasField('transforms_snapshot'):
                print(f" No transform snapshot available")
                continue
            
            frame_tree = shot.transforms_snapshot
            camera_frame = shot.frame_name_image_sensor
            body_frame = frame_helpers.BODY_FRAME_NAME
            
            print(f"\n   Frame Names:")
            print(f"      Camera frame: {camera_frame}")
            print(f"      Body frame: {body_frame}")
            
            # Get body_T_camera transform
            try:
                body_T_camera = frame_helpers.get_a_tform_b(
                    frame_tree, body_frame, camera_frame
                )
                
                print(f"\n   Transform: body_T_camera")
                print(f"      Position (meters):")
                print(f"         X: {body_T_camera.x:.4f} m")
                print(f"         Y: {body_T_camera.y:.4f} m")
                print(f"         Z: {body_T_camera.z:.4f} m")
                
                print_rotation_matrix(body_T_camera.rot, indent="      ")
                
                # Analyze orientation
                orientation = analyze_camera_orientation(body_T_camera.rot)
                print(f"\n      Orientation: {orientation}")
                
                # Calculate rotation angle for image correction
                cam_z = body_T_camera.rot.transform_point(0, 0, 1)
                z_vec = [cam_z[0], cam_z[1], cam_z[2]]
                max_component = max(enumerate(z_vec), key=lambda x: abs(x[1]))
                axis_idx, value = max_component
                
                if axis_idx == 2 and value > 0:  # Z+ (upright)
                    rotation_needed = 0
                elif axis_idx == 2 and value < 0:  # Z- (upside down)
                    rotation_needed = 180
                elif axis_idx == 1 and value > 0:  # Y+ (90° CCW)
                    rotation_needed = 90
                elif axis_idx == 1 and value < 0:  # Y- (90° CW)
                    rotation_needed = 270
                else:
                    rotation_needed = "CUSTOM"
                
                print(f"      Image rotation needed: {rotation_needed}°")
                
            except Exception as e:
                print(f"   Transform error: {e}")
        
        except Exception as e:
            print(f"  Failed to fetch image: {e}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--robot',
        required=True,
        help='Robot hostname or IP address'
    )
    parser.add_argument(
        '--user',
        help='Robot username (or set BOSDYN_CLIENT_USERNAME env var)'
    )
    parser.add_argument(
        '--password',
        help='Robot password (or set BOSDYN_CLIENT_PASSWORD env var)'
    )
    
    args = parser.parse_args()
    
    # Create SDK and robot instance
    sdk = create_standard_sdk('CameraTransformPrinter')
    robot = sdk.create_robot(args.robot)
    
    # Authenticate
    if args.user and args.password:
        robot.authenticate(args.user, args.password)
    else:
        import bosdyn.client.util
        bosdyn.client.util.authenticate(robot)
    
    # Time sync
    robot.time_sync.wait_for_sync()
    
    # Print transforms
    print_camera_transforms(robot)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
