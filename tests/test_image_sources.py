#!/usr/bin/env python3
# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Image source discovery test listing all available cameras from standard and Spot CAM services
# Acknowledgements: Boston Dynamics Spot SDK ImageClient, Claude for test implementation

"""Test script to list all available image sources from robot.

This script connects to the robot and lists all available image sources
from both the standard image service and Spot CAM (if available).

Usage:
    python test_list_image_sources.py --hostname ROBOT_IP

Purpose:
    - Discover correct source names for PTZ camera
    - Validate Spot CAM registration
    - Understand compositor screen vs image source relationship
"""
import argparse
import logging

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from grpc import services

# Optional Spot CAM imports
try:
    from bosdyn.client import spot_cam
    from bosdyn.client.spot_cam.compositor import CompositorClient
    from bosdyn.client.spot_cam.ptz import PtzClient
    SPOT_CAM_AVAILABLE = True
except ImportError:
    SPOT_CAM_AVAILABLE = False
    print("WARNING: Spot CAM modules not available")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args()
    
    # Create SDK and robot connection
    sdk = bosdyn.client.create_standard_sdk('ImageSourceLister')
    
    # Register Spot CAM if available
    if SPOT_CAM_AVAILABLE:
        spot_cam.register_all_service_clients(sdk)
        logger.info("Spot CAM services registered")
    
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()
    
    logger.info(f"Connected to robot at {options.hostname}")
    print("\n" + "="*80)
    print("IMAGE SOURCES DISCOVERY")
    print("="*80 + "\n")
    
    # 1. List standard image sources
    print("1. STANDARD IMAGE SERVICE SOURCES:")
    print("-" * 80)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    sources = image_client.list_image_sources()
    
    for i, source in enumerate(sources, 1):
        print(f"\n[{i}] Source: {source.name}")
        print(f"    Rows: {source.rows}, Cols: {source.cols}")
        
        # Check camera model
        if source.HasField('pinhole'):
            fx = source.pinhole.intrinsics.focal_length.x
            fy = source.pinhole.intrinsics.focal_length.y
            cx = source.pinhole.intrinsics.principal_point.x
            cy = source.pinhole.intrinsics.principal_point.y
            print(f"    Model: PINHOLE")
            print(f"    Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        elif source.HasField('kannala_brandt'):
            fx = source.kannala_brandt.intrinsics.focal_length.x
            fy = source.kannala_brandt.intrinsics.focal_length.y
            cx = source.kannala_brandt.intrinsics.principal_point.x
            cy = source.kannala_brandt.intrinsics.principal_point.y
            k1 = source.kannala_brandt.k1
            k2 = source.kannala_brandt.k2
            k3 = source.kannala_brandt.k3
            k4 = source.kannala_brandt.k4
            print(f"    Model: KANNALA-BRANDT (fisheye)")
            print(f"    Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            print(f"    Distortion: k1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}, k4={k4:.4f}")
        else:
            print(f"    Model: UNKNOWN")
        
        # Check for depth
        has_depth = 'depth' in source.name.lower()
        print(f"    Is Depth: {has_depth}")
    
    print(f"\n{'='*80}")
    print(f"Total sources found: {len(sources)}")
    
    # Categorize sources
    visual_sources = [s.name for s in sources if 'depth' not in s.name.lower()]
    depth_sources = [s.name for s in sources if 'depth' in s.name.lower()]
    ptz_sources = [s.name for s in sources if any(x in s.name.lower() for x in ['ptz', 'digi', 'mech', 'pano'])]
    
    print(f"\nVisual sources ({len(visual_sources)}):")
    for s in visual_sources:
        print(f"  - {s}")
    
    print(f"\nDepth sources ({len(depth_sources)}):")
    for s in depth_sources:
        print(f"  - {s}")
    
    print(f"\nPotential PTZ/Spot CAM sources ({len(ptz_sources)}):")
    for s in ptz_sources:
        print(f"  - {s}")
    
    # 2. List Spot CAM compositor screens if available
    if SPOT_CAM_AVAILABLE:
        print(f"\n{'='*80}")
        print("2. SPOT CAM COMPOSITOR SCREENS:")
        print("-" * 80)
        
        try:
            compositor_client = robot.ensure_client(CompositorClient.default_service_name)
            screens = compositor_client.list_screens()
            current_screen = compositor_client.get_screen()
            
            print(f"\nCurrent screen: {current_screen.name}")
            print(f"\nAvailable screens ({len(screens.screens)}):")
            for screen in screens.screens:
                marker = " <- CURRENT" if screen.name == current_screen.name else ""
                print(f"  - {screen.name}{marker}")
        except Exception as e:
            print(f"Could not query compositor: {e}")
        
        # 3. List PTZ cameras
        print(f"\n{'='*80}")
        print("3. SPOT CAM PTZ CAMERAS:")
        print("-" * 80)
        
        try:
            ptz_client = robot.ensure_client(PtzClient.default_service_name)
            ptzs = ptz_client.list_ptz()
            
            print(f"\nAvailable PTZ cameras ({len(ptzs)}):")
            for desc in ptzs:
                print(f"\n  PTZ: {desc.name}")
                # Get current position
                try:
                    pos = ptz_client.get_ptz_position(desc)
                    print(f"    Pan: {pos.pan.value:.1f}°")
                    print(f"    Tilt: {pos.tilt.value:.1f}°")
                    print(f"    Zoom: {pos.zoom.value:.1f}x")
                except Exception as e:
                    repr(e)
                    #print(f"    (Could not get position: {e})")
        except Exception as e:
            print(f"Could not query PTZ cameras: {e}")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR video_sources.py:")
    print("="*80)
    
    if ptz_sources:
        print("\n[OK] PTZ/Spot CAM sources found in image service:")
        print("  Update SpotPTZImageClient default source_name to one of:")
        for s in ptz_sources[:3]:  # Show top 3
            print(f"    - '{s}'")
    else:
        print("\n[X] No PTZ sources found in image service")
        print("  Options:")
        print("    1. Use WebRTC streaming (ptz_stream.py already implements this)")
        print("    2. Set compositor screen and capture from composite stream")
        print("    3. Check if Spot CAM is properly configured on robot")
    
    print("\n")

    from bosdyn.client.directory import DirectoryClient
    directory = robot.ensure_client(DirectoryClient.default_service_name)
    services = directory.list()
    print("=== Services Registered on Robot ===")
    for s in services:
      print(f"{s.name}  ->  {s.type}")

    image_services = [s for s in services if s.type == "bosdyn.api.ImageService"]

    print("\n=== Image Services Found ===")
    for svc in image_services:
        print(f"- {svc.name}")



    print("\n=== Image Sources Per Service ===")
    for svc in image_services:
        try:
            ic = robot.ensure_client(svc.name)
            sources = ic.list_image_sources()
            print(f"\nImage sources under service '{svc.name}':")
            for src in sources:
                print(f"  - {src.name}")
        except Exception as e:
            print(f"  Could not access {svc.name}: {e}")

    from bosdyn.api import image_pb2
    from bosdyn.client.image import save_images_as_files,build_image_request

    svc = "spot-cam-image"  # or "image" if that's where you found the source
    src = "ptz"  # actual name discovered above

    image_client = robot.ensure_client(svc)

    req = build_image_request(
    image_source_name=src,
    pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
    quality_percent=75,
    )

    resp = image_client.get_image([req])

    save_images_as_files(resp, filename="test", filepath="images")
    print("Successfully retrieved PTZ image!")

if __name__ == '__main__':
    main()
