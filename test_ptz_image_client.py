#!/usr/bin/env python3
"""Quick test of SpotPTZImageClient with correct Spot CAM service configuration.

Tests the corrected PTZ camera access using:
- Image service: 'spot-cam-image' (Spot CAM's dedicated image service)
- Source name: 'ptz' (PTZ camera source within Spot CAM service)

Usage:
    python test_ptz_image_client.py --hostname ROBOT_IP
"""
import argparse
import logging

import bosdyn.client
import bosdyn.client.util
from bosdyn.client import spot_cam

from video_sources import SpotPTZImageClient
from robot_io import create_robot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--frames', type=int, default=5, help='Number of frames to capture')
    options = parser.parse_args()
    
    logger.info(f"Connecting to robot at {options.hostname}...")
    
    # Create robot with Spot CAM services registered
    robot = create_robot(options.hostname, register_spot_cam=True, verbose=options.verbose)
    
    logger.info("Testing SpotPTZImageClient with Spot CAM configuration...")
    logger.info("  Image service: 'spot-cam-image'")
    logger.info("  Source name: 'ptz'")
    logger.info("  Expected resolution: 1920x1080 (Full HD)")
    
    # Create PTZ video source (uses global config from video_sources.py)
    with SpotPTZImageClient(robot) as ptz_source:
        logger.info(f"Successfully initialized PTZ client")
        
        # Capture test frames
        for i in range(options.frames):
            success, frame, depth = ptz_source.read()
            
            if success:
                logger.info(f"Frame {i+1}/{options.frames}: ✓ {frame.shape[1]}x{frame.shape[0]} pixels")
            else:
                logger.error(f"Frame {i+1}/{options.frames}: ✗ Failed to capture")
        
        logger.info("Test complete!")


if __name__ == '__main__':
    main()
