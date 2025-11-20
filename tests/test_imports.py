#!/usr/bin/env python3
# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Test script to verify module structure and import paths after refactoring
# Acknowledgements: Claude for test design

"""Quick test to verify new module structure imports work correctly."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all main imports work."""
    print("Testing module imports...")
    
    # Test src-level imports
    print("  Testing src.__init__...")
    from src import __version__
    print(f"    [OK] src package (version {__version__})")
    
    # Test robot module
    print("  Testing src.robot...")
    from src.robot import create_robot, RobotClients, ManagedLease, ManagedEstop
    from src.robot import RobotActionMonitor, ObserverBridge, ObserverConfig
    print("    [OK] robot module")
    
    # Test perception module
    print("  Testing src.perception...")
    from src.perception import PersonDetection, PerceptionPipeline
    from src.perception import YoloDetector, Detection
    print("    [OK] perception module")
    
    # Test behavior module
    print("  Testing src.behavior...")
    from src.behavior import ComfortModel, BehaviorExecutor, BehaviorLabel, PerceptionInput
    print("    [OK] behavior module")
    
    # Test video module
    print("  Testing src.video...")
    from src.video import create_video_source, VideoSource
    from src.video import WebcamSource, SpotPTZImageClient, SpotPTZWebRTC
    print("    [OK] video module")
    
    # Test visualization module
    print("  Testing src.visualization...")
    from src.visualization import visualize_pipeline_frame, close_all_windows
    print("    [OK] visualization module")
    
    print("\n[OK] All imports successful!")
    return True

if __name__ == "__main__":
    try:
        success = test_imports()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[X] Import test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
