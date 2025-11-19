"""
Pre-download models for offline environments.

This script triggers downloads of DeepFace emotion models and YOLO weights
so they're cached locally before running the pipeline in restricted networks.

Usage:
    python scripts/download_models.py
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def download_deepface_models():
    """Download DeepFace emotion detection models."""
    try:
        from deepface import DeepFace
        logger.info("Downloading DeepFace Emotion model...")
        DeepFace.build_model("Emotion")
        logger.info("DeepFace Emotion model downloaded")
    except Exception as e:
        logger.error(f"Failed to download DeepFace models: {e}")
        return False
    return True


def download_yolo_models():
    """Download YOLO person detection model."""
    try:
        from ultralytics import YOLO
        logger.info("Downloading YOLOv8x model...")
        model = YOLO("yolov8x.pt")
        logger.info("YOLOv8x model downloaded")
    except Exception as e:
        logger.error(f"Failed to download YOLO model: {e}")
        return False
    return True


def verify_mediapipe():
    """Verify MediaPipe hands/pose models are accessible."""
    try:
        import mediapipe as mp
        logger.info("Verifying MediaPipe models...")
        hands = mp.solutions.hands.Hands(static_image_mode=True)
        pose = mp.solutions.pose.Pose(static_image_mode=True)
        hands.close()
        pose.close()
        logger.info("MediaPipe models verified")
    except Exception as e:
        logger.error(f"MediaPipe verification failed: {e}")
        return False
    return True


def main():
    logger.info("Starting model downloads...")
    logger.info("This may take several minutes and download ~500MB of data.\n")
    
    success = True
    success &= download_deepface_models()
    success &= download_yolo_models()
    success &= verify_mediapipe()
    
    if success:
        logger.info("\nAll models downloaded and verified successfully!")
        logger.info("You can now run the pipeline offline.")
        return 0
    else:
        logger.error("\nSome models failed to download. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
