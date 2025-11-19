# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Recognize up to two hands, with gestures, from the current camera frame

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Sets parameters for the recognition model with Mediapipe
MODEL_PATH = 'gesture_recognizer.task'
MAX_NUM_HANDS = 2
MIN_CONFIDENCE = 0.4

class GestureRecognizer:
    """Recognizes hand gestures from video frames using MediaPipe."""
    
    def __init__(self, model_path=MODEL_PATH, max_num_hands=MAX_NUM_HANDS, min_confidence=MIN_CONFIDENCE):
        """Initialize the gesture recognizer.
        
        Args:
            model_path: Path to the gesture_recognizer.task model file
            max_num_hands: Maximum number of hands to detect (default 2)
            min_confidence: Minimum confidence threshold for hand detection (default 0.4)
        """
        self.max_num_hands = max_num_hands
        self.min_confidence = min_confidence
        
        # MediaPipe on Windows has broken path resolution - it resolves relative paths
        # from site-packages instead of CWD. Workaround: load model as bytes
        import os
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Gesture model not found at {model_path}")
        
        # Read model file into memory
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # Use model_asset_buffer instead of model_asset_path to bypass broken path resolution
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.GestureRecognizerOptions(base_options=base_options)
        options.num_hands = max_num_hands
        options.min_hand_detection_confidence = min_confidence
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
    
    def recognize_gestures(self, frame):
        """Recognize gestures from a BGR video frame.
        
        Args:
            frame: BGR image as numpy array (from cv2.VideoCapture)
        
        Returns:
            str: Gesture label describing detected gestures, e.g.:
                 "Left: Thumb_Up, Right: Open_Palm"
                 "Left: Pointing_Up"
                 "none" if no hands detected
        """
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run gesture recognition
        recognition_result = self.recognizer.recognize(mp_image)
        
        # Parse results into a readable label
        if recognition_result.gestures:
            gesture_parts = []
            for index in range(len(recognition_result.gestures)):
                predicted_gesture = recognition_result.gestures[index][0].category_name
                predicted_hand = recognition_result.handedness[index][0].category_name
                gesture_parts.append(f"{predicted_hand}: {predicted_gesture}")
            
            return ", ".join(gesture_parts)
        else:
            return "none"
    
    def close(self):
        """Release resources (if needed)."""
        # MediaPipe GestureRecognizer doesn't require explicit cleanup
        pass


# Standalone test code (preserved from original)
if __name__ == "__main__":
    import sys
    
    # Test with image file if provided
    if len(sys.argv) > 1:
        IMAGE_FILENAME = sys.argv[1]
        
        recognizer = GestureRecognizer()
        
        # Load image and convert to frame format
        frame = cv2.imread(IMAGE_FILENAME)
        if frame is None:
            print(f"Error: Could not load image {IMAGE_FILENAME}")
            sys.exit(1)
        
        result = recognizer.recognize_gestures(frame)
        print(f"Detected gestures: {result}")
    else:
        print("Usage: python streamlinedGestureExtraction.py <image_file>")
        print("Or import GestureRecognizer class for pipeline integration")