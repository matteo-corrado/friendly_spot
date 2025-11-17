# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/15/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Use Mediapipe for Pose Landmark Estimation.  Based on these results, predict activity/position 
# of person based on a rule-based metric.  Future progress includes converting rule-based
# approach to an ML-based approach for higher accuracy.
# Acknowledgements: Claude was used for outlining the structure of a rule-based metric for activity estimation.
# Mediapipe code is based on the documentation and examples openly available on CoLab's website.

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from datetime import datetime

# MediaPipe Landmark Indices
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX_FINGER = 19
RIGHT_INDEX_FINGER = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT = 31
RIGHT_FOOT = 32

# Model Parameters
MIN_CONFIDENCE = 0.5
MAX_ACTION_HISTORY = 10
STATIC_IMAGES = False
SMOOTH_LANDMARKS = True
COMPLEXITY = 1

# Threshold Values for Rule-Based Pose Configurations
MIN_BUFFER_MOTION = 3
MIN_WAVING = 8
STRAIGHT_KNEE = 120
UPRIGHT_TORSO = 30
MIN_WALK_SPEED = 0.02
MAX_WALK_SPEED = 0.15
SIMILAR_MEASUREMENT = 0.1 # used for multiple landmarks, comparing left to right for similar values
MIN_WAVE_MOTION = 0.1
MIN_KNEE_ANGLE_DIFFERENCE_WALKING = 30
MIN_HEAD_DIP_FOR_BEND = 0.25

# Video Parameters
FPS = 30
WIDTH = 1280
HEIGHT = 720

# Initialize Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseAnalyzer:
    
    def __init__(self, buffer_size=FPS):
        self.pose = mp_pose.Pose(static_image_mode=STATIC_IMAGES, model_complexity=COMPLEXITY, smooth_landmarks=SMOOTH_LANDMARKS, min_detection_confidence=MIN_CONFIDENCE, min_tracking_confidence=MIN_CONFIDENCE)
        self.landmark_buffer = deque(maxlen=buffer_size)
        self.action_history = deque(maxlen=MAX_ACTION_HISTORY)
        self.frame_count = 0
        
    def extract_landmarks(self, frame):
        # Frome frame, extract the pose landmarks with Mediapipe
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            return landmarks, results
        
        return None, None
    
    def get_landmark(self, landmarks, index):
        # Grab a specific landmark
        
        if landmarks is not None and len(landmarks) > index:
            return landmarks[index]
        return None
    
    def calculate_distance(self, point1, point2):
        # Calculates L2 distance between two points
        
        if point1 is None or point2 is None:
            return float('inf')
        return np.linalg.norm(point1 - point2)
    
    def calculate_angle(self, p1, p2, p3):
        # Calculates the angle at p2 formed by p1-p2-p3 (in degrees)
        
        if p1 is None or p2 is None or p3 is None:
            return 0
        
        # Creates two vectors to define angle
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Use definition of dot product to find the angle between the two vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6) # small nudge by 1e-6 for better performance
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def get_foot_distance(self, landmarks):
        # Get distance between feet
        left_ankle = self.get_landmark(landmarks, LEFT_ANKLE)
        right_ankle = self.get_landmark(landmarks, RIGHT_ANKLE)
        
        return self.calculate_distance(left_ankle, right_ankle)
    
    def get_hand_elevation(self, landmarks):
        # Get average hand elevation relative to shoulders
        
        left_hand = self.get_landmark(landmarks, LEFT_WRIST)
        right_hand = self.get_landmark(landmarks, RIGHT_WRIST)
        left_shoulder = self.get_landmark(landmarks, LEFT_SHOULDER)
        right_shoulder = self.get_landmark(landmarks, RIGHT_SHOULDER)
        
        if all([left_hand, right_hand, left_shoulder, right_shoulder]):
            hand_y = (left_hand[1] + right_hand[1]) / 2
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            return shoulder_y - hand_y  # positive = hands above shoulders
        
        return 0
    
    def get_torso_angle(self, landmarks):
        # Get angle of torso relative to vertical
        
        shoulder_center = (self.get_landmark(landmarks, LEFT_SHOULDER) + self.get_landmark(landmarks, RIGHT_SHOULDER)) / 2
        hip_center = (self.get_landmark(landmarks, LEFT_HIP) + self.get_landmark(landmarks, RIGHT_HIP)) / 2
        
        if shoulder_center is not None and hip_center is not None:
            # Calculate angle from vertical
            dx = shoulder_center[0] - hip_center[0]
            dy = shoulder_center[1] - hip_center[1]
            angle = np.degrees(np.arctan2(dx, dy))
            return abs(angle)
        
        return 0
    
    def calculate_motion_speed(self, landmarks):
        # Calculates overall motion speed from buffer
        
        if len(self.landmark_buffer) < MIN_BUFFER_MOTION:
            return 0
        
        current = landmarks
        previous = self.landmark_buffer[-1]
        
        motion = np.linalg.norm(current - previous)
        return motion
    
    def get_knee_angles(self, landmarks):
        # Get left and right knee angles
        
        left_knee_angle = self.calculate_angle(self.get_landmark(landmarks, LEFT_HIP), self.get_landmark(landmarks, LEFT_KNEE), self.get_landmark(landmarks, LEFT_ANKLE))
        right_knee_angle = self.calculate_angle(self.get_landmark(landmarks, RIGHT_HIP), self.get_landmark(landmarks, RIGHT_KNEE), self.get_landmark(landmarks, RIGHT_ANKLE))
        
        return left_knee_angle, right_knee_angle
        
    def is_standing(self, landmarks):
        # Recognize standing as knees straight, torso upright, and feet some distance apart
        
        left_knee, right_knee = self.get_knee_angles(landmarks)
        torso_angle = self.get_torso_angle(landmarks)
        foot_distance = self.get_foot_distance(landmarks)
        
        knees_straight = left_knee > STRAIGHT_KNEE and right_knee > STRAIGHT_KNEE
        torso_upright = torso_angle < UPRIGHT_TORSO
        feet_apart = foot_distance > SIMILAR_MEASUREMENT
        
        return knees_straight and torso_upright and feet_apart
    
    def is_walking(self, landmarks):
        # Walking based on knee angles, at different angles, torso still upright, and some speed of motion
       
        if len(self.landmark_buffer) < MIN_BUFFER_MOTION:
            return False
        
        left_knee, right_knee = self.get_knee_angles(landmarks)
        
        knee_difference = abs(left_knee - right_knee)
        motion_speed = self.calculate_motion_speed(landmarks)
        torso_angle = self.get_torso_angle(landmarks)
        
        return (knee_difference > MIN_KNEE_ANGLE_DIFFERENCE_WALKING) and (motion_speed > MIN_WALK_SPEED) and (motion_speed < MAX_WALK_SPEED) and (torso_angle < UPRIGHT_TORSO)
    
    def is_running(self, landmarks):
        # When running, siginificant knee bend and motion of the body
        
        if len(self.landmark_buffer) < MIN_BUFFER_MOTION:
            return False
        
        left_knee, right_knee = self.get_knee_angles(landmarks)
        motion_speed = self.calculate_motion_speed(landmarks)
        
        knees_bent = left_knee < STRAIGHT_KNEE and right_knee < STRAIGHT_KNEE
        high_speed = motion_speed > MAX_WALK_SPEED
        
        return knees_bent and high_speed
    
    def is_waving(self, landmarks):
        # Waving has one hand raised, with lateral motion
        
        if len(self.landmark_buffer) < MIN_BUFFER_MOTION:
            return False
        
        left_hand = self.get_landmark(landmarks, LEFT_WRIST)
        right_hand = self.get_landmark(landmarks, RIGHT_WRIST)
        left_shoulder = self.get_landmark(landmarks, LEFT_SHOULDER)
        right_shoulder = self.get_landmark(landmarks, RIGHT_SHOULDER)
        
        if any([left_hand is None, right_hand is None, left_shoulder is None, right_shoulder is None]):
            return False
        
        # Check if hands are above shoulders
        left_hand_high = left_hand[1] < left_shoulder[1]
        right_hand_high = right_hand[1] < right_shoulder[1]
        
        # Check for lateral motion (hand moving side to side)
        recent_landmarks = list(self.landmark_buffer)[-1*MIN_WAVING:]
        hand_x_positions_left = [lm[LEFT_WRIST][0] for lm in recent_landmarks]
        hand_x_range_left = max(hand_x_positions_left) - min(hand_x_positions_left)
        hand_x_positions_right = [lm[RIGHT_WRIST][0] for lm in recent_landmarks]
        hand_x_range_right = max(hand_x_positions_right) - min(hand_x_positions_right)
        
        lateral_motion_left = hand_x_range_left > MIN_WAVE_MOTION
        lateral_motion_right = hand_x_range_right > MIN_WAVE_MOTION
                
        return (left_hand_high and lateral_motion_left) or (right_hand_high and lateral_motion_right)
    
    def is_raising_hand(self, landmarks):
        # Generic raising hand is simply hand above head
        
        left_hand = self.get_landmark(landmarks, LEFT_WRIST)
        right_hand = self.get_landmark(landmarks, RIGHT_WRIST)
        nose = self.get_landmark(landmarks, NOSE)
        
        if any([left_hand is None, right_hand is None, nose is None]):
            return False
        
        # At least one hand above head
        left_above_head = left_hand[1] < nose[1]
        right_above_head = right_hand[1] < nose[1]
        
        return left_above_head or right_above_head
    
    def is_clapping(self, landmarks):
        # Clapping is hands close together, and relatively high
        
        if len(self.landmark_buffer) < MIN_BUFFER_MOTION:
            return False
        
        left_hand = self.get_landmark(landmarks, LEFT_WRIST)
        right_hand = self.get_landmark(landmarks, RIGHT_WRIST)
        
        if left_hand is None or right_hand is None:
            return False
        
        # Hands close together
        hand_distance = self.calculate_distance(left_hand, right_hand)
        hands_close = hand_distance < SIMILAR_MEASUREMENT
        
        # Both hands at chest level or above
        left_shoulder = self.get_landmark(landmarks, LEFT_SHOULDER)
        right_shoulder = self.get_landmark(landmarks, RIGHT_SHOULDER)
        
        if left_shoulder is None or right_shoulder is None:
            return False
        
        chest_level = (left_hand[1] < left_shoulder[1]) and (right_hand[1] < right_shoulder[1])
        
        return hands_close and chest_level
    
    def is_bending(self, landmarks):
        # Bending motion is torso evidently leaning, and the head (nose) has lowered relative to hips (can  be above or below hips)
        
        torso_angle = self.get_torso_angle(landmarks)
        nose = self.get_landmark(landmarks, NOSE)
        hip_center = (self.get_landmark(landmarks, LEFT_HIP) + self.get_landmark(landmarks, RIGHT_HIP)) / 2
        
        if nose is None or hip_center is None:
            return False
        
        # Significant forward lean
        significant_lean = torso_angle > UPRIGHT_TORSO
        
        # Head lower than typical
        vertical_distance = abs(nose[1] - hip_center[1])
        head_lower = vertical_distance < MIN_HEAD_DIP_FOR_BEND
        
        return significant_lean and head_lower
    
    def is_arms_crossed(self, landmarks):
        # Arms crossed typically has hands near the shoulders, and across the body, so across the position of the opposite shoulder
        
        left_shoulder = self.get_landmark(landmarks, LEFT_SHOULDER)
        right_shoulder = self.get_landmark(landmarks, RIGHT_SHOULDER)
        left_hand = self.get_landmark(landmarks, LEFT_WRIST)
        right_hand = self.get_landmark(landmarks, RIGHT_WRIST)
        
        if any([left_shoulder is None, right_shoulder is None, left_hand is None, right_hand is None]):
            return False
        
        # Left hand on right side, right hand on left side
        left_hand_right = left_hand[0] > right_shoulder[0]
        right_hand_left = right_hand[0] < left_shoulder[0]
        
        # Hands roughly at same height as shoulders
        similar_height = abs(left_hand[1] - right_hand[1]) < SIMILAR_MEASUREMENT
        
        return (left_hand_right or right_hand_left) and similar_height
        
    def detect_action(self, landmarks):
        # Determine the current action
        
        # Add to buffer
        self.landmark_buffer.append(landmarks)
        
        # Order is somewhat arbitrary, but also based on higher priority in instances where rules sets overlap
        # IE waving a hand above your head should be considered waving, not raising_hand
        if self.is_running(landmarks):
            return "running"
        elif self.is_walking(landmarks):
            return "walking"
        elif self.is_clapping(landmarks):
            return "clapping"
        elif self.is_waving(landmarks):
            return "waving"
        elif self.is_raising_hand(landmarks):
            return "raising_hand" 
        elif self.is_bending(landmarks):
            return "bending"
        elif self.is_arms_crossed(landmarks):
            return "arms_crossed"
        elif self.is_standing(landmarks):
            return "standing"
        else:
            return "unknown"


class PoseVisualizerApp:
    # Main driver for pose estimation and detecting actions
    
    def __init__(self):
        self.analyzer = PoseAnalyzer()
        self.current_action = "initializing"
        self.frame_width = WIDTH
        self.frame_height = HEIGHT
    
    def run(self, video_source=0):
        # Runs the pose action detection on video stream
        cap = cv2.VideoCapture(video_source)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        print("Starting Pose Action Detection...")
        print()
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("Error reading frame")
                break
            
            self.analyzer.frame_count += 1
            
            # Extract landmarks
            landmarks, results = self.analyzer.extract_landmarks(frame)
            
            if landmarks is not None:
                # Detect action
                action = self.analyzer.detect_action(landmarks)
                self.current_action = action
            
            # Output the estimated action
            if landmarks is not None:
                print(f"Frame {self.analyzer.frame_count}: {action}")
            else:
                print(f"Frame {self.analyzer.frame_count}")
                    
        cap.release()
        print("Estimation Complete")


def main():
    """Main entry point"""
    app = PoseVisualizerApp()
    
    # Source changes necessary for Spot
    app.run(video_source="test.MOV")


if __name__ == "__main__":
    main()