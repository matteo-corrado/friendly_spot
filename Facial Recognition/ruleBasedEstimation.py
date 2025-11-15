import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from datetime import datetime

# Initialize Mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseAnalyzer:
    """Analyzes pose landmarks and detects actions using rule-based heuristics"""
    
    def __init__(self, buffer_size=30):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.landmark_buffer = deque(maxlen=buffer_size)
        self.action_history = deque(maxlen=10)
        self.frame_count = 0
        
    def extract_landmarks(self, frame):
        """Extract pose landmarks from frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.pose_landmarks.landmark
            ])
            return landmarks, results
        
        return None, None
    
    def get_landmark(self, landmarks, index):
        """Safely get a landmark by index"""
        if landmarks is not None and len(landmarks) > index:
            return landmarks[index]
        return None
    
    # ============ HELPER FUNCTIONS ============
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        if point1 is None or point2 is None:
            return float('inf')
        return np.linalg.norm(point1 - point2)
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3 (in degrees)"""
        if p1 is None or p2 is None or p3 is None:
            return 0
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
        )
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def get_body_height(self, landmarks):
        """Get approximate body height"""
        nose = self.get_landmark(landmarks, 0)
        ankle = self.get_landmark(landmarks, 27)  # left ankle
        
        if nose is not None and ankle is not None:
            return abs(nose[1] - ankle[1])
        return 0
    
    def get_foot_distance(self, landmarks):
        """Get distance between feet"""
        left_ankle = self.get_landmark(landmarks, 27)
        right_ankle = self.get_landmark(landmarks, 28)
        
        return self.calculate_distance(left_ankle, right_ankle)
    
    def get_hand_elevation(self, landmarks):
        """Get average hand elevation relative to shoulders"""
        left_hand = self.get_landmark(landmarks, 15)
        right_hand = self.get_landmark(landmarks, 16)
        left_shoulder = self.get_landmark(landmarks, 11)
        right_shoulder = self.get_landmark(landmarks, 12)
        
        if all([left_hand, right_hand, left_shoulder, right_shoulder]):
            hand_y = (left_hand[1] + right_hand[1]) / 2
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            return shoulder_y - hand_y  # positive = hands above shoulders
        
        return 0
    
    def get_torso_angle(self, landmarks):
        """Get angle of torso relative to vertical"""
        shoulder_center = (
            self.get_landmark(landmarks, 11) + 
            self.get_landmark(landmarks, 12)
        ) / 2
        hip_center = (
            self.get_landmark(landmarks, 23) + 
            self.get_landmark(landmarks, 24)
        ) / 2
        
        if shoulder_center is not None and hip_center is not None:
            # Calculate angle from vertical
            dx = shoulder_center[0] - hip_center[0]
            dy = shoulder_center[1] - hip_center[1]
            angle = np.degrees(np.arctan2(dx, dy))
            return abs(angle)
        
        return 0
    
    def calculate_motion_speed(self, landmarks):
        """Calculate overall motion speed from buffer"""
        if len(self.landmark_buffer) < 2:
            return 0
        
        current = landmarks
        previous = self.landmark_buffer[-1]
        
        motion = np.linalg.norm(current - previous)
        return motion
    
    def get_knee_angles(self, landmarks):
        """Get left and right knee angles"""
        left_knee_angle = self.calculate_angle(
            self.get_landmark(landmarks, 23),  # left hip
            self.get_landmark(landmarks, 25),  # left knee
            self.get_landmark(landmarks, 27)   # left ankle
        )
        
        right_knee_angle = self.calculate_angle(
            self.get_landmark(landmarks, 24),  # right hip
            self.get_landmark(landmarks, 26),  # right knee
            self.get_landmark(landmarks, 28)   # right ankle
        )
        
        return left_knee_angle, right_knee_angle
    
    def get_elbow_angles(self, landmarks):
        """Get left and right elbow angles"""
        left_elbow_angle = self.calculate_angle(
            self.get_landmark(landmarks, 11),  # left shoulder
            self.get_landmark(landmarks, 13),  # left elbow
            self.get_landmark(landmarks, 15)   # left wrist
        )
        
        right_elbow_angle = self.calculate_angle(
            self.get_landmark(landmarks, 12),  # right shoulder
            self.get_landmark(landmarks, 14),  # right elbow
            self.get_landmark(landmarks, 16)   # right wrist
        )
        
        return left_elbow_angle, right_elbow_angle
    
    # ============ ACTION DETECTION FUNCTIONS ============
    
    def is_standing(self, landmarks):
        """Detect standing posture"""
        left_knee, right_knee = self.get_knee_angles(landmarks)
        torso_angle = self.get_torso_angle(landmarks)
        foot_distance = self.get_foot_distance(landmarks)
        
        # Knees mostly straight (> 140 degrees)
        knees_straight = left_knee > 140 and right_knee > 140
        
        # Torso relatively upright (< 20 degrees lean)
        torso_upright = torso_angle < 20
        
        # Feet somewhat apart
        feet_apart = foot_distance > 0.05
        
        return knees_straight and torso_upright and feet_apart
    
    def is_sitting(self, landmarks):
        """Detect sitting posture"""
        left_knee, right_knee = self.get_knee_angles(landmarks)
        torso_angle = self.get_torso_angle(landmarks)
        nose = self.get_landmark(landmarks, 0)
        hip_center = (
            self.get_landmark(landmarks, 23) + 
            self.get_landmark(landmarks, 24)
        ) / 2
        
        if nose is None or hip_center is None:
            return False
        
        # Knees bent (< 100 degrees)
        knees_bent = left_knee < 100 and right_knee < 100
        
        # Torso relatively upright
        torso_upright = torso_angle < 30
        
        # Head above hips
        head_above_hips = nose[1] < hip_center[1]
        
        return knees_bent and torso_upright and head_above_hips
    
    def is_lying_down(self, landmarks):
        """Detect lying down posture"""
        nose = self.get_landmark(landmarks, 0)
        hip_center = (
            self.get_landmark(landmarks, 23) + 
            self.get_landmark(landmarks, 24)
        ) / 2
        left_ankle = self.get_landmark(landmarks, 27)
        right_ankle = self.get_landmark(landmarks, 28)
        
        if any([nose is None, hip_center is None, left_ankle is None, right_ankle is None]):
            return False
        
        # Body is horizontal (small vertical spread)
        body_vertical_spread = abs(nose[1] - hip_center[1])
        
        # Large horizontal spread
        horizontal_spread = abs(left_ankle[0] - right_ankle[0])
        
        return body_vertical_spread < 0.15 and horizontal_spread > 0.2
    
    def is_walking(self, landmarks):
        """Detect walking motion"""
        if len(self.landmark_buffer) < 5:
            return False
        
        left_knee, right_knee = self.get_knee_angles(landmarks)
        
        # Knees flexing alternately (difference > 30 degrees)
        knee_difference = abs(left_knee - right_knee)
        
        # Moderate motion speed
        motion_speed = self.calculate_motion_speed(landmarks)
        
        # Torso somewhat upright
        torso_angle = self.get_torso_angle(landmarks)
        
        return (knee_difference > 30 and 
                motion_speed > 0.02 and 
                motion_speed < 0.15 and
                torso_angle < 30)
    
    def is_running(self, landmarks):
        """Detect running motion"""
        if len(self.landmark_buffer) < 5:
            return False
        
        left_knee, right_knee = self.get_knee_angles(landmarks)
        motion_speed = self.calculate_motion_speed(landmarks)
        
        # Knees significantly bent during running
        knees_bent = left_knee < 100 and right_knee < 100
        
        # High motion speed
        high_speed = motion_speed > 0.15
        
        return knees_bent and high_speed
    
    def is_jumping(self, landmarks):
        """Detect jumping motion"""
        if len(self.landmark_buffer) < 3:
            return False
        
        nose = self.get_landmark(landmarks, 0)
        left_ankle = self.get_landmark(landmarks, 27)
        right_ankle = self.get_landmark(landmarks, 28)
        
        if any([nose is None, left_ankle is None, right_ankle is None]):
            return False
        
        # Calculate vertical distance between nose and ankles
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2
        vertical_distance = nose[1] - ankle_y
        
        # High vertical distance indicates jumping
        return vertical_distance > 0.35
    
    def is_waving(self, landmarks):
        """Detect waving motion"""
        if len(self.landmark_buffer) < 10:
            return False
        
        left_hand = self.get_landmark(landmarks, 15)
        right_hand = self.get_landmark(landmarks, 16)
        left_shoulder = self.get_landmark(landmarks, 11)
        right_shoulder = self.get_landmark(landmarks, 12)
        
        if any([left_hand is None, right_hand is None, 
                left_shoulder is None, right_shoulder is None]):
            return False
        
        # Check if hands are above shoulders
        left_hand_high = left_hand[1] < left_shoulder[1] - 0.05
        right_hand_high = right_hand[1] < right_shoulder[1] - 0.05
        
        # Check for lateral motion (hand moving side to side)
        recent_landmarks = list(self.landmark_buffer)[-10:]
        hand_x_positions = [lm[15][0] for lm in recent_landmarks]
        hand_x_range = max(hand_x_positions) - min(hand_x_positions)
        
        lateral_motion = hand_x_range > 0.1
        
        hand_elevated = left_hand_high or right_hand_high
        
        return hand_elevated and lateral_motion
    
    def is_raising_hand(self, landmarks):
        """Detect hand raised above head"""
        left_hand = self.get_landmark(landmarks, 15)
        right_hand = self.get_landmark(landmarks, 16)
        nose = self.get_landmark(landmarks, 0)
        
        if any([left_hand is None, right_hand is None, nose is None]):
            return False
        
        # At least one hand above head
        left_above_head = left_hand[1] < nose[1] - 0.1
        right_above_head = right_hand[1] < nose[1] - 0.1
        
        return left_above_head or right_above_head
    
    def is_clapping(self, landmarks):
        """Detect clapping motion"""
        if len(self.landmark_buffer) < 5:
            return False
        
        left_hand = self.get_landmark(landmarks, 15)
        right_hand = self.get_landmark(landmarks, 16)
        
        if left_hand is None or right_hand is None:
            return False
        
        # Hands close together
        hand_distance = self.calculate_distance(left_hand, right_hand)
        hands_close = hand_distance < 0.1
        
        # Both hands at chest level or above
        left_shoulder = self.get_landmark(landmarks, 11)
        right_shoulder = self.get_landmark(landmarks, 12)
        
        if left_shoulder is None or right_shoulder is None:
            return False
        
        chest_level = (left_hand[1] < left_shoulder[1] and 
                       right_hand[1] < right_shoulder[1])
        
        return hands_close and chest_level
    
    def is_bending(self, landmarks):
        """Detect bending motion"""
        torso_angle = self.get_torso_angle(landmarks)
        nose = self.get_landmark(landmarks, 0)
        hip_center = (
            self.get_landmark(landmarks, 23) + 
            self.get_landmark(landmarks, 24)
        ) / 2
        
        if nose is None or hip_center is None:
            return False
        
        # Significant forward lean
        significant_lean = torso_angle > 20
        
        # Head lower than typical
        vertical_distance = abs(nose[1] - hip_center[1])
        head_lower = vertical_distance < 0.25
        
        return significant_lean and head_lower
    
    def is_arms_crossed(self, landmarks):
        """Detect arms crossed"""
        left_shoulder = self.get_landmark(landmarks, 11)
        right_shoulder = self.get_landmark(landmarks, 12)
        left_hand = self.get_landmark(landmarks, 15)
        right_hand = self.get_landmark(landmarks, 16)
        
        if any([left_shoulder is None, right_shoulder is None,
                left_hand is None, right_hand is None]):
            return False
        
        # Left hand on right side, right hand on left side
        left_hand_right = left_hand[0] > right_shoulder[0]
        right_hand_left = right_hand[0] < left_shoulder[0]
        
        # Hands roughly at same height as shoulders
        similar_height = abs(left_hand[1] - right_hand[1]) < 0.1
        
        return (left_hand_right or right_hand_left) and similar_height
    
    # ============ MAIN DETECTION FUNCTION ============
    
    def detect_action(self, landmarks):
        """Detect current action from landmarks"""
        
        # Add to buffer
        self.landmark_buffer.append(landmarks)
        
        # Check actions in order of likelihood/confidence
        if self.is_jumping(landmarks):
            return "jumping", 0.95
        
        elif self.is_running(landmarks):
            return "running", 0.9
        
        elif self.is_walking(landmarks):
            return "walking", 0.85
        
        elif self.is_lying_down(landmarks):
            return "lying_down", 0.92
        
        elif self.is_sitting(landmarks):
            return "sitting", 0.88
        
        elif self.is_clapping(landmarks):
            return "clapping", 0.8
        
        elif self.is_waving(landmarks):
            return "waving", 0.82
        
        elif self.is_raising_hand(landmarks):
            return "raising_hand", 0.78
        
        elif self.is_bending(landmarks):
            return "bending", 0.75
        
        elif self.is_arms_crossed(landmarks):
            return "arms_crossed", 0.7
        
        elif self.is_standing(landmarks):
            return "standing", 0.9
        
        else:
            return "unknown", 0.5


class PoseVisualizerApp:
    """Main application for visualizing pose and detecting actions"""
    
    def __init__(self):
        self.analyzer = PoseAnalyzer()
        self.current_action = "initializing"
        self.current_confidence = 0
        self.frame_width = 1280
        self.frame_height = 720
        
    def draw_info(self, frame, action, confidence):
        """Draw action information on frame"""
        # Draw background for text
        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
        
        # Draw action text
        action_text = f"Action: {action.upper()}"
        confidence_text = f"Confidence: {confidence:.1%}"
        timestamp_text = f"Time: {datetime.now().strftime('%H:%M:%S')}"
        
        cv2.putText(frame, action_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, confidence_text, (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, timestamp_text, (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {self.analyzer.frame_count}",
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def run(self, video_source=0):
        """Run the pose action detection on video stream"""
        cap = cv2.VideoCapture(video_source)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting Pose Action Detection...")
        print("Press 'q' to quit")
        print("Press 's' to save frame")
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
                action, confidence = self.analyzer.detect_action(landmarks)
                
                self.current_action = action
                self.current_confidence = confidence
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=2,
                        circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(
                        color=(0, 0, 255),
                        thickness=2
                    )
                )
            
            # Draw info
            self.draw_info(frame, self.current_action, self.current_confidence)
            
            # Display frame
            cv2.imshow('Pose Action Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                filename = f"pose_frame_{self.analyzer.frame_count}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved frame to {filename}")
            
            # Print periodic updates
            if self.analyzer.frame_count % 30 == 0:
                if landmarks is not None:
                    print(f"Frame {self.analyzer.frame_count}: {action} ({confidence:.1%})")
                else:
                    print(f"Frame {self.analyzer.frame_count}")
                    
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed")


def main():
    """Main entry point"""
    app = PoseVisualizerApp()
    
    # Use 0 for webcam, or provide a video file path
    app.run(video_source="test.MOV")


if __name__ == "__main__":
    main()