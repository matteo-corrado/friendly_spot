import os
import cv2
import time
import importlib.util
import importlib.machinery
import numpy as np

# Load workspace classes from files (handles folder name with space)
def load_module_from_path(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module

BASE = os.path.dirname(__file__)
pose_mod = load_module_from_path(
    "streamlined_pose",
    os.path.join(BASE, "Facial Recognition", "streamlinedRuleBasedEstimation.py"),
)
face_mod = load_module_from_path(
    "streamlined_face",
    os.path.join(BASE, "Facial Recognition", "streamlinedCombinedMemoryAndEmotion.py"),
)

from behavior_planner import PerceptionInput  # [`PerceptionInput`](behavior_planner.py)

PoseAnalyzer = pose_mod.PoseAnalyzer  # [`PoseAnalyzer`](Facial Recognition/streamlinedRuleBasedEstimation.py)
FaceRecognizer = face_mod.FaceRecognizer  # [`FaceRecognizer`](Facial Recognition/streamlinedCombinedMemoryAndEmotion.py)

# Simple gesture detector using MediaPipe Hands (keeps dependency local)
import mediapipe as mp
mp_hands = mp.solutions.hands

def detect_gesture_from_frame(frame, hands_detector):
    """
    Minimal hand/gesture detection:
      - returns 'none' if no hand
      - otherwise returns 'open_hand' | 'closed_fist' | 'unknown' (very coarse)
    This is a simple placeholder; replace with your gesture task model if available.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(img_rgb)
    if not res.multi_hand_landmarks:
        return "none"
    # crude heuristic: distance between index-tip and thumb-tip
    lm = res.multi_hand_landmarks[0].landmark
    idx = np.array([lm[8].x, lm[8].y])
    thumb = np.array([lm[4].x, lm[4].y])
    dist = np.linalg.norm(idx - thumb)
    # thresholds are heuristic (landmarks normalized)
    if dist > 0.06:
        return "open_hand"
    if dist < 0.03:
        return "closed_fist"
    return "unknown"

def estimate_distance_m(landmarks, frame_height, assumed_person_height_m=1.7):
    """
    Heuristic distance estimator using normalized landmark vertical span.
    landmarks are normalized (0..1). Returns None when not available.
    """
    if landmarks is None:
        return None
    ys = landmarks[:, 1]
    pixel_height = (ys.max() - ys.min()) * frame_height
    if pixel_height <= 5:
        return None
    # approximate focal length in pixels (heuristic)
    focal_px = frame_height * 1.2
    distance_m = (assumed_person_height_m * focal_px) / max(1.0, pixel_height)
    return float(distance_m)

class PerceptionPipeline:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        self.pose = PoseAnalyzer()
        self.face = FaceRecognizer()
        # train face recognizer on IMAGE_DIRECTORY (from FaceRecognizer module)
        image_dir = getattr(face_mod, "IMAGE_DIRECTORY", "dataset")
        try:
            self.face.initialize_facial_data(image_dir)
            self.face.train_from_directory(image_dir)
        except Exception:
            # ignore training failures for now
            pass
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4)

    def read_perception(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame_h, frame_w = frame.shape[:2]

        # Pose -> landmarks & action
        landmarks, results = self.pose.extract_landmarks(frame)
        if landmarks is not None:
            pose_label = self.pose.detect_action(landmarks)
            current_action = pose_label  # use same as pose for now
            distance_m = estimate_distance_m(landmarks, frame_h)
        else:
            pose_label = "unknown"
            current_action = "unknown"
            distance_m = None

        # Gesture
        gesture_label = detect_gesture_from_frame(frame, self.hands)

        # Face + Emotion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face.face_cascade.detectMultiScale(gray,
                                                       scaleFactor=getattr(face_mod, "IMAGE_SCALE_FACTOR", 1.1),
                                                       minNeighbors=getattr(face_mod, "MIN_NEIGHBORS", 5),
                                                       minSize=(getattr(face_mod, "MIN_FACE_DIMENSION", 30),
                                                                getattr(face_mod, "MIN_FACE_DIMENSION", 30)))
        if len(faces) == 0:
            face_label = "unknown"
            emotion_label = "neutral"
        else:
            # take most confident face (largest area)
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            face_roi_gray = gray[y:y + h, x:x + w]
            try:
                label, confidence = self.face.face_recognizer.predict(face_roi_gray)
                if confidence < getattr(face_mod, "CONFIDENCE_THRESHOLD", 70):
                    face_label = self.face.label_names.get(label, "unknown")
                else:
                    face_label = "unknown"
            except Exception:
                face_label = "unknown"
                confidence = 100
            # emotion using DeepFace as in face module (best-effort)
            try:
                # re-use DeepFace imported in the face module if present
                DeepFace = getattr(face_mod, "DeepFace")
                emotion = DeepFace.analyze(frame[y:y+h, x:x+w], actions=["emotion"], enforce_detection=False)
                emotion_label = emotion[0]["dominant_emotion"]
            except Exception:
                emotion_label = "neutral"

        perception = PerceptionInput(
            current_action=current_action,
            distance_m=distance_m,
            face_label=face_label,
            emotion_label=emotion_label,
            pose_label=pose_label,
            gesture_label=gesture_label,
        )
        return perception

    def stream(self, rate_hz=10):
        try:
            while True:
                p = self.read_perception()
                if p is None:
                    break
                print(p)
                time.sleep(1.0 / rate_hz)
        finally:
            self.cap.release()
            self.hands.close()

if __name__ == "__main__":
    # Example usage: webcam (0) or video file path
    pipeline = PerceptionPipeline(video_source=0)
    pipeline.stream(rate_hz=5)