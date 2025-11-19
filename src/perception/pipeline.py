"""Streamlined perception pipeline using existing recognition modules efficiently.

Integrates pose, face, emotion, and gesture detection with minimal overhead.
"""
import os
import sys
import cv2
import time
import importlib.util
import importlib.machinery
import numpy as np
import logging
from typing import Optional

from ..video.sources import VideoSource, create_video_source
from .detection_types import PersonDetection, validate_depth_against_heuristic, estimate_distance_from_bbox

logger = logging.getLogger(__name__)

# Load modules from Facial Recognition folder
def load_module_from_path(name, path):
    """Load Python module from absolute file path."""
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pose_mod = load_module_from_path(
    "streamlined_pose",
    os.path.join(BASE, "Facial Recognition", "streamlinedRuleBasedEstimation.py"),
)
face_mod = load_module_from_path(
    "streamlined_face",
    os.path.join(BASE, "Facial Recognition", "streamlinedCombinedMemoryAndEmotion.py"),
)
gesture_mod = load_module_from_path(
    "streamlined_gesture",
    os.path.join(BASE, "Facial Recognition", "streamlinedGestureExtraction.py"),
)

from ..behavior.planner import PerceptionInput

# Extract classes
PoseAnalyzer = pose_mod.PoseAnalyzer
FaceRecognizer = face_mod.FaceRecognizer
GestureRecognizer = gesture_mod.GestureRecognizer

def estimate_distance_from_landmarks(landmarks, frame_height, assumed_person_height_m=1.7):
    """Estimate distance using pose landmark vertical span (simple heuristic)."""
    if landmarks is None:
        return None
    ys = landmarks[:, 1]
    pixel_height = (ys.max() - ys.min()) * frame_height
    if pixel_height <= 5:
        return None
    focal_px = frame_height * 1.2
    distance_m = (assumed_person_height_m * focal_px) / max(1.0, pixel_height)
    return float(distance_m)


class PerceptionPipeline:
    """Streamlined perception pipeline using existing recognition modules."""
    
    def __init__(self, video_source: VideoSource, robot=None):
        """Initialize pipeline with video source and optional robot connection.
        
        Args:
            video_source: Video source (webcam, ImageClient, WebRTC)
            robot: Optional Robot instance for action monitoring
        """
        self.video_source = video_source
        self.robot = robot
        
        # Configure TensorFlow for DeepFace
        self._configure_tensorflow()
        
        # Initialize recognition modules
        self.pose = PoseAnalyzer()
        self.face = FaceRecognizer()
        
        # Train face recognizer from dataset
        image_dir = getattr(face_mod, "IMAGE_DIRECTORY", "dataset")
        try:
            self.face.initialize_facial_data(image_dir)
            self.face.train_from_directory(image_dir)
        except Exception as e:
            logger.warning(f"Face training skipped: {e}")
        
        # Initialize gesture recognizer
        gesture_model_path = os.path.join(BASE, "Facial Recognition", "gesture_recognizer.task")
        try:
            self.gesture = GestureRecognizer(model_path=gesture_model_path)
        except Exception as e:
            logger.warning(f"Gesture recognizer unavailable: {e}")
            self.gesture = None
        
        # Pre-warm DeepFace
        self._warmup_deepface()
        
        # Lazy-init robot action monitor
        self._action_monitor = None
    
    def _configure_tensorflow(self):
        """Configure TensorFlow to suppress warnings and use CPU."""
        try:
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
        except ImportError:
            pass  # TensorFlow not installed
    
    def _warmup_deepface(self):
        """Pre-load DeepFace model to avoid first-frame delay."""
        try:
            from deepface import DeepFace
            logger.info("Warming up DeepFace...")
            dummy = np.zeros((48, 48, 3), dtype=np.uint8)
            DeepFace.analyze(dummy, actions=['emotion'], enforce_detection=False, 
                           detector_backend='skip', silent=True)
            logger.info("DeepFace ready")
        except Exception:
            pass  # Model will load on first real use

    def read_perception(self, person_detection: Optional[PersonDetection] = None):
        """Extract all perception data from frame in one streamlined pass.
        
        Args:
            person_detection: Optional PersonDetection with pre-computed distance/depth
        
        Returns:
            PerceptionInput with all perception data, or None if frame read fails
        """
        # Get frame (from person_detection or video source)
        if person_detection and person_detection.frame is not None:
            frame = person_detection.frame
            depth_frame = person_detection.depth_frame
        else:
            ret, frame, depth_frame = self.video_source.read()
            if not ret or frame is None:
                return None
        
        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. POSE: Extract landmarks and classify action
        landmarks, _ = self.pose.extract_landmarks(frame)
        if landmarks is not None:
            pose_label = self.pose.detect_action(landmarks)
            
            # Distance estimation priority:
            # 1) person_detection.distance_m (pre-validated)
            # 2) depth_frame extraction with validation
            # 3) landmark-based heuristic
            if person_detection and person_detection.distance_m:
                distance_m = person_detection.distance_m
            elif depth_frame is not None:
                # Extract bbox from landmarks for depth sampling
                xs, ys = landmarks[:, 0] * frame_w, landmarks[:, 1] * frame_h
                bbox_xywh = (int(xs.min()), int(ys.min()), 
                           int(xs.max() - xs.min()), int(ys.max() - ys.min()))
                
                try:
                    from people_observer.tracker import estimate_detection_depth_m
                    from people_observer.detection import Detection
                    det = Detection(source="ptz", bbox_xywh=bbox_xywh, conf=1.0, mask=None)
                    depth_dist = estimate_detection_depth_m(depth_frame, det, use_mask=False)
                    
                    if depth_dist and validate_depth_against_heuristic(depth_dist, bbox_xywh, frame_h, 2.5):
                        distance_m = depth_dist
                    else:
                        distance_m = estimate_distance_from_bbox(bbox_xywh, frame_h)
                except ImportError:
                    distance_m = estimate_distance_from_landmarks(landmarks, frame_h)
            else:
                distance_m = estimate_distance_from_landmarks(landmarks, frame_h)
        else:
            pose_label = "unknown"
            distance_m = person_detection.distance_m if person_detection else None
        
        # 2. ROBOT ACTION: Get current robot state (lazy-init monitor)
        current_action = "idle"
        if self.robot:
            try:
                if self._action_monitor is None:
                    from robot_action_monitor import RobotActionMonitor
                    self._action_monitor = RobotActionMonitor(self.robot)
                current_action = self._action_monitor.get_current_action(distance_m)
            except Exception as e:
                logger.debug(f"Robot action failed: {e}")
        
        # 3. FACE + EMOTION: Use streamlinedCombinedMemoryAndEmotion approach
        face_label, emotion_label, emotion_scores, face_bbox = self._recognize_face_and_emotion(frame, gray)
        
        # 4. GESTURE: Hand gesture recognition
        if self.gesture:
            try:
                gesture_label = self.gesture.recognize_gestures(frame)
            except Exception:
                gesture_label = "none"
        else:
            gesture_label = "none"
        
        # 5. PTZ: Extract pan/tilt if available
        ptz_pan = getattr(person_detection, 'ptz_pan', None) if person_detection else None
        ptz_tilt = getattr(person_detection, 'ptz_tilt', None) if person_detection else None
        
        # Build and return perception input
        return PerceptionInput(
            current_action=current_action,
            distance_m=distance_m,
            face_label=face_label,
            emotion_label=emotion_label,
            pose_label=pose_label,
            gesture_label=gesture_label,
            face_bbox=face_bbox,
            pose_landmarks=landmarks,
            emotion_scores=emotion_scores,
            frame=frame,
            ptz_pan=ptz_pan,
            ptz_tilt=ptz_tilt
        )

    def _recognize_face_and_emotion(self, frame, gray):
        """Combined face recognition and emotion detection using streamlined approach.
        
        Uses the same face detection bbox for both face recognition (on gray)
        and emotion detection (on RGB), following streamlinedCombinedMemoryAndEmotion.py pattern.
        
        Returns:
            tuple: (face_label, emotion_label, emotion_scores, face_bbox)
        """
        # Detect faces using cascade from FaceRecognizer
        faces = self.face.face_cascade.detectMultiScale(
            gray,
            scaleFactor=getattr(face_mod, "IMAGE_SCALE_FACTOR", 1.1),
            minNeighbors=getattr(face_mod, "MIN_NEIGHBORS", 5),
            minSize=(getattr(face_mod, "MIN_FACE_DIMENSION", 30), 
                    getattr(face_mod, "MIN_FACE_DIMENSION", 30))
        )
        
        if len(faces) == 0:
            return "unknown", "neutral", None, None
        
        # Use largest face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        face_bbox = (x, y, w, h)
        
        # Face recognition (on gray ROI)
        face_label = "unknown"
        if len(self.face.face_data) > 0:  # Only if trained
            try:
                face_roi = gray[y:y+h, x:x+w]
                label, confidence = self.face.face_recognizer.predict(face_roi)
                
                if confidence < getattr(face_mod, "CONFIDENCE_THRESHOLD", 70):
                    face_label = self.face.label_names.get(label, "unknown")
            except Exception:
                pass  # Keep as "unknown"
        
        # Emotion detection (on RGB ROI, same bbox)
        emotion_label = "neutral"
        emotion_scores = None
        try:
            from deepface import DeepFace
            # Convert gray ROI to RGB for DeepFace
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            face_roi_rgb = rgb[y:y+h, x:x+w]
            
            emotion = DeepFace.analyze(
                face_roi_rgb,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip',
                silent=True
            )
            emotion_label = emotion[0]['dominant_emotion']
            emotion_scores = emotion[0].get('emotion', {})
        except Exception:
            pass  # Keep defaults
        
        return face_label, emotion_label, emotion_scores, face_bbox

    def cleanup(self):
        """Release resources."""
        if self.video_source:
            self.video_source.release()
        if self.pose:
            self.pose.close()
        if self.gesture:
            self.gesture.close()
        cv2.destroyAllWindows()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


if __name__ == "__main__":
    # Standalone test mode (use friendly_spot_main.py for full system)
    video_source = create_video_source('webcam', device=0)
    
    with PerceptionPipeline(video_source) as pipeline:
        logger.info("Streaming perception at 5 Hz (Ctrl+C to stop)")
        try:
            while True:
                perception = pipeline.read_perception()
                if perception:
                    print(f"Pose:{perception.pose_label} Face:{perception.face_label} "
                          f"Emotion:{perception.emotion_label} Gesture:{perception.gesture_label} "
                          f"Dist:{perception.distance_m:.2f}m" if perception.distance_m else "Dist:None")
                time.sleep(0.2)
        except KeyboardInterrupt:
            logger.info("Stopped")