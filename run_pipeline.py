import os
import sys
import cv2
import time
import importlib.util
import importlib.machinery
import numpy as np
import logging

from video_sources import VideoSource, create_video_source

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load workspace classes from files (handles folder name with space)
def load_module_from_path(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module

BASE = os.path.dirname(os.path.abspath(__file__))
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

from behavior_planner import PerceptionInput  # [`PerceptionInput`](behavior_planner.py)

PoseAnalyzer = pose_mod.PoseAnalyzer  # [`PoseAnalyzer`](Facial Recognition/streamlinedRuleBasedEstimation.py)
FaceRecognizer = face_mod.FaceRecognizer  # [`FaceRecognizer`](Facial Recognition/streamlinedCombinedMemoryAndEmotion.py)
GestureRecognizer = gesture_mod.GestureRecognizer  # [`GestureRecognizer`](Facial Recognition/streamlinedGestureDetection.py)

# Reuse mediapipe from pose_mod for efficiency (already imported there)
mp = getattr(pose_mod, "mp", None)
if mp is None:
    import mediapipe as mp
mp_hands = mp.solutions.hands

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
    def __init__(self, video_source: VideoSource, robot=None):
        """Initialize perception pipeline with configurable video source.
        
        Args:
            video_source: VideoSource instance (webcam, ImageClient, or WebRTC)
            robot: Optional Robot instance for behavior command execution
        
        Note: Cross-platform compatible. Optimizes TensorFlow for DeepFace performance
        with GPU memory growth and model pre-building to avoid repeated loading overhead.
        """
        self.video_source = video_source
        self.robot = robot
        
        # TensorFlow optimization: Configure GPU memory growth (prevents OOM)
        # This is more efficient than limiting threads
        self._configure_tensorflow()
        
        # Pre-build DeepFace models
        self._prebuild_deepface_models()
        
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
        
        # Initialize gesture recognizer (loads model as bytes to avoid Windows path issues)
        gesture_model_path = os.path.join(BASE, "Facial Recognition", "gesture_recognizer.task")
        try:
            self.gesture = GestureRecognizer(model_path=gesture_model_path)
        except Exception as e:
            logger.warning(f"Gesture recognizer initialization failed: {e} - gestures will be 'none'")
            self.gesture = None
    
    def _configure_tensorflow(self):
        """Configure TensorFlow for optimal DeepFace performance.
        
        Enables GPU memory growth to prevent OOM errors and allows TensorFlow
        to use GPU efficiently. More effective than limiting threading.
        """
        try:
            import tensorflow as tf
            
            # Enable GPU memory growth (prevents TF from allocating all GPU memory)
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU available: {len(gpus)} device(s) - memory growth enabled")
            else:
                logger.info("No GPU detected - using CPU for DeepFace")
            
            # Suppress TensorFlow warnings
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
        except Exception as e:
            logger.warning(f"TensorFlow configuration failed: {e}")
    
    def _prebuild_deepface_models(self):
        """Pre-build DeepFace models to avoid loading overhead on every call.
        
        
        Note: DeepFace emotion analysis doesn't use build_model() - the emotion
        model is loaded automatically on first DeepFace.analyze() call. This method
        triggers that initial load with a dummy analysis to avoid first-frame delay.
        """
        try:
            # Reuse DeepFace from face module if already imported
            DeepFace = getattr(face_mod, "DeepFace", None)
            if DeepFace is None:
                # Fallback: import directly if not in face module
                from deepface import DeepFace
            
            logger.info("Pre-loading DeepFace emotion model (this may take a few seconds)...")
            
            # Create a small dummy image to trigger emotion model loading
            # This downloads the model weights if not already cached
            dummy_img = np.zeros((48, 48, 3), dtype=np.uint8)
            try:
                DeepFace.analyze(
                    dummy_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip',
                    silent=True
                )
                logger.info("Emotion model pre-loaded successfully")
            except Exception as e:
                # Expected to fail on dummy image, but model should be loaded now
                logger.info("Emotion model cached - ready for use")
            
        except Exception as e:
            logger.warning(f"DeepFace model pre-loading failed: {e} - will load on first use")

    def read_perception(self):
        """Read frame from video source and extract perception data."""
        ret, frame = self.video_source.read()
        if not ret or frame is None:
            return None
        frame_h, frame_w = frame.shape[:2]

        # Pose -> landmarks & action
        landmarks, results = self.pose.extract_landmarks(frame)
        if landmarks is not None:
            pose_label = self.pose.detect_action(landmarks)
            current_action = pose_label  # TODO: implement spot's current action
            distance_m = estimate_distance_m(landmarks, frame_h) # TODO: use spot's distance estimator from robot using depth if available
        else:
            pose_label = "unknown"
            current_action = "unknown"
            distance_m = None

        # Gesture
        if self.gesture is not None:
            try:
                gesture_label = self.gesture.recognize_gestures(frame)
            except Exception as e:
                logger.debug(f"Gesture recognition failed: {e}")
                gesture_label = "none"
        else:
            gesture_label = "none"

        # Face + Emotion (reuse cv2 from face_mod for efficiency)
        _cv2 = getattr(face_mod, "cv2", cv2)
        gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
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
            # emotion using DeepFace (optimized with pre-built model)
            try:
                # Reuse DeepFace from face module for efficiency
                DeepFace = getattr(face_mod, "DeepFace", None)
                if DeepFace is None:
                    from deepface import DeepFace
                
                # Optimization flags:
                # - enforce_detection=False: skip redundant detection (we have faces)
                # - detector_backend='skip': use our Haar cascade results
                # - silent=True: reduce logging I/O overhead
                emotion = DeepFace.analyze(
                    frame[y:y+h, x:x+w], 
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip',
                    silent=True
                )
                emotion_label = emotion[0]['dominant_emotion']
            except Exception:
                emotion_label = 'neutral'

        perception = PerceptionInput(
            current_action=current_action,
            distance_m=distance_m,
            face_label=face_label,
            emotion_label=emotion_label,
            pose_label=pose_label,
            gesture_label=gesture_label,
        )
        return perception

    def cleanup(self):
        """Release all resources (cross-platform compatible)."""
        if hasattr(self, 'video_source') and self.video_source:
            self.video_source.release()
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        if hasattr(self, 'pose') and self.pose:
            self.pose.close()
        if hasattr(self, 'gesture') and self.gesture:
            self.gesture.close()
        cv2.destroyAllWindows()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def stream(self, rate_hz=10):
        try:
            while True:
                p = self.read_perception()
                if p is None:
                    break
                print(p)
                time.sleep(1.0 / rate_hz)
        finally:
            self.cleanup()

if __name__ == "__main__":
    # Example usage: webcam mode
    # For robot mode, use friendly_spot_main.py instead
    logger.info("Running perception pipeline in webcam mode (development)")
    video_source = create_video_source('webcam', device=0)
    
    with PerceptionPipeline(video_source) as pipeline:
        pipeline.stream(rate_hz=5)