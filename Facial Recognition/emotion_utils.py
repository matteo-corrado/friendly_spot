
import warnings
from typing import Dict, Tuple

import numpy as np

# Try a few lightweight options, fall back gracefully if not installed.

class _FerBackend:
    def __init__(self):
        from fer import FER  # pip install fer
        self.detector = FER(mtcnn=False)

    def predict(self, bgr_face: np.ndarray) -> Tuple[str, float, float, float]:
        import cv2
        rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        preds = self.detector.detect_emotions(rgb)
        if not preds:
            return "neutral", 0.0, 0.2, 0.0
        emo_dict = preds[0]["emotions"]
        label = max(emo_dict, key=emo_dict.get)
        conf = float(emo_dict[label])
        v, a = EmotionEstimator.map_discrete_to_VA(label, conf)
        return label, v, a, conf


class _DeepfaceBackend:
    def __init__(self):
        from deepface import DeepFace  # pip install deepface
        self.df = DeepFace

    def predict(self, bgr_face: np.ndarray) -> Tuple[str, float, float, float]:
        import cv2
        rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        res = self.df.analyze(rgb, actions=['emotion'], enforce_detection=False)
        if isinstance(res, list):
            res = res[0]
        emo = res.get('dominant_emotion', 'neutral')
        conf = float(res.get('emotion', {}).get(emo, 0.0))
        v, a = EmotionEstimator.map_discrete_to_VA(emo, conf)
        return emo, v, a, conf


class EmotionEstimator:
    """
    Unified interface with graceful fallbacks:
      - Backend 'fer' (recommended: pip install fer)
      - Backend 'deepface' (optional: pip install deepface)
      - Backend 'none' (always available: returns neutral zeros)
    """
    def __init__(self, backend: str = "auto"):
        self.backend_name = "none"
        self.backend = None
        import importlib

        def try_backend(name, cls):
            try:
                self.backend = cls()
                self.backend_name = name
                return True
            except Exception as e:
                warnings.warn(f"[EmotionEstimator] {name} backend unavailable: {e}")
                return False

        if backend == "fer":
            if try_backend("fer", _FerBackend):
                return
        elif backend == "deepface":
            if try_backend("deepface", _DeepfaceBackend):
                return
        else:  # auto
            if try_backend("fer", _FerBackend):
                return
            if try_backend("deepface", _DeepfaceBackend):
                return

        warnings.warn("[EmotionEstimator] No emotion backend available. Using neutral fallback.")
        self.backend = None
        self.backend_name = "none"

    @staticmethod
    def map_discrete_to_VA(label: str, confidence: float = 1.0) -> Tuple[float, float]:
        table: Dict[str, Tuple[float, float]] = {
            "angry":   (-0.8, 0.8),
            "disgust": (-0.7, 0.5),
            "fear":    (-0.9, 0.9),
            "sad":     (-0.8, 0.3),
            "neutral": ( 0.0, 0.2),
            "surprise":( 0.2, 0.9),
            "happy":   ( 0.9, 0.6),
            "contempt":(-0.5, 0.4),
            "confused":(-0.2, 0.5),
            "calm":    ( 0.2, 0.2),
        }
        l = (label or "neutral").lower()
        base_v, base_a = table.get(l, table["neutral"])
        v = float(base_v * confidence)
        a = float((base_a - 0.2) * confidence + 0.2)
        v = max(-1.0, min(1.0, v))
        a = max(0.0, min(1.0, a))
        return v, a

    def predict(self, bgr_face: np.ndarray) -> Tuple[str, float, float, float]:
        if self.backend is None:
            return "neutral", 0.0, 0.2, 0.0
        return self.backend.predict(bgr_face)


def compute_comfort(valence: float, arousal: float) -> float:
    c = 0.8 * float(valence) - 0.3 * (float(arousal) - 0.3)
    return max(-1.0, min(1.0, c))
