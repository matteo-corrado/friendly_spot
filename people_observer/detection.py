"""YOLO detector wrapper and detection struct.

Use ultralytics for convenience. Keep person-only class filtering.

Functions/classes
- Detection: dataclass holding source name, bbox (xywh), confidence.
- YoloDetector(model_path, imgsz, conf, iou)
    .predict_batch(bgr_list) -> list[list[Detection]]
        Run model on a list of BGR images and return detections per image.
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from .config import PERSON_CLASS_ID


@dataclass
class Detection:
    source: str
    bbox_xywh: Tuple[int, int, int, int]
    conf: float


class YoloDetector:
    def __init__(self, model_path: str = "yolov8n.pt", imgsz: int = 640, conf: float = 0.3, iou: float = 0.5):
        self.model = YOLO(model_path)
        self.model.fuse()
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

    def predict_batch(self, bgr_list: List[np.ndarray]) -> List[List[Detection]]:
        """Run YOLO on a batch of BGR images.

        Input: list of np.ndarray images (BGR)
        Output: list (per image) of Detection entries (source is filled by caller)
        """
        out: List[List[Detection]] = []
        results = self.model.predict(
            bgr_list, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            classes=[PERSON_CLASS_ID], device=0, half=True, verbose=False
        )
        for r, img in zip(results, bgr_list):
            dets: List[Detection] = []
            if r.boxes is not None:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    conf = float(b.conf[0])
                    dets.append(Detection("", (x1, y1, x2 - x1, y2 - y1), conf))
            out.append(dets)
        return out
