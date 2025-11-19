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
import logging

import numpy as np
import torch
from ultralytics import YOLO

from .config import (
    PERSON_CLASS_ID,
    DEFAULT_YOLO_MODEL,
    YOLO_IMG_SIZE,
    MIN_CONFIDENCE,
    YOLO_IOU_THRESHOLD,
    YOLO_DEVICE,
    YOLO_HALF,
    YOLO_VERBOSE
)

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """YOLO detection result.
    
    Attributes:
        source: Camera source name
        bbox_xywh: Bounding box as (x1, y1, width, height) in pixels
        conf: Detection confidence [0.0, 1.0]
    """
    source: str
    bbox_xywh: Tuple[int, int, int, int]
    conf: float


class YoloDetector:
    def __init__(self, model_path: str = DEFAULT_YOLO_MODEL, imgsz: int = YOLO_IMG_SIZE, 
                 conf: float = MIN_CONFIDENCE, iou: float = YOLO_IOU_THRESHOLD, 
                 device: str = YOLO_DEVICE, half: bool = YOLO_HALF, verbose: bool = YOLO_VERBOSE):
        """Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights file
            imgsz: Input image size (must be multiple of 32)
            conf: Confidence threshold for detections [0.0, 1.0]
            iou: IOU threshold for NMS [0.0, 1.0]
            device: 'cuda' or 'cpu' for inference device
            half: Use FP16 half-precision (only effective on CUDA)
            verbose: Enable verbose YOLO logging
        """
        # Verify model file exists
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO model not found at {model_path}. "
                f"Please download model to {os.path.dirname(model_path)}/"
            )
        
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self.model.fuse()
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.verbose = verbose
        
        # Determine device (GPU if available, fallback to CPU)
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            # Half precision only works on CUDA
            self.half = half
            logger.info(f"YOLO using GPU: {torch.cuda.get_device_name(0)} (FP16={self.half})")
        else:
            self.device = "cpu"
            # Force half=False on CPU (not supported)
            self.half = False
            if device == "cuda":
                logger.warning("CUDA requested but not available, falling back to CPU")
            if half:
                logger.warning("FP16 half-precision not supported on CPU, using FP32")
            logger.info("YOLO using CPU")

    def predict_batch(self, bgr_list: List[np.ndarray]) -> List[List[Detection]]:
        """Run YOLO on a batch of BGR images.

        Input: list of np.ndarray images (BGR)
        Output: list (per image) of Detection entries (source is filled by caller)
        """
        out: List[List[Detection]] = []
        results = self.model.predict(
            bgr_list, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            classes=[PERSON_CLASS_ID], device=self.device, 
            half=self.half,  # FP16 only on GPU
            verbose=self.verbose
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
