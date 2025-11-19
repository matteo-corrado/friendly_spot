# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/17/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: YOLO-based person detection wrapper using Ultralytics with GPU acceleration,
# batch processing for multiple camera frames, and person class filtering
# Acknowledgements: Ultralytics YOLOv8 documentation for model API and detection parameters,
# Claude for batch prediction optimization and dataclass design

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
    """YOLO detection result with segmentation mask.
    
    Attributes:
        source: Camera source name
        bbox_xywh: Bounding box as (x1, y1, width, height) in pixels
        conf: Detection confidence [0.0, 1.0]
        mask: Segmentation mask as boolean numpy array (H, W) or None if unavailable
    """
    source: str
    bbox_xywh: Tuple[int, int, int, int]
    conf: float
    mask: np.ndarray = None  # Optional segmentation mask


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
        logger.debug(f"Model config: imgsz={imgsz}, conf={conf}, iou={iou}, device={device}, half={half}")
        self.model = YOLO(model_path)
        self.model.fuse()
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.verbose = verbose
        
        # Check if model supports segmentation
        self.has_segmentation = hasattr(self.model, 'task') and 'segment' in str(self.model.task)
        if self.has_segmentation:
            logger.info("Model supports instance segmentation")
            logger.debug(f"Model task: {self.model.task}")
        else:
            logger.info("Model is detection-only (no segmentation)")
            logger.debug(f"Model task: {getattr(self.model, 'task', 'unknown')}")
        
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
        """Run YOLO on a batch of BGR images with segmentation if available.

        Input: list of np.ndarray images (BGR)
        Output: list (per image) of Detection entries with masks (source is filled by caller)
        """
        logger.debug(f"Running YOLO prediction on batch of {len(bgr_list)} images")
        out: List[List[Detection]] = []
        results = self.model.predict(
            bgr_list, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            classes=[PERSON_CLASS_ID], device=self.device, 
            half=self.half,  # FP16 only on GPU
            verbose=self.verbose
        )
        logger.debug(f"YOLO prediction complete, processing {len(results)} results")
        for img_idx, (r, img) in enumerate(zip(results, bgr_list)):
            dets: List[Detection] = []
            if r.boxes is not None:
                num_detections = len(r.boxes)
                logger.debug(f"Image {img_idx}: Found {num_detections} person detections")
                
                # Extract segmentation masks if available
                masks = None
                if self.has_segmentation and hasattr(r, 'masks') and r.masks is not None:
                    # masks.data is tensor of shape (N, H, W) where N is number of detections
                    masks = r.masks.data.cpu().numpy()  # Convert to numpy
                    logger.debug(f"Image {img_idx}: Extracted {len(masks)} segmentation masks, shape: {masks[0].shape if len(masks) > 0 else 'N/A'}")
                elif self.has_segmentation:
                    logger.debug(f"Image {img_idx}: Model supports segmentation but no masks in results")
                else:
                    logger.debug(f"Image {img_idx}: Detection-only model, no masks available")
                
                for i, b in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    conf = float(b.conf[0])
                    bbox_w, bbox_h = x2 - x1, y2 - y1
                    logger.debug(f"Image {img_idx}, Detection {i}: bbox=({x1},{y1},{bbox_w},{bbox_h}), conf={conf:.3f}")
                    
                    # Extract mask for this detection if available
                    mask = None
                    if masks is not None and i < len(masks):
                        # Resize mask to original image size if needed
                        mask_data = masks[i]
                        if mask_data.shape != (img.shape[0], img.shape[1]):
                            import cv2
                            logger.debug(f"Image {img_idx}, Detection {i}: Resizing mask from {mask_data.shape} to ({img.shape[0]}, {img.shape[1]})")
                            mask_data = cv2.resize(mask_data.astype(np.float32), 
                                                  (img.shape[1], img.shape[0]))
                        # Convert to boolean mask (threshold at 0.5)
                        mask = (mask_data > 0.5).astype(bool)
                        mask_pixels = np.sum(mask)
                        logger.debug(f"Image {img_idx}, Detection {i}: Mask extracted, {mask_pixels} pixels ({100*mask_pixels/(img.shape[0]*img.shape[1]):.1f}% of image)")
                    
                    dets.append(Detection("", (x1, y1, x2 - x1, y2 - y1), conf, mask))
                logger.debug(f"Image {img_idx}: Total {len(dets)} detections added")
            out.append(dets)
        return out
