# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/18/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Main detection and tracking loop that fetches fisheye camera frames, runs YOLO detection,
# computes depth-based or area-based prioritization, calculates PTZ angles, and commands tracking
# Acknowledgements: Boston Dynamics Spot SDK for image acquisition and time sync patterns,
# Ultralytics YOLO for person detection, Claude for depth integration and prioritization logic

"""Main orchestrator: frames -> detections -> selection -> target -> PTZ.

Modes
- bearing: bearing-only mapping using camera intrinsics (Kannala-Brandt or pinhole)
    with frame transforms. Falls back to simple HFOV projection if intrinsics unavailable.
- transform: compute pixel->ray using SDK intrinsics, rotate ray into BODY, derive pan/tilt.

Function
- run_loop(robot, image_client, ptz_client, cfg):
    1) Fetch camera intrinsics from robot (once at startup).
    2) Fetch frames from surround cameras.
    3) Run YOLO detections.
    4) Select target person by nearest depth (if available); fallback to largest area.
    5) Compute pan/tilt using SDK intrinsics and frame transforms, command PTZ.
    6) TODO: Handle additional people in group interactions (policy to be defined).
"""
import logging
import time
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

from .config import (
    SURROUND_SOURCES,
    ROTATION_ANGLE,
    MIN_AREA_PX,
    RuntimeConfig,
)
from . import cameras
from . import geometry
from .detection import YoloDetector, Detection
from .ptz_control import set_ptz
from .visualization import show_detections_grid

logger = logging.getLogger(__name__)


def pick_largest(dets: List[Detection]):
    """Return detection with the largest pixel area from a list (or None)."""
    return max(dets, key=lambda d: d.bbox_xywh[2] * d.bbox_xywh[3]) if dets else None


def estimate_detection_depth_m(depth_img: Optional[np.ndarray], det: Detection, use_mask: bool = True) -> Optional[float]:
    """Estimate distance (meters) to the detected person from depth image.
    
    Uses segmentation mask if available for precise depth extraction, otherwise falls back
    to sampling a central region of the bounding box.

    Args: 
        depth_img: Depth image in meters (np.ndarray with NaN for invalid pixels), or None
        det: Detection with bounding box and optional segmentation mask
        use_mask: If True and mask available, use mask-based extraction (more precise)

    Returns: 
        Median depth in meters, or None if depth unavailable or no valid pixels
    """
    if depth_img is None:
        logger.debug("Depth image is None, cannot estimate distance")
        return None
    
    img_h, img_w = depth_img.shape
    logger.debug(f"Estimating depth from image {img_w}x{img_h}, bbox={det.bbox_xywh}, use_mask={use_mask}, has_mask={det.mask is not None}")
    
    # Method 1: Use segmentation mask if available (most precise)
    if use_mask and det.mask is not None:
        # Ensure mask matches depth image dimensions
        mask = det.mask
        if mask.shape != (img_h, img_w):
            logger.debug(f"Resizing mask from {mask.shape} to ({img_h}, {img_w})")
            # Resize mask to match depth image
            mask = cv2.resize(mask.astype(np.uint8), (img_w, img_h), 
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Extract depth values where mask is True
        mask_pixels = np.sum(mask)
        logger.debug(f"Mask covers {mask_pixels} pixels ({100*mask_pixels/(img_h*img_w):.1f}% of image)")
        masked_depths = depth_img[mask]
        valid_depths = masked_depths[~np.isnan(masked_depths)]
        
        if len(valid_depths) > 0:
            # Use median for robustness, could also use percentile (e.g., 10th for closest point)
            median_depth = float(np.median(valid_depths))
            min_depth = float(np.min(valid_depths))
            max_depth = float(np.max(valid_depths))
            logger.debug(f"Mask-based depth: median={median_depth:.2f}m, range=[{min_depth:.2f}, {max_depth:.2f}]m, {len(valid_depths)} valid pixels")
            return median_depth
        else:
            logger.debug(f"No valid depth pixels in mask ({mask_pixels} mask pixels, all NaN)")
    
    # Method 2: Fallback to bbox sampling (less precise but always works)
    logger.debug("Using bbox-based depth extraction (fallback method)")
    # Extract bounding box (x1, y1, width, height)
    x1, y1, w, h = det.bbox_xywh
    
    # Calculate center and sample from inner 60% of bbox to avoid edge artifacts
    # This focuses on the torso/center of person rather than arms/edges
    x_center = x1 + w / 2
    y_center = y1 + h / 2
    sample_w = w * 0.6
    sample_h = h * 0.6
    logger.debug(f"Sampling inner 60% of bbox: center=({x_center:.0f},{y_center:.0f}), sample_size=({sample_w:.0f}x{sample_h:.0f})")
    
    # Convert to sample ROI pixel coordinates
    sample_x1 = int(x_center - sample_w / 2)
    sample_y1 = int(y_center - sample_h / 2)
    sample_x2 = int(x_center + sample_w / 2)
    sample_y2 = int(y_center + sample_h / 2)
    
    # Clamp to image bounds
    sample_x1 = max(0, min(sample_x1, img_w - 1))
    sample_x2 = max(0, min(sample_x2, img_w - 1))
    sample_y1 = max(0, min(sample_y1, img_h - 1))
    sample_y2 = max(0, min(sample_y2, img_h - 1))
    
    # Ensure we have a valid ROI
    if sample_x2 <= sample_x1 or sample_y2 <= sample_y1:
        return None
    
    # Extract ROI and get valid depth values (excluding NaN)
    roi = depth_img[sample_y1:sample_y2, sample_x1:sample_x2]
    roi_pixels = (sample_x2 - sample_x1) * (sample_y2 - sample_y1)
    valid_depths = roi[~np.isnan(roi)]
    
    if len(valid_depths) == 0:
        logger.debug(f"No valid depth pixels in bbox ROI ({roi_pixels} total pixels, all NaN)")
        return None
    
    # Return median depth to be robust against outliers
    median_depth = float(np.median(valid_depths))
    min_depth = float(np.min(valid_depths))
    max_depth = float(np.max(valid_depths))
    logger.debug(f"Bbox-based depth: median={median_depth:.2f}m, range=[{min_depth:.2f}, {max_depth:.2f}]m, {len(valid_depths)}/{roi_pixels} valid pixels")
    return median_depth


def run_loop(robot, image_client, ptz_client, cfg: RuntimeConfig, shutdown_requested=None):
    """Main tracking loop with graceful shutdown support.
    
    Args:
        robot: Robot client instance
        image_client: Image service client
        ptz_client: PTZ control client
        cfg: Runtime configuration
        shutdown_requested: Callable or global flag to check for shutdown request
    """
    # Fetch camera intrinsics from robot (once at startup)
    logger.info("Fetching camera intrinsics from robot...")
    image_sources = cameras.fetch_image_sources(image_client)
    
    # Log intrinsics for all cameras we'll use
    for source_name in cfg.sources:
        intrinsics = cameras.get_camera_intrinsics(source_name, image_sources)
        if intrinsics:
            logger.info(f"{source_name}: {intrinsics['model']} model (fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f})")
        else:
            logger.warning(f"{source_name}: no intrinsics available, will use simple fallback")
    
    detector = YoloDetector(
        model_path=cfg.yolo.model_path,
        imgsz=cfg.yolo.img_size,
        conf=cfg.yolo.min_confidence,
        iou=cfg.yolo.iou_threshold,
        device=cfg.yolo.device,
        half=cfg.yolo.half,
        verbose=cfg.yolo.verbose
    )
    period = 1.0 / max(1, cfg.loop_hz)
    
    iteration = 0

    while True:
        # Check for shutdown request
        if shutdown_requested is not None and shutdown_requested():
            logger.info("Shutdown requested, exiting tracking loop gracefully")
            break
        
        iteration += 1
        start = time.time()
        frames, responses, depth_frames = cameras.get_frames(image_client, cfg.sources, include_depth=True)
        if not frames:
            time.sleep(period)
            continue

        names = list(frames.keys())
        imgs = list(frames.values())

        all_dets = detector.predict_batch(imgs)

        # Rank candidates by (depth if available) else by area
        best = None  # (name, det, width, resp, rank_value)
        detections_by_camera: Dict[str, List[Detection]] = {}
        
        for name, dets, img, resp in zip(names, all_dets, imgs, responses):
            detections_by_camera[name] = dets  # Store for visualization
            if not dets:
                continue
            # attach source name into Detection
            for d in dets:
                d.source = name
            # Prefer smallest depth when available
            depth_rank: List[Tuple[Detection, float]] = []
            for d in dets:
                # Get depth image for this camera source
                depth_img = depth_frames.get(name)
                dist_m = estimate_detection_depth_m(depth_img, d)
                if dist_m is not None:
                    depth_rank.append((d, dist_m))

            if depth_rank:
                # pick the minimum distance
                cand, dist = min(depth_rank, key=lambda x: x[1])
                rank_value = dist  # smaller is better
            else:
                cand = pick_largest(dets)
                if cand is None:
                    continue
                area = cand.bbox_xywh[2] * cand.bbox_xywh[3]
                if area < cfg.yolo.min_area_px:
                    continue
                rank_value = -area  # more area is better -> use negative for min-compare

            if best is None or rank_value < best[4]:
                best = (name, cand, img.shape[1], resp, rank_value)

        # Visualize detections if enabled
        if cfg.visualize:
            key = show_detections_grid(frames, detections_by_camera, depth_frames)
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                logger.info("User requested quit via visualization window")
                break
        
        # Save annotated frames if enabled
        if cfg.save_images:
            import os
            from .visualization import save_annotated_frames
            abs_path = os.path.abspath(cfg.save_images)
            logger.info(f"Calling save_annotated_frames: dir={abs_path}, iteration={iteration}, frames={len(frames)}")
            save_annotated_frames(frames, detections_by_camera, cfg.save_images, iteration, depth_frames)

        # Check once mode before continuing
        if cfg.once:
            logger.info(f"ONCE MODE: Completed iteration {iteration}, exiting")
            break

        if best is None:
            # No person found; skip
            if cfg.dry_run:
                logger.info("No person detected in any camera")
            time.sleep(max(0, period - (time.time() - start)))
            continue

        name, det, img_w, resp, rank_value = best
        
        # Log closest person distance if depth was available
        if rank_value > 0:  # rank_value is distance when depth is available
            logger.info(f"CLOSEST PERSON: {rank_value:.2f}m away in {name}")
        
        # Get camera intrinsics for this source
        intrinsics = cameras.get_camera_intrinsics(name, image_sources)
        cam_yaw_deg = geometry.get_camera_yaw_fallback(name)
        
        # Log intrinsics status
        if intrinsics:
            logger.debug(f"Intrinsics available for {name}: model={intrinsics.get('model')}")
        else:
            logger.warning(f"No intrinsics available for {name} - will use simple projection fallback")
        
        if cfg.dry_run:
            x, y, w, h = det.bbox_xywh
            logger.info(f"Detection: camera={name}, confidence={det.conf:.2f}, bbox=({x:.0f},{y:.0f},{w:.0f},{h:.0f})")

        # Compute PTZ target point from bbox
        # Use upper third of bbox (head/torso) instead of geometric center for better tracking
        x, y, w, h = det.bbox_xywh
        cx = x + w / 2.0
        cy = y + h * 0.3  # Target person's head/torso, not feet
        logger.debug(f"Target point: bbox center=({x+w/2:.0f}, {y+h/2:.0f}) to head/torso=({cx:.0f}, {cy:.0f})")
        
        # Get image dimensions for rotation correction
        img_height, img_width = frames[name].shape[:2]
        
        # Try transform mode first (accurate), fall back to simple projection if unavailable
        try:
            if cfg.observer_mode == "transform" and intrinsics:
                # Use full 3D transform pipeline with intrinsics
                ptz_pan_deg, ptz_tilt_deg = geometry.pixel_to_ptz_angles_transform(
                    cx, cy,
                    intrinsics,
                    resp,
                    robot,
                    img_width,
                    img_height
                )
            else:
                # Use simple geometric projection (fallback)
                if cfg.observer_mode == "transform":
                    if not intrinsics:
                        logger.warning(f"FALLBACK TO SIMPLE PROJECTION: Transform mode requested but no intrinsics available for {name}")
                    else:
                        logger.warning(f"FALLBACK TO SIMPLE PROJECTION: Transform mode requested but intrinsics invalid for {name}")
                else:
                    logger.info(f"Using simple projection mode (bearing mode explicitly selected)")
                
                # Calculate HFOV from intrinsics if available, otherwise use fallback
                if intrinsics:
                    hfov = cameras.calculate_hfov_from_intrinsics(intrinsics)
                else:
                    hfov = 133.0  # Fisheye fallback
                    logger.warning(f"No intrinsics for {name}, using fallback HFOV={hfov}°")
                
                ptz_pan_deg, ptz_tilt_deg = geometry.pixel_to_ptz_angles_simple(
                    cx, cy,
                    img_w,
                    frames[name].shape[0],  # img_height
                    hfov,
                    cam_yaw_deg,
                    cfg.ptz.default_tilt_deg
                )
        except Exception as e:
            logger.error(f"Failed to compute PTZ angles: {e}, using simple projection fallback")
            hfov = 133.0
            ptz_pan_deg, ptz_tilt_deg = geometry.pixel_to_ptz_angles_simple(
                cx, cy, img_w, frames[name].shape[0], hfov, cam_yaw_deg, cfg.ptz.default_tilt_deg
            )
        
        # Log and send PTZ command
        logger.info(f"PTZ Command: pan={ptz_pan_deg:.2f}°, tilt={ptz_tilt_deg:.2f}°, zoom={cfg.ptz.default_zoom:.2f} (camera={name}, conf={det.conf:.2f})")
        set_ptz(ptz_client, cfg.ptz.name, ptz_pan_deg, ptz_tilt_deg, cfg.ptz.default_zoom, dry_run=cfg.dry_run)

        # Exit after detection if in exit-on-detection mode
        if cfg.exit_on_detection:
            logger.info(f"EXIT-ON-DETECTION MODE: Person detected and PTZ commanded, exiting")
            break

        # TODO: Additional people handling
        # If multiple people are present, we may:
        # - Keep PTZ on primary target while tracking secondary targets for context.
        # - Cycle attention with time decay or user policy.
        # - Blend cues (e.g., center of mass of group) when no single dominant target exists.

        # Pacing with interruptible sleep
        sleep_left = period - (time.time() - start)
        if sleep_left > 0:
            # Sleep in small increments to allow faster response to shutdown signals
            sleep_increment = 0.1  # Check every 100ms
            elapsed = 0
            while elapsed < sleep_left:
                if shutdown_requested is not None and shutdown_requested():
                    logger.info("Shutdown requested during sleep, exiting immediately")
                    return
                time.sleep(min(sleep_increment, sleep_left - elapsed))
                elapsed += sleep_increment
    
    logger.info(f"Tracking loop finished after {iteration} iterations")
