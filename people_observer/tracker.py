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
from typing import List, Tuple, Dict

import cv2
import numpy as np

from .config import RuntimeConfig
from . import cameras
from . import geometry
from .detection import YoloDetector, Detection
from .ptz_control import set_ptz
from .visualization import show_detections_grid

logger = logging.getLogger(__name__)


def pick_largest(dets: List[Detection]):
    """Return detection with the largest pixel area from a list (or None)."""
    return max(dets, key=lambda d: d.bbox_xywh[2] * d.bbox_xywh[3]) if dets else None


def estimate_detection_depth_m(resp, det: Detection) -> float:
    """Estimate distance to detection in meters from a depth-capable ImageResponse.

    Inputs:
    - resp: ImageResponse for the same source the detection came from
    - det: Detection with bbox in pixel coordinates

    Output: float distance in meters (smaller is closer). Returns None if depth
    is unavailable for this response or cannot be decoded.

    Note: Surround fisheye cameras often do not provide depth. This function is a
    placeholder for future integration with depth-capable sources (e.g., hand depth)
    or NCB outputs. When unavailable, caller must fall back to largest-area policy.
    """
    # TODO: Implement per-source depth extraction when depth ImageResponses are available.
    # Options:
    # - Use bosdyn.client.image.depth_image_to_pointcloud to compute 3D points, then
    #   sample the ROI around bbox center and take median range.
    # - Use pixel-format specific decoding if depths are encoded as U16 (meters or mm).
    _ = (resp, det)  # keep signature used
    return None


def run_loop(robot, image_client, ptz_client, cfg: RuntimeConfig):
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
        device=cfg.yolo.device
    )
    period = 1.0 / max(1, cfg.loop_hz)
    
    iteration = 0

    while True:
        iteration += 1
        start = time.time()
        frames, responses = cameras.get_frames(image_client, cfg.sources)
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
                dist_m = estimate_detection_depth_m(resp, d)
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
            key = show_detections_grid(frames, detections_by_camera)
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                logger.info("User requested quit via visualization window")
                break
        
        # Save annotated frames if enabled
        if cfg.save_images:
            import os
            from .visualization import save_annotated_frames
            abs_path = os.path.abspath(cfg.save_images)
            logger.info(f"Calling save_annotated_frames: dir={abs_path}, iteration={iteration}, frames={len(frames)}")
            save_annotated_frames(frames, detections_by_camera, cfg.save_images, iteration)

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

        name, det, img_w, resp, _ = best
        
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

        # Compute PTZ angles from pixel detection
        cx = det.bbox_xywh[0] + det.bbox_xywh[2] / 2.0
        cy = det.bbox_xywh[1] + det.bbox_xywh[3] / 2.0
        
        # Try transform mode first (accurate), fall back to simple projection if unavailable
        try:
            if cfg.observer_mode == "transform" and intrinsics:
                # Use full 3D transform pipeline with intrinsics
                ptz_pan_deg, ptz_tilt_deg = geometry.pixel_to_ptz_angles_transform(
                    cx, cy,
                    intrinsics,
                    resp,
                    robot
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

        # pacing
        sleep_left = period - (time.time() - start)
        if sleep_left > 0:
            time.sleep(sleep_left)
    
    # Cleanup OpenCV windows
    if cfg.visualize:
        cv2.destroyAllWindows()
