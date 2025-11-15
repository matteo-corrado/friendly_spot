"""Main orchestrator: frames → detections → selection → target → PTZ.

Modes
- bearing: bearing-only mapping using HFOV + camera yaw (prefer yaw from transforms).
- transform: compute pixel→ray using pinhole intrinsics, rotate ray into BODY, then derive pan/tilt.

Function
- run_loop(robot, image_client, ptz_client, cfg):
    1) Fetch frames from surround cameras.
    2) Run YOLO detections.
    3) Select target person by nearest depth (if a depth image is available for that source);
       fallback to largest area.
    4) Compute pan/tilt (bearing or transform mode) and command PTZ.
    5) TODO: Handle additional people in group interactions (policy to be defined).
"""
import time
from typing import List, Tuple

import numpy as np

from .config import RuntimeConfig
from .cameras import get_frames
from .detection import YoloDetector, Detection
from .geometry import (
    bbox_center,
    bearing_from_detection_bearing_only,
    camera_yaw_from_transforms,
    pixel_to_ray_pinhole,
    transform_direction,
)
from .ptz_control import set_ptz


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
    detector = YoloDetector(conf=cfg.min_conf)
    period = 1.0 / max(1, cfg.loop_hz)

    while True:
        start = time.time()
        frames, responses = get_frames(image_client, cfg.sources)
        if not frames:
            time.sleep(period)
            continue

        names = list(frames.keys())
        imgs = list(frames.values())

        all_dets = detector.predict_batch(imgs)

        # Rank candidates by (depth if available) else by area
        best = None  # (name, det, width, resp, rank_value)
        for name, dets, img, resp in zip(names, all_dets, imgs, responses):
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
                if area < cfg.min_area_px:
                    continue
                rank_value = -area  # more area is better → use negative for min-compare

            if best is None or rank_value < best[4]:
                best = (name, cand, img.shape[1], resp, rank_value)

        if best is None:
            # No person found; skip
            time.sleep(max(0, period - (time.time() - start)))
            continue

        name, det, img_w, resp, _ = best

        if cfg.observer_mode.mode == "transform":
            # Transform-based pan/tilt: pixel → ray (camera), rotate to BODY
            cx, cy = bbox_center(det.bbox_xywh)
            try:
                d_cam = pixel_to_ray_pinhole(resp, cx, cy)
                d_body = transform_direction(resp.shot.transforms_snapshot,
                                             resp.shot.frame_name_image_sensor,
                                             "body",  # BODY frame
                                             d_cam)
                # BODY axes: x forward, y left, z up
                pan_rad = float(np.arctan2(d_body[1], d_body[0]))
                hyp = float(np.hypot(d_body[0], d_body[1]))
                tilt_rad = float(np.arctan2(-d_body[2], hyp))
                pan_deg = np.degrees(pan_rad)
                tilt_deg = np.degrees(tilt_rad)
            except Exception:
                # Fallback to bearing-only if intrinsics/frames missing
                try:
                    cam_yaw = camera_yaw_from_transforms(resp.shot.transforms_snapshot,
                                                         resp.shot.frame_name_image_sensor)
                except Exception:
                    cam_yaw = 0.0
                hfov = cfg.hfov_deg.get(name, 133.0)
                pan_deg = bearing_from_detection_bearing_only(name, det.bbox_xywh, img_w, hfov, cam_yaw)
                tilt_deg = cfg.default_tilt_deg
        else:
            # Bearing-only yaw offset via HFOV mapping with transform-derived camera yaw
            try:
                cam_yaw = camera_yaw_from_transforms(resp.shot.transforms_snapshot,
                                                     resp.shot.frame_name_image_sensor)
            except Exception:
                cam_yaw = 0.0
            hfov = cfg.hfov_deg.get(name, 133.0)
            pan_deg = bearing_from_detection_bearing_only(name, det.bbox_xywh, img_w, hfov, cam_yaw)
            tilt_deg = cfg.default_tilt_deg

        set_ptz(ptz_client, cfg.ptz_name, pan_deg, tilt_deg, 0.0)

        # TODO: Additional people handling
        # If multiple people are present, we may:
        # - Keep PTZ on primary target while tracking secondary targets for context.
        # - Cycle attention with time decay or user policy.
        # - Blend cues (e.g., center of mass of group) when no single dominant target exists.

        # pacing
        sleep_left = period - (time.time() - start)
        if sleep_left > 0:
            time.sleep(sleep_left)
