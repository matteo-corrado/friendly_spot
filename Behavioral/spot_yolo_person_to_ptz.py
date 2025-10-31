import argparse
import time
import math
from collections import OrderedDict

import numpy as np
import cv2
from ultralytics import YOLO

from bosdyn.client import create_standard_sdk
from bosdyn.client.image import ImageClient
from bosdyn.client.spot_cam.ptz import PtzClient, PtzPosition
from bosdyn.client.spot_cam.compositor import CompositorClient
from bosdyn.client.spot_cam.streamquality import StreamQualityClient
from bosdyn.api import image_pb2


# --- CONFIG ---
# Surround cameras as used in the Fetch tutorial (names end with "_image")
SURROUND_SOURCES = [
    "frontleft_fisheye_image", "frontright_fisheye_image",
    "left_fisheye_image", "right_fisheye_image",
    "back_fisheye_image"
]

# Approx horizontal FOVs (deg). Works fine for aiming. Refine later if needed.
SURROUND_HFOV_DEG = {
    "frontleft_fisheye_image": 133.0,
    "frontright_fisheye_image": 133.0,
    "left_fisheye_image": 133.0,
    "right_fisheye_image": 133.0,
    "back_fisheye_image": 133.0,
}

# Camera mounting yaw wrt robot forward (+X), deg (CCW positive when viewed from above).
CAM_YAW_DEG = {
    "frontleft_fisheye_image":  +45.0,
    "frontright_fisheye_image": -45.0,
    "left_fisheye_image":      +90.0,
    "right_fisheye_image":     -90.0,
    "back_fisheye_image":      180.0,
}

PTZ_NAME = "ptz"          # typical default
COMPOSITOR_SCREEN = "mech" # PTZ-focused layout; change if yours differs
TARGET_BITRATE = 2_000_000  # 2 Mbps for stable WebRTC on laptop
YOLO_MODEL = "yolov8n.pt"   # use 'yolov8s.pt' for more recall if FPS allows
YOLO_CONF = 0.30
YOLO_IOU = 0.50
IMG_SIZE = 640
SLEEP_BETWEEN_POLLS = 0.15  # seconds


# --- UTILS ---
def connect_robot(ip, user, password):
    sdk = create_standard_sdk("SpotYOLOPersonToPTZ")
    robot = sdk.create_robot(ip)
    robot.authenticate(user, password)
    return robot

def ensure_available_sources(image_client, desired):
    available = {s.name for s in image_client.list_image_sources()}
    usable = [s for s in desired if s in available]
    if not usable:
        raise RuntimeError(f"No desired surround cameras found. Available: {sorted(available)}")
    return usable

def decode_image(resp):
    img = None
    fmt = resp.shot.image.format
    data = resp.shot.image.data
    if fmt == image_pb2.Image.FORMAT_JPEG:
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif fmt == image_pb2.Image.FORMAT_RAW:
        rows, cols = resp.shot.image.rows, resp.shot.image.cols
        # assume raw grayscale or bayer; many Spot fisheyes come JPEG so this is rare
        img = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def get_frames(image_client, sources):
    # Single RPC to fetch all sources
    frames = OrderedDict()
    responses = image_client.get_image_from_sources(sources)
    for r in responses:
        img = decode_image(r)
        if img is not None:
            frames[r.source.name] = img
    return frames

def yolo_setup():
    model = YOLO(YOLO_MODEL)
    model.fuse()  # small speed boost
    return model

def yolo_detect_batch(model, bgr_list):
    # returns list-of-lists: per image -> [(x,y,w,h,conf), ...] for class 'person' (0)
    out = []
    for r in model.predict(
        bgr_list, imgsz=IMG_SIZE, conf=YOLO_CONF, iou=YOLO_IOU,
        classes=[0], device=0, half=True, verbose=False
    ):
        dets = []
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                dets.append((x1, y1, x2 - x1, y2 - y1, conf))
        out.append(dets)
    return out

def pick_largest(dets):
    # dets: [(x,y,w,h,conf), ...]
    return max(dets, key=lambda d: d[2] * d[3]) if dets else None

def bbox_center(bbox):
    x, y, w, h, *_ = bbox
    return (x + w / 2.0, y + h / 2.0)

def pixel_to_yaw_offset(cx, img_w, hfov_deg):
    norm_x = (cx - (img_w / 2.0)) / (img_w / 2.0)  # -1 .. +1
    return norm_x * (hfov_deg / 2.0)

def bearing_from_detection(source_name, bbox, img_width):
    hfov = SURROUND_HFOV_DEG.get(source_name, 133.0)
    cam_yaw = CAM_YAW_DEG.get(source_name, 0.0)
    cx, _ = bbox_center(bbox)
    yaw_off = pixel_to_yaw_offset(cx, img_width, hfov)
    return cam_yaw + yaw_off  # robot-frame approximate bearing (deg)

def configure_ptz_stream(robot):
    robot.ensure_client(CompositorClient.default_service_name).set_screen(COMPOSITOR_SCREEN)
    robot.ensure_client(StreamQualityClient.default_service_name).set_stream_params(
        target_bitrate=TARGET_BITRATE
    )

def aim_ptz(robot, pan_deg, tilt_deg=-5.0, zoom=0.0):
    ptz = robot.ensure_client(PtzClient.default_service_name)
    pos = PtzPosition(
        pan=math.radians(pan_deg),
        tilt=math.radians(tilt_deg),
        zoom=zoom
    )
    # duration=0 for fastest movement
    ptz.set_ptz_position(PTZ_NAME, pos, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, help="Spot hostname or IP")
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--tilt", type=float, default=-5.0, help="PTZ tilt while slewing (deg)")
    ap.add_argument("--once", action="store_true", help="Run one detection+aim cycle and exit")
    args = ap.parse_args()

    # Connect SDK
    robot = connect_robot(args.robot, args.user, args.password)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    configure_ptz_stream(robot)

    # Camera availability
    sources = ensure_available_sources(image_client, SURROUND_SOURCES)
    print("Using surround cameras:", sources)

    # YOLO
    model = yolo_setup()
    print("YOLO loaded →", YOLO_MODEL)

    try:
        while True:
            # 1) grab frames
            frames = get_frames(image_client, sources)  # OrderedDict[name -> bgr]
            if not frames:
                time.sleep(SLEEP_BETWEEN_POLLS)
                continue

            names = list(frames.keys())
            imgs  = list(frames.values())

            # 2) run YOLO on the batch
            all_dets = yolo_detect_batch(model, imgs)

            # 3) pick the best (largest area) person across cameras
            best_name, best_bbox, best_width, best_area = None, None, None, -1
            for name, dets, img in zip(names, all_dets, imgs):
                d = pick_largest(dets)
                if not d:
                    continue
                area = d[2] * d[3]
                if area > best_area:
                    best_area  = area
                    best_name  = name
                    best_bbox  = d
                    best_width = img.shape[1]

            if best_name is None:
                # no person found
                time.sleep(SLEEP_BETWEEN_POLLS)
                if args.once:
                    print("No person found.")
                    return
                continue

            # 4) compute bearing and aim PTZ
            bearing = bearing_from_detection(best_name, best_bbox, best_width)
            print(f"[PERSON] {best_name} → pan ≈ {bearing:.1f}° (area {best_area})")
            aim_ptz(robot, pan_deg=bearing, tilt_deg=args.tilt, zoom=0.0)

            if args.once:
                return

            time.sleep(SLEEP_BETWEEN_POLLS)

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        pass


if __name__ == "__main__":
    main()
