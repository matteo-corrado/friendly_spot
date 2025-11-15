"""Camera utilities for surround capture.

Prefer batch calls to ImageClient.get_image_from_sources so responses in a cycle
share a similar transforms_snapshot. Functions here do minimal decode and return
both images and the raw ImageResponses so downstream code can access intrinsics
and frame transforms.

Functions
- ensure_available_sources(image_client, desired) -> list[str]
    Validate desired source names against the robot's advertised image sources.
- decode_image(resp) -> np.ndarray | None
    JPEG/RAW to BGR image decode for visualization/inference. Returns None if
    format is unsupported.
- get_frames(image_client, sources) -> (OrderedDict[name->image], list[resp])
    Batch-fetch frames. Returns decoded images and their corresponding
    ImageResponses in the same order for metadata access.
"""
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import cv2
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient


def ensure_available_sources(image_client: ImageClient, desired: List[str]) -> List[str]:
    """Return a filtered list of desired image source names that exist on the robot.

    Inputs:
    - image_client: Spot ImageClient
    - desired: list of preferred source names

    Output: list of usable source names (order preserved from desired).
    Raises RuntimeError if none are available.
    """
    available = {s.name for s in image_client.list_image_sources()}
    usable = [s for s in desired if s in available]
    if not usable:
        raise RuntimeError(f"No desired surround cameras found. Available: {sorted(available)}")
    return usable


def decode_image(resp) -> np.ndarray:
    """Decode an ImageResponse to a BGR ndarray.

    Supports JPEG and RAW (grayscale) images. Returns a 3-channel BGR image.
    """
    fmt = resp.shot.image.format
    data = resp.shot.image.data
    if fmt == image_pb2.Image.FORMAT_JPEG:
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif fmt == image_pb2.Image.FORMAT_RAW:
        rows, cols = resp.shot.image.rows, resp.shot.image.cols
        img = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = None
    return img


def get_frames(image_client: ImageClient, sources: List[str]):
    """Fetch a batch of images for the given sources.

    Inputs:
    - image_client: Spot ImageClient
    - sources: list of image source names

    Returns:
    - frames: OrderedDict mapping source name -> BGR image (np.ndarray)
    - responses: list of ImageResponse objects in the same order as request
    """
    frames: "OrderedDict[str, np.ndarray]" = OrderedDict()
    responses = image_client.get_image_from_sources(sources)
    for r in responses:
        img = decode_image(r)
        if img is not None:
            frames[r.source.name] = img
    return frames, responses
