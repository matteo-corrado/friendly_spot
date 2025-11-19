# friendly_spot

Behavioral and vision scripts for Spot: person-to-PTZ aiming with YOLO, the Boston Dynamics Fetch tutorial flow (Network Compute Bridge), image extraction helpers, and a facial recognition PoC.

## What’s here
- `Behavioral/spot_yolo_person_to_ptz.py` — Run YOLO on surround fisheyes, map pixel X to robot-frame bearing (per-camera yaw/FOV), and slew the Spot CAM PTZ.
- `Behavioral/fetch/network_compute_server.py` — TensorFlow object detection worker registering with Spot’s directory (Network Compute Bridge).
- `Behavioral/fetch/fetch.py` — Client side of Fetch: request detections, walk, grasp, carry, and drop.
- `Behavioral/fetch/capture_images.py` — Save images from a chosen camera source.
- `Behavioral/human_image_extractor.py` — WIP scaffold: capture images and emulate parts of the Fetch flow.
- `Facial Recognition/trainMemory.py` — LBPH trainer + webcam recognizer demo (requires `opencv-contrib-python`).

## Requirements
Install the Boston Dynamics Spot SDK wheels first from the sibling `spot-sdk/prebuilt` directory in this workspace, then the Python dependencies in `requirements.txt`.

Authentication: your venv `Activate.ps1` supplies credentials/tokens; do not include user/password on the CLI. Scripts that rely on `bosdyn.client.util.authenticate` will use your stored token. Avoid committing secrets.

Notes:
- The Fetch tutorial files depend on live robot services (Image, NetworkComputeBridge, Manipulation, PTZ) and time sync.
- The TensorFlow Object Detection API is required for `network_compute_server.py` (`object_detection.utils.label_map_util`). Install per the TF Models instructions, or point to a model directory exported with a SavedModel and a matching labels `.pbtxt`.

## Install
1) Activate your venv (with auth in `Activate.ps1`).
2) Install Spot SDK wheels from `spot-sdk/prebuilt/*.whl`.
3) `pip install -r requirements.txt`.

## Run snippets (no credentials on CLI)
- YOLO → PTZ aiming
  - `python Behavioral/spot_yolo_person_to_ptz.py --robot <ROBOT_IP>`
  - Adjust constants in the script for compositor screen, bitrate, and camera yaw/FOV if needed.

- Network Compute worker (TensorFlow)
  - `python Behavioral/fetch/network_compute_server.py -m <MODEL_DIR> <LABELS.pbtxt> -n <SERVICE_NAME> --hostname <ROBOT_IP>`
  - This registers the worker; ensure the TF OD API is installed and the model is valid.

- Fetch client
  - `python Behavioral/fetch/fetch.py --hostname <ROBOT_IP> --ml-service <SERVICE_NAME> --model <MODEL_NAME> --person-model <PERSON_MODEL>`

- Capture images
  - `python Behavioral/fetch/capture_images.py --hostname <ROBOT_IP> --image-source frontleft_fisheye_image --folder out/`

- Facial recognition PoC
  - Prepare a `dataset/` with one subfolder per person of images, then:
  - `python "Facial Recognition/trainMemory.py"`

## Conventions
- Always perform time sync before issuing commands. Use a `LeaseKeepAlive` during motion/manipulation sequences.
- Surround camera names: `frontleft_fisheye_image`, `frontright_fisheye_image`, `left_fisheye_image`, `right_fisheye_image`, `back_fisheye_image`.
- PTZ streaming is configured via `CompositorClient` and `StreamQualityClient`; tune `COMPOSITOR_SCREEN` and bitrate for your setup.

## Troubleshooting
- Empty detections: verify model and labels, confirm `NetworkComputeBridge` worker is registered, and check Wi‑Fi.
- PTZ not moving: confirm CAM PTZ name and permissions; compositor screen must show PTZ.
- Import error `object_detection`: install TensorFlow Object Detection API.
- Torch/TensorFlow wheels on Windows: prefer CPU builds unless you have a compatible GPU/CUDA stack.
