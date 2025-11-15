# People Observer

Detect people in Spot's surround fisheye streams and aim the Spot CAM PTZ to center them. Built to extend from a fast bearing-only approach to a transform-aware approach using Boston Dynamics frame helpers and image snapshots.

Key references:
- Fetch example (Network Compute Bridge): use transforms_snapshot and `frame_helpers.get_a_tform_b`.
- Visualizer example: `get_image_from_sources`, snapshot-based transforms.
- Spot CAM: `CompositorClient`, `StreamQualityClient`, `PtzClient`.

## Modes
- bearing (default):
  - YOLO detect persons on surround frames.
  - Map pixel X → yaw via HFOV; add camera yaw derived from transforms (fallback to config yaw/HFOV if needed).
  - Apply fixed tilt policy.
- transform (planned):
  - If pinhole intrinsics and depth are available, unproject pixel → ray in camera frame, transform to PTZ frame, compute pan/tilt.
  - For fisheye (no pinhole intrinsics), retain bearing-only or use a calibrated fisheye model.

## Why not hardcode camera yaws/HFOVs?
- Camera directions and fields of view can vary across hardware/firmware. We compute camera yaw from the transforms snapshot when possible and keep HFOVs centralized in `config.py` for explicit updates from BD documentation.

## Run
Auth is sourced from venv.

```
python -m friendly_spot.people_observer.app --hostname <ROBOT_IP>
```

## Notes
- Always ensure time sync before using transforms; handled in io_robot.connect.
- PTZ only: no lease required. If you add mobility, use a LeaseKeepAlive pattern.
- Use the Network Compute Bridge if you need 3D positions of detections (as in Fetch) rather than bearing-only.
