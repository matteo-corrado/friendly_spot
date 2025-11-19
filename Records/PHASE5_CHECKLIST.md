# Phase 5: Testing & Cleanup Checklist

**Status:** Optional (v2.0 structure is functional)  
**Purpose:** Validate refactored code with robot hardware and clean up deprecated files

---

## Pre-Testing Setup

- [ ] **Backup current workspace**
  ```powershell
  Copy-Item -Recurse friendly_spot friendly_spot_backup
  ```

- [ ] **Set PYTHONPATH** (if not in root)
  ```powershell
  $env:PYTHONPATH = "C:\Users\corra\Spot_SDK_Master\friendly_spot"
  ```

- [ ] **Set robot credentials**
  ```powershell
  $env:BOSDYN_CLIENT_USERNAME = "user"
  $env:BOSDYN_CLIENT_PASSWORD = "password"
  ```

---

## Import Validation (No Robot)

- [x] ✅ Run import test
  ```powershell
  python tests/test_imports.py
  ```
  **Result:** All 6 modules import successfully

- [ ] Test webcam mode (no robot required)
  ```powershell
  python friendly_spot_main.py --webcam
  ```
  **Expected:** Webcam opens, YOLO detection runs, visualizations display

- [ ] Test perception module standalone
  ```powershell
  python -c "from src.perception import YoloDetector; print('OK')"
  ```

- [ ] Test behavior module standalone
  ```powershell
  python -c "from src.behavior import ComfortModel; print('OK')"
  ```

---

## Robot Connection Tests

- [ ] **Test robot connection**
  ```powershell
  python -c "from src.robot import create_robot; r = create_robot('192.168.80.3'); print('Connected:', r.is_powered_on())"
  ```

- [ ] **Test image acquisition**
  ```powershell
  python tests/test_image_sources.py --robot 192.168.80.3
  ```
  **Expected:** Captures frames from all cameras, displays in grid

- [ ] **Test PTZ control** (if Spot CAM available)
  ```powershell
  python tests/test_ptz_convention.py --robot 192.168.80.3
  ```
  **Expected:** PTZ camera moves to test positions (0°, 45°, -45° pan)

---

## Behavior Validation

- [ ] **Test MAINTAIN_DISTANCE behavior**
  - Run: `python friendly_spot_main.py --robot 192.168.80.3`
  - Stand 2-3m from robot
  - **Expected:** Robot stands in place

- [ ] **Test GO_CLOSE behavior** (critical - new trajectory command)
  - Stand >3m from robot
  - **Expected:** Robot walks toward you using SE2 trajectory
  - **Validate:** Uses vision frame, stops at target_distance

- [ ] **Test GO_AWAY behavior**
  - Stand <1.5m from robot
  - **Expected:** Robot backs away

- [ ] **Test EXPLORE behavior**
  - Remove all people from view
  - **Expected:** Robot rotates to scan area

- [ ] **Test behavior debouncing**
  - Walk in/out of 2m boundary repeatedly
  - **Expected:** No rapid oscillation (2s cooldown)

---

## PTZ Tracking Tests

- [ ] **Test single person tracking**
  - Run: `python friendly_spot_main.py --robot 192.168.80.3 --ptz-only`
  - Walk left/right
  - **Expected:** PTZ follows person smoothly

- [ ] **Test multi-person priority**
  - Two people in view
  - **Expected:** Tracks nearest person (if depth enabled)

- [ ] **Test camera fallback**
  - On robot without Spot CAM
  - **Expected:** Falls back to hand → pano → fisheye (check logs)

---

## Edge Cases

- [ ] **Test no detections**
  - Empty room, no people
  - **Expected:** EXPLORE behavior, no crashes

- [ ] **Test detection loss**
  - Person leaves frame suddenly
  - **Expected:** Graceful fallback to EXPLORE

- [ ] **Test lease conflicts**
  - Tablet controls robot, then run script
  - **Expected:** Error message or force_take_lease works

- [ ] **Test E-Stop**
  - Press physical E-Stop during behavior
  - **Expected:** Commands stop, error logged

---

## Performance Tests

- [ ] **Measure detection FPS** (with GPU)
  - **Target:** >15 FPS on 2 cameras
  - **Tool:** Add FPS counter to visualization

- [ ] **Measure trajectory latency**
  - Time from detection → command → robot movement
  - **Target:** <1s end-to-end

- [ ] **Check memory usage**
  - Run for 10 minutes
  - **Expected:** No memory leaks (stable RAM usage)

---

## Data File Migration

- [ ] **Move YOLO models**
  ```powershell
  Move-Item yolov8n*.pt data/models/
  ```

- [ ] **Create output directories**
  ```powershell
  New-Item -ItemType Directory -Force data/outputs/logs
  New-Item -ItemType Directory -Force data/outputs/videos
  New-Item -ItemType Directory -Force data/outputs/images
  ```

- [ ] **Update config paths**
  - Edit `src/perception/config.py`: `DEFAULT_YOLO_MODEL = "data/models/yolov8n-seg.pt"`

- [ ] **Test with new paths**
  ```powershell
  python friendly_spot_main.py --webcam
  ```

---

## Cleanup & Archival

- [ ] **Archive deprecated files**
  ```powershell
  New-Item -ItemType Directory -Force deprecated
  Move-Item robot_io.py deprecated/
  Move-Item behavior_executor.py deprecated/
  Move-Item behavior_planner.py deprecated/
  Move-Item video_sources.py deprecated/
  Move-Item run_pipeline.py deprecated/
  Move-Item unified_visualization.py deprecated/
  Move-Item detection_types.py deprecated/
  # ... (all files now in src/)
  ```

- [ ] **Update .gitignore**
  ```gitignore
  # Add to .gitignore
  deprecated/
  data/models/*.pt
  data/outputs/
  __pycache__/
  *.pyc
  .vscode/
  ```

- [ ] **Remove backup** (after validation)
  ```powershell
  Remove-Item -Recurse friendly_spot_backup
  ```

---

## Documentation Updates

- [ ] **Update main README**
  ```powershell
  Move-Item README_v2.md README.md -Force
  ```

- [ ] **Add CHANGELOG**
  - Create `CHANGELOG.md` documenting v2.0 changes

- [ ] **Update requirements.txt**
  - Verify all dependencies listed
  - Add version pins if needed

---

## Git Operations

- [ ] **Commit refactored code**
  ```powershell
  git add src/ tests/ data/ docs/
  git commit -m "refactor: Modular architecture v2.0 with comprehensive docs"
  ```

- [ ] **Tag release**
  ```powershell
  git tag -a v2.0.0 -m "Version 2.0.0: Modular refactor complete"
  ```

- [ ] **Push to remote**
  ```powershell
  git push origin main --tags
  ```

---

## Final Validation Checklist

- [ ] All import tests pass
- [ ] Webcam mode works
- [ ] Robot connection works
- [ ] All 5 behaviors execute correctly
- [ ] GO_CLOSE uses SE2 trajectory (not bearing command)
- [ ] PTZ tracking follows person
- [ ] Camera fallback works
- [ ] No memory leaks after 10min
- [ ] Documentation accurate
- [ ] YOLO models in data/models/
- [ ] Deprecated files archived
- [ ] Git tagged v2.0.0

---

## Rollback Plan (If Issues Found)

**If critical issues discovered:**

1. **Restore backup**
   ```powershell
   Remove-Item -Recurse friendly_spot
   Copy-Item -Recurse friendly_spot_backup friendly_spot
   ```

2. **Document issues**
   - Add to `REFACTOR_STATUS.md` Issues section
   - Create GitHub issues for tracking

3. **Incremental fix**
   - Fix one issue at a time
   - Re-run relevant tests after each fix
   - Commit fixes separately

---

## Success Criteria

✅ **Ready for Production** when:
- All tests pass
- Robot behaviors work as expected
- Documentation matches implementation
- No regressions from v1.0 functionality
- Performance meets targets (>15 FPS, <1s latency)

---

**Estimated Time:** 2-4 hours (depends on robot hardware availability)

**Priority:** Medium (v2.0 structure already functional, this validates with hardware)

**Blocker:** Robot hardware access required for full validation
