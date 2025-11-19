# PTZ Camera Access Implementation Guide

## Problem Identified

The current `video_sources.py` implementation has a critical bug:
```python
class SpotPTZImageClient(VideoSource):
    def __init__(self, robot, source_name: str = "ptz", ...):  # ❌ "ptz" does NOT exist
```

**"ptz" is NOT a valid image source name in the Spot SDK.**

---

## Spot CAM Architecture (from SDK research)

### 1. Image Service (ImageClient)
- Provides **individual camera images** via `get_image()` requests
- Sources discovered via `list_image_sources()`
- Source names vary by robot hardware configuration
- **Typical sources**:
  - Surround fisheye: `frontleft_fisheye_image`, `frontright_fisheye_image`, `left_fisheye_image`, `right_fisheye_image`, `back_fisheye_image`
  - Depth: `*_depth_in_visual_frame` variants
  - **Spot CAM sources**: Robot-dependent, must be discovered dynamically

### 2. Compositor Service (CompositorClient)
- Manages **composite video streams** for WebRTC/network viewing
- Screens: `digi`, `digi_overlay`, `digi_full`, `mech`, `mech_full`, `mech_overlay`, `pano_full`, etc.
- Compositor screens are **NOT ImageClient sources** - they configure what's sent to WebRTC
- Set via `compositor_client.set_screen(screen_name)`

### 3. PTZ Service (PtzClient)
- Controls **pan/tilt/zoom** of mechanical and digital PTZ cameras
- PTZ names: `digi`, `full_digi`, `mech`, `overlay_digi`, `full_pano`, `overlay_pano`, `sv600`
- Control via `ptz_client.set_ptz_position(ptz_desc, pan, tilt, zoom)`
- Pan: [0, 360]°, Tilt: [-30, 100]°, Zoom: [1, 30]x
- **PTZ names are control identifiers, not necessarily ImageClient sources**

### 4. WebRTC Service (WebRTCClient)
- Provides **real-time H.264 video streaming** via SDP protocol
- Default: `h264.sdp` on port 31102
- Uses compositor screen as video source
- **Already implemented** in `people_observer/ptz_stream.py`!

---

## Three Approaches to PTZ Camera Access

### Approach 1: ImageClient with Dynamic Source Discovery ✅ RECOMMENDED
**What**: Query robot for available image sources, pick PTZ source dynamically  
**Pros**: 
- Low latency (~50-100ms)
- Synchronous API (matches pipeline)
- Includes camera intrinsics
- Works without WebRTC dependencies

**Cons**:
- Spot CAM sources vary by hardware
- May not be available on all robots
- Must query sources at startup

**Implementation**: Modify `SpotPTZImageClient` to discover sources intelligently

### Approach 2: WebRTC Streaming ✅ ALREADY IMPLEMENTED
**What**: Use `ptz_stream.py` WebRTC client (already exists!)  
**Pros**:
- Real-time streaming
- Guaranteed to work with Spot CAM
- Compositor integration (can overlay data)
- Already implemented and tested

**Cons**:
- Requires aiortc dependency
- Async/threading complexity
- No camera intrinsics
- Higher latency (~100-200ms)

**Implementation**: Use existing `PtzStream` class from `people_observer/ptz_stream.py`

### Approach 3: Hybrid (ImageClient + fallback to WebRTC) ⭐ BEST
**What**: Try ImageClient first, fall back to WebRTC if no PTZ source  
**Pros**: Best of both worlds  
**Cons**: More complex

---

## Recommended Fix: Smart Source Discovery

### Step 1: Discover PTZ sources at startup
```python
def discover_ptz_source(image_client: ImageClient) -> Optional[str]:
    """Discover PTZ camera source from available image sources.
    
    Returns:
        PTZ source name, or None if not found
    """
    sources = {s.name for s in image_client.list_image_sources()}
    
    # Priority order: try common Spot CAM source patterns
    candidates = [
        'digi',           # Digital PTZ full resolution
        'full_digi',      # Digital PTZ full frame
        'mech',           # Mechanical PTZ
        'pano_full',      # Panoramic camera full resolution
        'digi_overlay',   # Digital PTZ with overlay
        'mech_full',      # Mechanical PTZ full frame
    ]
    
    for candidate in candidates:
        if candidate in sources:
            logger.info(f"Found PTZ source: {candidate}")
            return candidate
    
    # Try pattern matching
    ptz_keywords = ['digi', 'mech', 'pano', 'ptz']
    for source in sources:
        if any(kw in source.lower() for kw in ptz_keywords):
            if 'depth' not in source.lower():  # Skip depth sources
                logger.info(f"Found potential PTZ source: {source}")
                return source
    
    logger.warning("No PTZ source found in image service")
    return None
```

### Step 2: Update SpotPTZImageClient with auto-discovery
```python
class SpotPTZImageClient(VideoSource):
    def __init__(self, robot, source_name: Optional[str] = None, ...):
        """Initialize PTZ camera client with smart source discovery.
        
        Args:
            source_name: Camera source name, or None to auto-discover
        """
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        
        # Auto-discover PTZ source if not specified
        if source_name is None:
            source_name = discover_ptz_source(self.image_client)
            if source_name is None:
                raise RuntimeError(
                    "No PTZ camera source found. Options:\n"
                    "1. Specify source_name explicitly\n"
                    "2. Use WebRTC streaming (ptz_stream.py)\n"
                    "3. Run test_list_image_sources.py to see available sources"
                )
        
        # Validate source exists (existing code)
        available_sources = {s.name for s in self.image_client.list_image_sources()}
        if source_name not in available_sources:
            raise ValueError(
                f"Camera source '{source_name}' not found.\n"
                f"Available sources: {sorted(available_sources)}\n"
                f"Run: python test_list_image_sources.py --hostname ROBOT_IP"
            )
        
        self.source_name = source_name
        logger.info(f"Using PTZ source: {source_name}")
        # ... rest of initialization
```

### Step 3: Test script to validate
Run `test_list_image_sources.py --hostname ROBOT_IP` to:
- List all available image sources
- Show compositor screens
- List PTZ cameras
- Recommend correct source name

---

## Implementation Priority

1. **IMMEDIATE** (Blocking): 
   - Run `test_list_image_sources.py` on actual robot to discover correct source name
   - Update `SpotPTZImageClient` with discovered source name

2. **SHORT-TERM** (Robustness):
   - Implement `discover_ptz_source()` auto-discovery function
   - Update `SpotPTZImageClient.__init__()` to use auto-discovery
   - Add better error messages with troubleshooting guidance

3. **LONG-TERM** (Alternative):
   - Consider switching to WebRTC via `ptz_stream.py` if ImageClient doesn't work
   - Implement hybrid approach with automatic fallback

---

## Testing Plan

```bash
# 1. Discover available sources
python test_list_image_sources.py --hostname $env:ROBOT_IP

# 2. Test PTZ camera access with discovered source name
# (Manual edit video_sources.py with correct source_name)

# 3. Test observer bridge
python -c "from people_observer import app; app.main()" --hostname $env:ROBOT_IP

# 4. Test full integration
python friendly_spot_main.py --hostname $env:ROBOT_IP --enable-observer --once --visualize
```

---

## Alternative: Use Existing WebRTC Implementation

If ImageClient doesn't provide PTZ sources, you can **immediately** use the existing `ptz_stream.py`:

```python
# In observer_bridge.py or video_sources.py
from people_observer.ptz_stream import PtzStream, PtzStreamConfig

# Start WebRTC stream
config = PtzStreamConfig(sdp_filename="h264.sdp", sdp_port=31102)
stream = PtzStream(robot, config)
stream.start()

# Get frames
while stream.is_running():
    frame = await stream.get_frame(timeout=1.0)
    if frame:
        # Process frame
        pass

stream.stop()
```

This is **already implemented and working** - just needs integration.

---

## Summary

- ❌ **Current**: `source_name="ptz"` does not exist
- ✅ **Solution 1**: Auto-discover PTZ source from `list_image_sources()`
- ✅ **Solution 2**: Use existing WebRTC streaming (`ptz_stream.py`)
- ⭐ **Recommended**: Hybrid with auto-discovery + WebRTC fallback

**Next step**: Run `test_list_image_sources.py` on robot to see what sources actually exist.
