"""Test YOLO models on local laptop webcam with performance benchmarking.

Tests each YOLOv8 model variant on live webcam feed and measures:
- Inference time per frame
- FPS (frames per second)
- Detection latency
- Memory usage

Press 'q' to quit or 'n' to switch to next model.
"""
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config for device and other settings
from people_observer.config import YOLO_DEVICE, LOOP_HZ as TARGET_LOOP_HZ_CONFIG

# Test configuration constants
# Note: Uses YOLO_DEVICE and TARGET_LOOP_HZ_CONFIG from config for consistency
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_MAX_FRAMES = 300
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
WARM_UP_FRAMES = 1  # Number of warm-up inferences

# Visualization constants
OVERLAY_ALPHA = 0.7
OVERLAY_BETA = 0.3
STATS_PANEL_WIDTH = 400
STATS_PANEL_BASE_HEIGHT = 120
STATS_PANEL_CONFIDENCE_EXTRA = 20
STATS_TEXT_Y_START = 30
STATS_TEXT_Y_SPACING = 20
FPS_UPDATE_INTERVAL_SEC = 0.5
FPS_WINDOW_FRAMES = 30  # Rolling window for FPS calculation
INFERENCE_WINDOW_FRAMES = 30  # Rolling window for inference time
CONFIDENCE_WINDOW_FRAMES = 30  # Rolling window for confidence display

# Model selection constants
MIN_RECOMMENDED_THRESHOLD = 0.25  # Don't recommend thresholds below this
CONFIDENCE_STD_DEV_MULTIPLIER = 1.0  # Std deviations below mean for threshold

# YOLOv8 model variants to test
# Note: config.DEFAULT_YOLO_MODEL sets the model used by the main app (currently yolov8x.pt)
YOLO_MODELS = [
    ("yolov8n.pt", "Nano"),
    ("yolov8s.pt", "Small"),
    ("yolov8m.pt", "Medium"),
    ("yolov8l.pt", "Large"),
    ("yolov8x.pt", "Extra Large"),
]

# Colors for visualization (BGR)
COLOR_BOX = (0, 255, 0)  # Green
COLOR_TEXT_BG = (0, 0, 0)  # Black
COLOR_TEXT = (255, 255, 255)  # White
COLOR_STATS_BG = (0, 0, 0)  # Black with transparency
FONT = cv2.FONT_HERSHEY_SIMPLEX
DEVICE = "cuda" if YOLO_DEVICE == "cuda" else "cpu"

def draw_detection_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                       class_name: str, confidence: float, inference_time_ms: float):
    """Draw a single detection box with label and timing info."""
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
    
    # Create label with timing
    label = f"{class_name} {confidence:.2f} ({inference_time_ms:.0f}ms)"
    text_size = cv2.getTextSize(label, FONT, 0.5, 1)[0]
    
    # Draw label background
    cv2.rectangle(frame, (x1, y1 - text_size[1] - 8), 
                 (x1 + text_size[0] + 8, y1), COLOR_TEXT_BG, -1)
    
    # Draw label text
    cv2.putText(frame, label, (x1 + 4, y1 - 4), FONT, 0.5, COLOR_TEXT, 1)


def draw_stats_overlay(frame: np.ndarray, model_name: str, fps: float, 
                       avg_inference_ms: float, detection_count: int,
                       frame_count: int, avg_confidence: float = None):
    """Draw performance statistics overlay."""
    h, w = frame.shape[:2]
    
    # Create semi-transparent overlay panel
    overlay = frame.copy()
    panel_h = STATS_PANEL_BASE_HEIGHT + (STATS_PANEL_CONFIDENCE_EXTRA if avg_confidence is not None else 0)
    cv2.rectangle(overlay, (10, 10), (STATS_PANEL_WIDTH, panel_h), COLOR_STATS_BG, -1)
    frame_with_overlay = cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, OVERLAY_BETA, 0)
    
    # Draw stats text
    y_offset = STATS_TEXT_Y_START
    stats_lines = [
        f"Model: {model_name}",
        f"FPS: {fps:.1f}",
        f"Avg Inference: {avg_inference_ms:.1f}ms",
        f"Detections: {detection_count}",
        f"Frame: {frame_count}",
    ]
    
    if avg_confidence is not None:
        stats_lines.append(f"Avg Confidence: {avg_confidence:.2f}")
    
    for line in stats_lines:
        cv2.putText(frame_with_overlay, line, (20, y_offset), 
                   FONT, 0.6, COLOR_TEXT, 2)
        y_offset += STATS_TEXT_Y_SPACING
    
    # Draw instructions
    instructions = "Press 'q' to quit | 'n' for next model | 's' to save frame"
    text_size = cv2.getTextSize(instructions, FONT, 0.5, 1)[0]
    cv2.rectangle(frame_with_overlay, (10, h - 35), 
                 (text_size[0] + 20, h - 10), COLOR_STATS_BG, -1)
    cv2.putText(frame_with_overlay, instructions, (15, h - 18), 
               FONT, 0.5, COLOR_TEXT, 1)
    
    return frame_with_overlay


def test_model_on_webcam(model_name: str, description: str, 
                         confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                         max_frames: int = DEFAULT_MAX_FRAMES) -> Dict:
    """Test a single YOLO model on webcam feed.
    
    Returns:
        Dict with performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_name} ({description})")
    print('='*60)
    
    try:
        from ultralytics import YOLO
        
        # Load model
        print(f"Loading {model_name}...")
        model = YOLO(model_name)
        print(f"Model loaded")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam")
            return {'error': 'Webcam not available'}
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
        
        print(f"Webcam opened ({WEBCAM_WIDTH}x{WEBCAM_HEIGHT})")
        
        # Performance tracking
        inference_times: List[float] = []
        frame_times: List[float] = []
        confidence_scores: List[float] = []  # Track all detection confidences
        total_detections = 0
        frame_count = 0
        
        # Warm-up inference (first inference is slower)
        ret, warmup_frame = cap.read()
        if ret:
            _ = model.predict(warmup_frame, conf=confidence_threshold, 
                            verbose=False, device=DEVICE)
            print("Warm-up inference complete")
        
        # Main loop
        last_fps_time = time.time()
        fps = 0.0
        
        while frame_count < max_frames:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Run inference
            inference_start = time.time()
            results = model.predict(frame, conf=confidence_threshold, 
                                   verbose=False, device=YOLO_DEVICE)
            inference_time = (time.time() - inference_start) * 1000  # ms
            inference_times.append(inference_time)
            
            # Process detections
            detection_count = 0
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # Track confidence scores
                    confidence_scores.append(confidence)
                    
                    # Draw detection
                    draw_detection_box(frame, x1, y1, x2, y2, 
                                      class_name, confidence, inference_time)
                    detection_count += 1
                    total_detections += 1
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
            if time.time() - last_fps_time > FPS_UPDATE_INTERVAL_SEC:
                if frame_times:
                    fps = 1.0 / (sum(frame_times[-FPS_WINDOW_FRAMES:]) / min(FPS_WINDOW_FRAMES, len(frame_times)))
                last_fps_time = time.time()
            
            # Draw stats overlay
            avg_inference = np.mean(inference_times[-INFERENCE_WINDOW_FRAMES:]) if inference_times else 0
            avg_conf = np.mean(confidence_scores[-CONFIDENCE_WINDOW_FRAMES:]) if confidence_scores else None
            frame_with_stats = draw_stats_overlay(
                frame, model_name, fps, avg_inference, 
                detection_count, frame_count, avg_conf
            )
            
            # Display frame
            cv2.imshow(f"YOLO Webcam Test - {model_name}", frame_with_stats)
            
            frame_count += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nUser quit")
                cap.release()
                cv2.destroyAllWindows()
                return {'user_quit': True}
            elif key == ord('n'):
                print("\nSwitching to next model")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"yolo_test_{model_name.replace('.pt', '')}_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_stats)
                print(f"Saved frame to {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate final metrics
        if inference_times and frame_times:
            metrics = {
                'model_name': model_name,
                'success': True,
                'frames_processed': frame_count,
                'total_detections': total_detections,
                'avg_inference_ms': np.mean(inference_times),
                'min_inference_ms': np.min(inference_times),
                'max_inference_ms': np.max(inference_times),
                'std_inference_ms': np.std(inference_times),
                'avg_fps': 1.0 / np.mean(frame_times),
                'detections_per_frame': total_detections / frame_count if frame_count > 0 else 0,
            }
            
            # Add confidence statistics if we have detections
            if confidence_scores:
                metrics['avg_confidence'] = np.mean(confidence_scores)
                metrics['min_confidence'] = np.min(confidence_scores)
                metrics['max_confidence'] = np.max(confidence_scores)
                metrics['std_confidence'] = np.std(confidence_scores)
            
            # Print summary
            print(f"\nPerformance Summary:")
            print(f"  Frames processed: {metrics['frames_processed']}")
            print(f"  Total detections: {metrics['total_detections']}")
            print(f"  Avg detections/frame: {metrics['detections_per_frame']:.2f}")
            print(f"  Avg inference time: {metrics['avg_inference_ms']:.1f}ms")
            print(f"  Min/Max inference: {metrics['min_inference_ms']:.1f}ms / {metrics['max_inference_ms']:.1f}ms")
            print(f"  Std deviation: {metrics['std_inference_ms']:.1f}ms")
            print(f"  Average FPS: {metrics['avg_fps']:.1f}")
            
            if confidence_scores:
                print(f"\nConfidence Statistics:")
                print(f"  Average: {metrics['avg_confidence']:.3f}")
                print(f"  Min/Max: {metrics['min_confidence']:.3f} / {metrics['max_confidence']:.3f}")
                print(f"  Std deviation: {metrics['std_confidence']:.3f}")
            
            return metrics
        else:
            return {'error': 'No frames processed'}
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    print("="*60)
    print("YOLO Webcam Performance Test")
    print("="*60)
    print("\nThis will test each YOLO model on your laptop webcam")
    print("and measure inference time and detection performance.")
    
    try:
        from ultralytics import YOLO
        print("\nUltralytics YOLO imported successfully")
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("  Run: pip install ultralytics opencv-python")
        sys.exit(1)
    
    # Test each model
    all_results = []
    
    for model_name, description in YOLO_MODELS:
        result = test_model_on_webcam(model_name, description)
        
        if result.get('user_quit'):
            print("\nUser requested quit. Stopping tests.")
            break
        
        all_results.append(result)
        
        # Short pause between models
        if not result.get('error'):
            print("\nWaiting 2 seconds before next model...")
            time.sleep(2)
    
    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    successful_results = [r for r in all_results if r.get('success')]
    
    if successful_results:
        print("\nModel Performance Ranking (by inference time):")
        sorted_results = sorted(successful_results, 
                               key=lambda x: x['avg_inference_ms'])
        
        for i, result in enumerate(sorted_results, 1):
            print(f"\n{i}. {result['model_name']}")
            print(f"   Avg inference: {result['avg_inference_ms']:.1f}ms")
            print(f"   Avg FPS: {result['avg_fps']:.1f}")
            print(f"   Detections/frame: {result['detections_per_frame']:.2f}")
            if 'avg_confidence' in result:
                print(f"   Avg confidence: {result['avg_confidence']:.3f}")
        
        print("\nRecommendations:")
        fastest = sorted_results[0]
        print(f"  - Fastest: {fastest['model_name']} ({fastest['avg_inference_ms']:.1f}ms)")
        
        if len(sorted_results) > 1:
            slowest = sorted_results[-1]
            speedup = slowest['avg_inference_ms'] / fastest['avg_inference_ms']
            print(f"  - Slowest: {slowest['model_name']} ({slowest['avg_inference_ms']:.1f}ms)")
            print(f"  - Speed difference: {speedup:.1f}x")
        
        # Recommend model based on target FPS
        print(f"\n  For real-time tracking at {TARGET_LOOP_HZ_CONFIG} Hz:")
        target_ms = 1000 / TARGET_LOOP_HZ_CONFIG
        suitable = [r for r in sorted_results if r['avg_inference_ms'] < target_ms]
        if suitable:
            print(f"    Suitable models: {', '.join(r['model_name'] for r in suitable)}")
        else:
            print(f"    WARNING: No models meet target (all > {target_ms:.0f}ms)")
        
        # Confidence threshold recommendations
        results_with_conf = [r for r in successful_results if 'avg_confidence' in r]
        if results_with_conf:
            print(f"\n  Confidence Threshold Recommendations:")
            print(f"  (to reduce false positives, use threshold = avg - {CONFIDENCE_STD_DEV_MULTIPLIER}*std_dev)")
            for result in results_with_conf:
                avg = result['avg_confidence']
                std = result['std_confidence']
                recommended = max(MIN_RECOMMENDED_THRESHOLD, avg - (CONFIDENCE_STD_DEV_MULTIPLIER * std))
                print(f"    {result['model_name']}: >= {recommended:.2f} (avg={avg:.3f}, std={std:.3f})")
    
    else:
        print("\nWARNING: No successful test runs")
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main()
