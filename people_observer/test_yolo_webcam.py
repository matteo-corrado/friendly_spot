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

# YOLOv8 model variants to test
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
                       frame_count: int):
    """Draw performance statistics overlay."""
    h, w = frame.shape[:2]
    
    # Create semi-transparent overlay panel
    overlay = frame.copy()
    panel_h = 120
    cv2.rectangle(overlay, (10, 10), (400, panel_h), COLOR_STATS_BG, -1)
    frame_with_overlay = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Draw stats text
    y_offset = 30
    stats_lines = [
        f"Model: {model_name}",
        f"FPS: {fps:.1f}",
        f"Avg Inference: {avg_inference_ms:.1f}ms",
        f"Detections: {detection_count}",
        f"Frame: {frame_count}",
    ]
    
    for line in stats_lines:
        cv2.putText(frame_with_overlay, line, (20, y_offset), 
                   FONT, 0.6, COLOR_TEXT, 2)
        y_offset += 20
    
    # Draw instructions
    instructions = "Press 'q' to quit | 'n' for next model | 's' to save frame"
    text_size = cv2.getTextSize(instructions, FONT, 0.5, 1)[0]
    cv2.rectangle(frame_with_overlay, (10, h - 35), 
                 (text_size[0] + 20, h - 10), COLOR_STATS_BG, -1)
    cv2.putText(frame_with_overlay, instructions, (15, h - 18), 
               FONT, 0.5, COLOR_TEXT, 1)
    
    return frame_with_overlay


def test_model_on_webcam(model_name: str, description: str, 
                         confidence_threshold: float = 0.25,
                         max_frames: int = 300) -> Dict:
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Webcam opened (640x480)")
        
        # Performance tracking
        inference_times: List[float] = []
        frame_times: List[float] = []
        total_detections = 0
        frame_count = 0
        
        # Warm-up inference (first inference is slower)
        ret, warmup_frame = cap.read()
        if ret:
            _ = model.predict(warmup_frame, conf=confidence_threshold, 
                            verbose=False, device='cpu')
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
                                   verbose=False, device='cpu')
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
                    
                    # Draw detection
                    draw_detection_box(frame, x1, y1, x2, y2, 
                                      class_name, confidence, inference_time)
                    detection_count += 1
                    total_detections += 1
            
            # Calculate FPS
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
            if time.time() - last_fps_time > 0.5:  # Update FPS every 0.5s
                if frame_times:
                    fps = 1.0 / (sum(frame_times[-30:]) / min(30, len(frame_times)))
                last_fps_time = time.time()
            
            # Draw stats overlay
            avg_inference = np.mean(inference_times[-30:]) if inference_times else 0
            frame_with_stats = draw_stats_overlay(
                frame, model_name, fps, avg_inference, 
                detection_count, frame_count
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
            
            # Print summary
            print(f"\nPerformance Summary:")
            print(f"  Frames processed: {metrics['frames_processed']}")
            print(f"  Total detections: {metrics['total_detections']}")
            print(f"  Avg detections/frame: {metrics['detections_per_frame']:.2f}")
            print(f"  Avg inference time: {metrics['avg_inference_ms']:.1f}ms")
            print(f"  Min/Max inference: {metrics['min_inference_ms']:.1f}ms / {metrics['max_inference_ms']:.1f}ms")
            print(f"  Std deviation: {metrics['std_inference_ms']:.1f}ms")
            print(f"  Average FPS: {metrics['avg_fps']:.1f}")
            
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
        
        print("\nRecommendations:")
        fastest = sorted_results[0]
        print(f"  - Fastest: {fastest['model_name']} ({fastest['avg_inference_ms']:.1f}ms)")
        
        if len(sorted_results) > 1:
            slowest = sorted_results[-1]
            speedup = slowest['avg_inference_ms'] / fastest['avg_inference_ms']
            print(f"  - Slowest: {slowest['model_name']} ({slowest['avg_inference_ms']:.1f}ms)")
            print(f"  - Speed difference: {speedup:.1f}x")
        
        # Recommend model based on target FPS
        print(f"\n  For real-time tracking at 7 Hz:")
        target_ms = 1000 / 7  # ~143ms budget per frame
        suitable = [r for r in sorted_results if r['avg_inference_ms'] < target_ms]
        if suitable:
            print(f"    Suitable models: {', '.join(r['model_name'] for r in suitable)}")
        else:
            print(f"    WARNING: No models meet target (all > {target_ms:.0f}ms)")
    
    else:
        print("\nWARNING: No successful test runs")
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main()
