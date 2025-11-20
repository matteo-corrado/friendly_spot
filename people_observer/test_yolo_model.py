# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/19/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: YOLO model loading verification test ensuring all model variants initialize correctly
# Acknowledgements: Ultralytics YOLO, Claude for test implementation

"""Quick test to verify YOLO model loads correctly."""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# YOLOv8 model variants to test
# Note: config.DEFAULT_YOLO_MODEL sets the model used by the main app
YOLO_MODELS = [
    ("yolo11n-seg.pt", "Nano - fastest, least accurate"),
    ("yolo11s-seg.pt", "Small"),
    ("yolo11m-seg.pt", "Medium"),
    ("yolo11l-seg.pt", "Large"),
    ("yolo11x-seg.pt", "Extra Large - slowest, most accurate"),
]

def check_model(model_name: str, description: str):
    """Test loading and inspecting a single YOLO model."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name} ({description})")
    print('='*60)
    
    try:
        from ultralytics import YOLO
        
        # Check if model exists locally
        model_path = Path(__file__).parent.parent / model_name
        if model_path.exists():
            print(f"Found locally at: {model_path}")
            model = YOLO(str(model_path))
        else:
            print(f"Not found locally, downloading from Ultralytics...")
            model = YOLO(model_name)
            print(f"Downloaded successfully")
        
        print(f"Model loaded successfully")
        
        # Display model information
        print(f"\nModel Information:")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Total classes: {len(model.names)}")
        print(f"  Person class: ID 0 = '{model.names.get(0, 'N/A')}'")
        
        # Check if all models have same classes
        return {
            'name': model_name,
            'loaded': True,
            'num_classes': len(model.names),
            'class_names': model.names,
            'person_class': model.names.get(0, None)
        }
        
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return {
            'name': model_name,
            'loaded': False,
            'error': str(e)
        }


def main():
    print("="*60)
    print("YOLO Model Test Suite")
    print("="*60)
    
    try:
        from ultralytics import YOLO
        print("\nUltralytics YOLO imported successfully")
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("  Run: pip install ultralytics")
        sys.exit(1)
    
    # Show available models
    print("\nYOLOv8 Models to Test:")
    for model_name, desc in YOLO_MODELS:
        print(f"  - {model_name}: {desc}")
    print("\nNote: First-time use will auto-download models (~6-140 MB each)")
    
    # Test each model
    results = []
    for model_name, description in YOLO_MODELS:
        result = check_model(model_name, description)
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    loaded_models = [r for r in results if r.get('loaded')]
    failed_models = [r for r in results if not r.get('loaded')]
    
    print(f"\nSuccessfully loaded: {len(loaded_models)}/{len(results)} models")
    if loaded_models:
        for r in loaded_models:
            print(f"  - {r['name']}: {r['num_classes']} classes")
    
    if failed_models:
        print(f"\nFailed to load: {len(failed_models)} models")
        for r in failed_models:
            print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")
    
    # Compare class consistency across models
    if len(loaded_models) > 1:
        print("\nClass Consistency Check:")
        first_model = loaded_models[0]
        all_consistent = True
        
        for model in loaded_models[1:]:
            if model['num_classes'] != first_model['num_classes']:
                print(f"  WARNING: {model['name']} has {model['num_classes']} classes, expected {first_model['num_classes']}")
                all_consistent = False
            elif model['person_class'] != first_model['person_class']:
                print(f"  WARNING: {model['name']} person class is '{model['person_class']}', expected '{first_model['person_class']}'")
                all_consistent = False
        
        if all_consistent:
            print(f"  All models have consistent classes ({first_model['num_classes']} classes)")
            print(f"  Person class ID 0 = '{first_model['person_class']}' across all models")
    
    # Show class names from first successful model
    if loaded_models:
        print("\nClass Names (from first model):")
        first_model = loaded_models[0]
        class_names = first_model['class_names']
        
        # Show first 20 classes
        print("  First 20 classes:")
        for i in range(min(20, len(class_names))):
            print(f"    {i:2d}: {class_names[i]}")
        
        if len(class_names) > 20:
            print(f"  ... and {len(class_names) - 20} more classes")
    
    # Final status
    print("\n" + "="*60)
    if len(loaded_models) == len(results):
        print("ALL TESTS PASSED - All models ready to use!")
        sys.exit(0)
    else:
        print(f"PARTIAL SUCCESS - {len(loaded_models)}/{len(results)} models loaded")
        sys.exit(1)


if __name__ == "__main__":
    main()
