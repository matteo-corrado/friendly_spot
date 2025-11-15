import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = 'gesture_recognizer.task'
IMAGE_FILENAME = 'img7.jpg'
MAX_NUM_HANDS = 2

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(base_options=base_options)
options.num_hands = MAX_NUM_HANDS
options.min_hand_detection_confidence = 0.4
recognizer = vision.GestureRecognizer.create_from_options(options)

image = mp.Image.create_from_file(IMAGE_FILENAME)
recognition_result = recognizer.recognize(image)

if recognition_result.gestures:
    for index in range(len(recognition_result.gestures)):
        predicted_gesture = recognition_result.gestures[index][0].category_name
        predicted_hand = recognition_result.handedness[index][0].category_name
        print(predicted_hand, predicted_gesture)
else:
    print("No hands recognized")