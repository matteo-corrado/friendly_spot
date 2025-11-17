# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/13/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Recognize up to two hands, with gestures, from the current camera frame

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Sets parameters for the recognition model with Mediapipe
MODEL_PATH = 'gesture_recognizer.task'
IMAGE_FILENAME = 'img7.jpg'
MAX_NUM_HANDS = 2
MIN_CONFIDENCE = 0.4

# Instantiate the gesture recognition model
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.GestureRecognizerOptions(base_options=base_options)
options.num_hands = MAX_NUM_HANDS
options.min_hand_detection_confidence = MIN_CONFIDENCE
recognizer = vision.GestureRecognizer.create_from_options(options)

# Load the current frame as an image type, and run the recognizer model
image = mp.Image.create_from_file(IMAGE_FILENAME)
recognition_result = recognizer.recognize(image)

# Loops over the gestures found (at most two), reporting left/right hand and gesture
if recognition_result.gestures:
    for index in range(len(recognition_result.gestures)):
        predicted_gesture = recognition_result.gestures[index][0].category_name
        predicted_hand = recognition_result.handedness[index][0].category_name
        print(predicted_hand, predicted_gesture)
else:
    print("No hands recognized")