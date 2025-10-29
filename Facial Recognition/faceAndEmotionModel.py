# NEEDS A FEW UPDATES, GET EMOTION MODEL WORKING FIRST

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import os

class AdvancedFaceEmotionAnalyzer:
    def __init__(self, dataset_path=None):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load emotion model
        try:
            self.emotion_model = load_model('emotion_model.h5')
        except:
            print("Emotion model not found. Install it first.")
            self.emotion_model = None
        
        # Load face recognizer
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_names = {}
        self.trained = False
        
        if dataset_path:
            self.train_face_recognizer(dataset_path)
        
        self.emotions = [
            'Angry', 'Disgusted', 'Fearful', 'Happy',
            'Neutral', 'Sad', 'Surprised'
        ]
        
        self.emotion_colors = {
            'Angry': (0, 0, 255),
            'Disgusted': (0, 165, 255),
            'Fearful': (75, 0, 130),
            'Happy': (0, 255, 0),
            'Neutral': (128, 128, 128),
            'Sad': (255, 0, 0),
            'Surprised': (255, 255, 0)
        }
        
        # Store person-emotion data
        self.person_emotions = {}
    
    def train_face_recognizer(self, dataset_path):
        """
        Train face recognizer from dataset
        """
        print("Training face recognizer...")
        face_data = []
        face_labels = []
        current_label = 0
        
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            self.label_names[current_label] = person_name
            
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                img = cv2.imread(image_path)
                
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 1.1, 5, minSize=(30, 30)
                )
                
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y + h, x:x + w]
                    face_data.append(face_roi)
                    face_labels.append(current_label)
            
            current_label += 1
        
        if len(face_data) > 0:
            self.face_recognizer.train(face_data, np.array(face_labels))
            self.trained = True
            print(f"Face recognizer trained on {len(self.label_names)} people")
    
    def detect_emotion(self, face_roi_gray):
        """
        Detect emotion from face
        """
        if self.emotion_model is None:
            return 'Unknown', 0.0
        
        face_roi = cv2.resize(face_roi_gray, (48, 48))
        face_roi = face_roi.astype('float') / 255.0
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)
        
        predictions = self.emotion_model.predict(face_roi, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        emotion = self.emotions[emotion_idx]
        confidence = predictions[0][emotion_idx]
        
        return emotion, confidence
    
    def recognize_person(self, face_roi_gray):
        """
        Recognize person from face
        """
        if not self.trained:
            return 'Unknown', 0.0