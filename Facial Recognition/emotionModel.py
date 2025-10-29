import cv2
import numpy as np
from keras.models import load_model # leading to a segmentation fault
from keras.preprocessing.image import img_to_array

class EmotionDetector:
    def __init__(self):
        # Load pre-trained models
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Emotion detection model
        # Download from: https://github.com/isseu/emotion-recognition-neural-networks
        self.emotion_model_path = 'emotion_model.h5'
        self.emotion_model = self.load_emotion_model()
        
        # Emotion labels
        self.emotions = [
            'Angry', 'Disgusted', 'Fearful', 'Happy',
            'Neutral', 'Sad', 'Surprised'
        ]
        
        # Color mapping for emotions
        self.emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgusted': (0, 165, 255), # Orange
            'Fearful': (75, 0, 130),   # Indigo
            'Happy': (0, 255, 0),      # Green
            'Neutral': (128, 128, 128), # Gray
            'Sad': (255, 0, 0),        # Blue
            'Surprised': (255, 255, 0) # Cyan
        }
    
    def load_emotion_model(self):
        """
        Load pre-trained emotion detection model
        If model doesn't exist, create a simple one
        """
        try:
            model = load_model(self.emotion_model_path)
            print(f"Loaded emotion model from {self.emotion_model_path}")
            return model
        except:
            print(f"Model not found at {self.emotion_model_path}")
            print("Please download from: https://github.com/isseu/emotion-recognition-neural-networks")
            return None
    
    def detect_emotion(self, face_roi):
        """
        Detect emotion from face region of interest
        """
        if self.emotion_model is None:
            return 'Unknown', 0.0
        
        # Preprocess face
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.astype('float') / 255.0
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)
        
        # Predict emotion
        predictions = self.emotion_model.predict(face_roi, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        emotion = self.emotions[emotion_idx]
        confidence = predictions[0][emotion_idx]
        
        return emotion, confidence
    
    def detect_from_webcam(self):
        """
        Real-time emotion detection from webcam
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        print("Starting emotion detection. Press 'q' to quit...")
        emotion_history = []
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                
                # Detect emotion
                emotion, confidence = self.detect_emotion(face_roi)
                emotion_history.append(emotion)
                
                # Keep only last 10 emotions for smoothing
                if len(emotion_history) > 10:
                    emotion_history.pop(0)
                
                # Get most common emotion for stability
                if emotion_history:
                    most_common_emotion = max(set(emotion_history), 
                                             key=emotion_history.count)
                else:
                    most_common_emotion = emotion
                
                color = self.emotion_colors.get(emotion, (255, 255, 255))
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Display emotion and confidence
                label = f"{emotion}: {confidence:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                
                # Display smoothed emotion
                smooth_label = f"Smoothed: {most_common_emotion}"
                cv2.putText(
                    frame,
                    smooth_label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
            
            # Display emotion statistics
            self._display_statistics(frame, emotion_history)
            
            cv2.imshow('Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _display_statistics(self, frame, emotion_history):
        """
        Display emotion statistics on frame
        """
        if not emotion_history:
            return
        
        # Count emotions
        emotion_counts = {}
        for emotion in self.emotions:
            emotion_counts[emotion] = emotion_history.count(emotion)
        
        # Display top 3 emotions
        y_offset = 30
        cv2.putText(frame, "Emotion Stats:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        sorted_emotions = sorted(emotion_counts.items(), 
                                key=lambda x: x[1], reverse=True)
        
        for i, (emotion, count) in enumerate(sorted_emotions[:3]):
            y_offset += 25
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            text = f"{emotion}: {count}"
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.detect_from_webcam()