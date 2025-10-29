import cv2
import os
import numpy as np
from pathlib import Path

IMAGE_DIRECTORY = "dataset"
RETRAIN_TERMINAL_COUNT = 30 # desired as roughly Hz, but not precisely

MIN_FACE_DIMENSION = 30
MIN_NEIGHBORS = 5
IMAGE_SCALE_FACTOR = 1.1

class FaceRecognizer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_data = []
        self.face_labels = []
        self.label_names = {}
        self.current_label = 0
    
    def train_from_directory(self, dataset_path):
        """
        Train the recognizer from a directory of labeled face images
        Directory structure:
        dataset/
            ├── person1/
            │   ├── image1.jpg
            │   └── image2.jpg
            └── person2/
                └── image1.jpg
        """
        if not os.path.isdir(dataset_path):
            print(f"Error: Directory '{dataset_path}' not found")
            return False
        
        print("Training face recognizer...")
        
        # Reset values for training every time retrained
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_data = []
        self.face_labels = []
        self.label_names = {}
        self.current_label = 0
        
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            self.label_names[self.current_label] = person_name
            
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=IMAGE_SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBORS,
                    minSize=(MIN_FACE_DIMENSION, MIN_FACE_DIMENSION)
                )
                
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y + h, x:x + w]
                    self.face_data.append(face_roi)
                    self.face_labels.append(self.current_label)
            
            self.current_label += 1
        
        if len(self.face_data) == 0:
            print("Error: No faces found in dataset")
            return False
        
        # Train the recognizer
        self.face_recognizer.train(self.face_data, np.array(self.face_labels))
        print(f"Training complete! Recognized {len(self.label_names)} people")
        return True
    
    def initialize_facial_data(self, directory):
        
        imageCounts = []
        for entry in os.listdir(directory):
            
            path = os.path.join(directory, entry)
            if os.path.isdir(path):
                fileCount = 0
                for file in os.listdir(path):
                    fileCount += 1
            
                personNum = int(entry[-1])
                while len(imageCounts) < (personNum + 1):
                    imageCounts.append(0)
                imageCounts[personNum] = fileCount
            
        return imageCounts
    
    def recognize_from_webcam(self, confidence_threshold=70):
        """
        Real-time face recognition from webcam
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        print("Starting face recognition. Press 'q' to quit...")
        
        retrainCount = RETRAIN_TERMINAL_COUNT
        imageCountTracker = self.initialize_facial_data(IMAGE_DIRECTORY)
                
        while True:
            
            if retrainCount==RETRAIN_TERMINAL_COUNT:
                hasFaceData = self.train_from_directory(IMAGE_DIRECTORY)
                retrainCount = 0
            
            ret, frame = cap.read()
            
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=IMAGE_SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBORS,
                minSize=(MIN_FACE_DIMENSION, MIN_FACE_DIMENSION)
            )
            
            for (x, y, w, h) in faces:
                
                if hasFaceData:
                    face_roi = gray[y:y + h, x:x + w]
                    label, confidence = self.face_recognizer.predict(face_roi)
                
                    # Determine if recognized or unknown
                    if confidence < confidence_threshold:
                        name = self.label_names.get(label, "Unknown")
                        confidence_text = f"{confidence:.2f}"
                        color = (0, 255, 0)  # Green
                    else:
                        name = "Unknown"
                        confidence_text = f"{confidence:.2f}"
                        color = (0, 0, 255)  # Red
                else:
                    name = "Unknown"
                    confidence_text = "No confidence to report"
                    color = (0, 0, 255) # Red
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"{name} ({confidence_text})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )
                
                if retrainCount == RETRAIN_TERMINAL_COUNT - 1:
                    if ret:
                        cropped_image = frame[y:y+h, x:x+w]
                        
                        imagePath = ""
                        if name == "Unknown":
                            imagePath = f"{IMAGE_DIRECTORY}/person{len(imageCountTracker)}"
                            os.mkdir(imagePath)
                            imagePath = imagePath + "/0.jpg"
                            imageCountTracker.append(1)
                        else:
                            personCount = int(name[-1])
                            imagePath = f"{IMAGE_DIRECTORY}/person{personCount}/{imageCountTracker[personCount]}.jpg"
                            imageCountTracker[personCount] += 1
                        
                        print(imagePath)
                        cv2.imwrite(imagePath, cropped_image)
                    
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(int(1000/RETRAIN_TERMINAL_COUNT)) & 0xFF == ord('q'): # creates a wait that will roughly update the image at 30 Hz
                break
            
            retrainCount += 1
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    recognizer = FaceRecognizer()
    
    recognizer.recognize_from_webcam(confidence_threshold=70)

if __name__ == "__main__":
    main()