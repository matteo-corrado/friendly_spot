# Authors: Thor Lemke, Sally Hyun Hahm, Matteo Corrado
# Last Update: 11/07/2025
# Course: COSC 69.15/169.15 at Dartmouth College in 25F with Professor Alberto Quattrini Li
# Purpose: Facial recognition that is able to identify people that the system has seen before, based upon re-training of a recognition model

import cv2
import os
import numpy as np
import time

# System constants and parameters
IMAGE_DIRECTORY = "dataset"
RETRAIN_TERMINAL_COUNT = 30 # desired as roughly Hz/FPS, but not precisely

MIN_FACE_DIMENSION = 30
MIN_NEIGHBORS = 5
IMAGE_SCALE_FACTOR = 1.1

L1_BINDING_THRESHOLD = 30
MIN_SIMILAR_FRAMES = 16

CONFIDENCE_THRESHOLD = 70

class FaceRecognizer:
    def __init__(self):
        
        # Initialize a classifier to find faces within the scene
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initializes the variables required for recognizing familiar faces
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
            ├── person0/
            │   ├── 1.jpg
            │   └── 2.jpg
            └── person1/
                └── 1.jpg
        """
        if not os.path.isdir(dataset_path):
            print(f"Error: Directory '{dataset_path}' not found")
            return False
        
        print("Training face recognizer...")
        
        # Reset values for training every time retrained
        # Could be made to be more efficient by storing data and labels across re-train sessions, and only append new images
        # Not done, as would require additional structure behind storing information about what images are new
        # For the purposes of our project, able to run at real-time, and model did not run into the re-training bottleneck 
        self.face_data = []
        self.face_labels = []
        self.label_names = {}
        self.current_label = 0
        
        # Loop over all people within the training directory
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            self.label_names[self.current_label] = person_name
            
            # Loop over all available images of the current person
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # Extract faces from images in the training data
                # Could be avoided if all data comes from automated image adding, as those images are already cropped
                # However, for more general capability of being able to manually add other images to training, kept
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=IMAGE_SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBORS,
                    minSize=(MIN_FACE_DIMENSION, MIN_FACE_DIMENSION)
                )
                
                # Append the faces to the data and labels that will be trained on
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
        """
        Looks at the directory (same structure as described in method above), an intializes an array to store 
        information about how many people there are, and how many images each individual has to be trained on.
        
        Necessary in order to label new images.
        
        Assumes the structure above, with no additional directories of images beyond those designed to be used for training
        facial recognition.
        """
        
        imageCounts = []
        
        # If no dataset directory, create one, and obviously no existing data to train on
        if not os.path.isdir(directory):
            os.mkdir(directory)
            return imageCounts
        
        # All entries in dataset should be directories themselves, of people
        for entry in os.listdir(directory):
            
            path = os.path.join(directory, entry)
            # Loop over all the people directories
            if os.path.isdir(path):
                fileCount = 0
                # Count how many files are in each person directory
                for file in os.listdir(path):
                    fileCount += 1
            
                # Assume all people named person_, with _ as a number
                personNum = int(entry[6:])
                
                # Aligns _ number with index in array
                # Loop structure verifies that system will work even if directories are not processed in sequential order
                while len(imageCounts) < (personNum + 1):
                    imageCounts.append(0)
                
                # Store number of images for person_ at index _
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
        
        # Initalization process include capturing the existing number of people and images in the dataset, and training based upond existing data
        retrainCount = 1
        imageCountTracker = self.initialize_facial_data(IMAGE_DIRECTORY)
        hasFaceData = self.train_from_directory(IMAGE_DIRECTORY)
        trackFacesAppearances = {}
                
        while True:                
            
            # Store time so able to update at roughly 30 FPS
            start_time = time.perf_counter()
            
            # Stores the current frame from the webcam
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Converting to gray scale and detecting faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=IMAGE_SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBORS,
                minSize=(MIN_FACE_DIMENSION, MIN_FACE_DIMENSION)
            )
            
            # Loops over all faces
            for (x, y, w, h) in faces:
                
                # If there are faces to recognize in the training data,
                if hasFaceData:
                    
                    # Extract the faces and attempt to predict who they might be
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
                # Otherwise, simply mark as unknown
                else:
                    name = "Unknown"
                    confidence_text = "No confidence to report"
                    color = (0, 0, 255) # Red
                    confidence = 100 # Only assigned to simplify tracking procedure below
                
                # For names that are recognized, store position, confidence, and image in array assigned to name as key in dictionary
                # This includes names of "Unknown"   
                if name not in trackFacesAppearances:
                    trackFacesAppearances[name] = []
                trackFacesAppearances[name].append(((x, y, w, h), confidence, frame[y:y+h, x:x+w]))
                
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
            
            # Show the image, and the bounding boxes on faces with labels on the computer screen
            cv2.imshow('Face Recognition', frame)
            
            end_time = time.perf_counter()
            
            # Using the time to process the frame, create a delay that will allow feed to be processed at roughly 30 FPS
            wait_time = 1.0/RETRAIN_TERMINAL_COUNT
            if (end_time - start_time) < wait_time:
                wait_time = int(1000 * (wait_time - (end_time - start_time)))
            else:
                wait_time = 1
            
            if (cv2.waitKey(wait_time) & 0xFF) == ord('q'): # creates a wait that will roughly update the image at 30 Hz
                break
            
            # At the 30th frame, time to retrain
            if retrainCount == RETRAIN_TERMINAL_COUNT:
                  
                # Loop over all the names for faces found within the last second  
                for nameFound in trackFacesAppearances:
                    
                    frames_used = []
                    
                    # Loop over all the images found for that face
                    for i in range(len(trackFacesAppearances[nameFound])):
                        
                        # Include current image as similar to itself
                        countSimilarFrames = 1
                        
                        # If face was already recognized as a match, can skip to the next frame
                        if i in frames_used:
                            continue
                        
                        # Store the current face and frame num that the face came from
                        currentFace = trackFacesAppearances[nameFound][i]   
                        
                        frames_might_use = [i]
                        currentValues = currentFace[0]
                        
                        # Loop over all images on name to see if enough to create a match
                        # Must go over all images, as threshold distance in or out may vary based on current face parameters
                        for j in range(len(trackFacesAppearances[nameFound])):
                            
                            # Don't compare current face to current face
                            if i == j:
                                continue
                            
                            # Don't use images already used in a match found
                            if j in frames_used:
                                continue
                            
                            comparisonFace = trackFacesAppearances[nameFound][j]
                            comparisonValues = comparisonFace[0]
                            
                            # Calculate L1 distance between two faces, as x, y, w, and h, and compare to threshold to see if exstimated to be same face
                            # Naive approach, would even see improvement based on updating comparison values through consecutive images, allowing for face motion through the one second
                            # However, based on the assumption of the scene we are designing for, there is relatively little motion for people, so not considering motion can work for the purposes of this project
                            distance = abs(currentValues[0] - comparisonValues[0]) + abs(currentValues[1] - comparisonValues[1]) + abs(currentValues[2] - comparisonValues[2]) + abs(currentValues[3] - comparisonValues[3])
                            if distance <= L1_BINDING_THRESHOLD:
                                    countSimilarFrames += 1
                                    frames_might_use.append(j)
                      
                        # If meet threshold of number of faces estimated to be the same, add image to the dataset for recognition
                        if countSimilarFrames >= MIN_SIMILAR_FRAMES:
                        
                            # Add all frames that were matched to set of used frames, so as to not use again for comparisons
                            frames_used.extend(frames_might_use)
                        
                            # Find most confident image of face (smallest confidence value) from set of "matching" images
                            most_index = 0
                            most_confidence = 100
                            for index in frames_might_use:
                                confidence = trackFacesAppearances[nameFound][index][1]
                                if confidence < most_confidence:
                                    most_index = index
                                    most_confidence = confidence
                                
                            cropped_image = trackFacesAppearances[nameFound][most_index][2]
                        
                            # Add the most confident image to directory for appropraite person directory, whether new person or existing person
                            imagePath = ""
                            if nameFound == "Unknown":
                                imagePath = f"{IMAGE_DIRECTORY}/person{len(imageCountTracker)}"
                                os.mkdir(imagePath)
                                imagePath = imagePath + "/0.jpg"
                                imageCountTracker.append(1)
                            else:
                                personCount = int(nameFound[-1])
                                imagePath = f"{IMAGE_DIRECTORY}/person{personCount}/{imageCountTracker[personCount]}.jpg"
                                imageCountTracker[personCount] += 1
                        
                            # Reporte that new image is being saved, and save it to the paths defined above
                            print(imagePath)
                            cv2.imwrite(imagePath, cropped_image)
                
                # Re-train the model, and reset parameters for faces seen every cycle and number of frames before re-training
                hasFaceData = self.train_from_directory(IMAGE_DIRECTORY)
                retrainCount = 0
                trackFacesAppearances = {}
            
            # Update retrain count by one, which effectively counts frames since last re-training
            retrainCount += 1
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    recognizer = FaceRecognizer()
    
    recognizer.recognize_from_webcam(confidence_threshold=CONFIDENCE_THRESHOLD)

if __name__ == "__main__":
    main()