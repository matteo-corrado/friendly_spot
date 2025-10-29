import cv2
import os
import numpy as np
from datetime import datetime
from collections import defaultdict

class DynamicFaceRecognizer:
    def __init__(self, db_path='face_database', confidence_threshold=70):
        self.database = DynamicFaceDatabase(db_path)
        self.confidence_threshold = confidence_threshold
        
        # Track unknown faces
        self.unknown_faces = defaultdict(list)
        self.unknown_counter = 0
        self.pending_enrollment = {}
    
    def recognize_from_webcam(self, auto_save_unknown=False, frames_to_capture=5):
        """
        Real-time recognition with option to auto-save unknown faces
        
        Args:
            auto_save_unknown: If True, save unknown faces for later enrollment
            frames_to_capture: Number of frames to capture for unknown faces
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        print("Starting facial recognition...")
        print("Press 'q' to quit")
        print("Press 'e' to enroll selected unknown person")
        print("Press 's' to save current frame")
        
        frame_count = 0
        current_unknown_id = None
        capture_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.database.face_cascade.detectMultiScale(
                gray, 1.1, 5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                person_name, confidence, label = self.database.recognize_face(
                    face_roi, self.confidence_threshold
                )
                
                if person_name != 'Unknown':
                    # Known person
                    color = (0, 255, 0)  # Green
                    label_text = f"{person_name} ({confidence:.2f})"
                else:
                    # Unknown person
                    color = (0, 0, 255)  # Red
                    unknown_id = self.get_or_create_unknown_id(face_roi)
                    label_text = f"Unknown-{unknown_id}"
                    current_unknown_id = unknown_id
                    
                    if auto_save_unknown:
                        capture_frames.append({
                            'frame': frame.copy(),
                            'roi': face_roi.copy(),
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'unknown_id': unknown_id
                        })
                        if len(capture_frames) > frames_to_capture:
                            capture_frames.pop(0)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )
            
            # Display instructions
            cv2.putText(frame, "Press 'e' to enroll, 's' to save, 'q' to quit",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Dynamic Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('e') and current_unknown_id is not None:
                # Enroll selected unknown person
                person_name = input(f"Enter name for Unknown-{current_unknown_id}: ")
                if person_name:
                    self.enroll_unknown_person(person_name, current_unknown_id, 
                                              capture_frames)
                    current_unknown_id = None
                    capture_frames = []
            elif key == ord('s') and capture_frames:
                # Save current frames
                self.save_unknown_frames(capture_frames)
                print(f"Saved {len(capture_frames)} frames")
                capture_frames = []
        
        cap.release()
        cv2.destroyAllWindows()
    
    def get_or_create_unknown_id(self, face_roi):
        """Get or create ID for unknown face"""
        roi_hash = hash(face_roi.tobytes()) % 10000
        return roi_hash
    
    def enroll_unknown_person(self, person_name, unknown_id, capture_frames):
        """
        Enroll a recognized unknown person into the database
        
        Args:
            person_name: Name to enroll as
            unknown_id: ID of the unknown person
            capture_frames: List of captured frames
        """
        if not capture_frames:
            print("No frames to enroll")
            return False
        
        # Extract face ROIs from captured frames
        temp_dir = f'temp_unknown_{unknown_id}'
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_paths = []
        for i, frame_data in enumerate(capture_frames):
            temp_path = os.path.join(temp_dir, f'face_{i}.jpg')
            cv2.imwrite(temp_path, frame_data['frame'])
            temp_paths.append(temp_path)
        
        # Add to database
        success = self.database.add_person(person_name, temp_paths)
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if success:
            print(f"\n✓ Successfully enrolled '{person_name}'")
            print(f"  Total people in database: {len(self.database.metadata)}")
        
        return success
    
    def save_unknown_frames(self, capture_frames):
        """Save unknown frames for later review/enrollment"""
        unknown_dir = 'unknown_faces'
        os.makedirs(unknown_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unknown_subdir = os.path.join(unknown_dir, timestamp)
        os.makedirs(unknown_subdir, exist_ok=True)
        
        for i, frame_data in enumerate(capture_frames):
            filename = os.path.join(unknown_subdir, f'unknown_{i}.jpg')
            cv2.imwrite(filename, frame_data['frame'])
        
        print(f"Saved unknown faces to {unknown_subdir}")