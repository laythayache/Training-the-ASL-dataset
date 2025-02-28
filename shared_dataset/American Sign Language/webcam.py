import cv2
import mediapipe as mp
import torch
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Load trained model
class LandmarkClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(63, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 29)
        )
    
    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LandmarkClassifier().to(device)
model.load_state_dict(torch.load("asl_landmark_model.pth", map_location=device))
model.eval()

# Class labels mapping based on your sequence
CLASS_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# Initialize webcam
cap = cv2.VideoCapture(0)

# FPS variables
prev_time = 0
curr_time = 0

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.7

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB and process with MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    # Reset prediction
    prediction_text = "No hand detected"
    
    if results.multi_hand_landmarks:
        # Extract landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        # Convert to tensor and predict
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(landmarks_tensor.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get prediction text with confidence threshold
        if confidence.item() >= CONFIDENCE_THRESHOLD:
            class_label = CLASS_LABELS[predicted_idx.item()]
            # Handle special commands
            if class_label == 'space':
                prediction_text = "SPACE"
            elif class_label == 'del':
                prediction_text = "DELETE"
            elif class_label == 'nothing':
                prediction_text = "NO SIGN"
            else:
                prediction_text = class_label
            prediction_text += f" ({confidence.item():.2f})"
        else:
            prediction_text = "Low confidence prediction"
        
        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # Display prediction and FPS
    cv2.putText(frame, f"Prediction: {prediction_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('ASL Recognition', frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()