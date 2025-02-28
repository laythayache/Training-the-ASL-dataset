import cv2
import mediapipe as mp
import torch
import numpy as np
import time

print("Starting webcam for real-time ASL prediction...")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define class labels
CLASS_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

# Define the LandmarkClassifier model (should match the training architecture)
class LandmarkClassifier(torch.nn.Module):
    def __init__(self, input_size=63, num_classes=29):
        super(LandmarkClassifier, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Set device and load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Webcam prediction using device: {device}")

model = LandmarkClassifier().to(device)
MODEL_PATH = "asl_landmark_model.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded for webcam prediction.")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)
else:
    print("Webcam successfully opened.")

prev_time = time.time()
CONFIDENCE_THRESHOLD = 0.7

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    prediction_text = "No hand detected"
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        if len(landmarks) == 63:
            landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(landmarks_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            if confidence.item() >= CONFIDENCE_THRESHOLD:
                predicted_label = CLASS_LABELS[predicted_idx.item()]
                prediction_text = f"{predicted_label} ({confidence.item():.2f})"
            else:
                prediction_text = "Low confidence prediction"
        else:
            print("Warning: Extracted landmarks do not have 63 values.")
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"Prediction: {prediction_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("ASL Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting webcam loop.")
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam and windows closed.")
# End of shared_dataset/American%20Sign%20Language/webcam.py    

