import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf

# Load TFLite model using TensorFlow
interpreter = tf.lite.Interpreter(model_path="asl_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

CLASS_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
CONFIDENCE_THRESHOLD = 0.7

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    prediction_text = "No hand detected"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        if len(landmarks) == 63:
            # Prepare input tensor
            input_data = np.array(landmarks, dtype=np.float32).reshape(1, 63)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Process results
            probabilities = output_data[0]
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            
            if confidence >= CONFIDENCE_THRESHOLD:
                prediction_text = f"{CLASS_LABELS[predicted_idx]} ({confidence:.2f})"
            else:
                prediction_text = "Low confidence"

    # Display results
    cv2.putText(frame, f"Pred: {prediction_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('ASL Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()