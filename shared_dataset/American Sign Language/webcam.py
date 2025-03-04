import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf

class ASLRecognizer:
    def __init__(self):
        self.log("Initializing ASL Recognizer")
        
        # Model setup
        self.log("Loading TFLite model")
        self.interpreter = tf.lite.Interpreter(model_path="asl_model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # MediaPipe setup
        self.log("Initializing MediaPipe Hands")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.CLASS_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
        self.CONFIDENCE_THRESHOLD = 0.7
        self.last_prediction_time = time.time()
        self.fps = 0
        
        self.log("System ready")

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{timestamp}] [WEBCAM] {message}")

    def process_frame(self, frame):
        try:
            start_time = time.time()
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Hand detection
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                
                # Extract landmarks
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                if len(landmarks) == 63:
                    # Prepare input tensor
                    input_data = np.array(landmarks, dtype=np.float32).reshape(1, 63)
                    
                    # Run inference
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                    inference_start = time.time()
                    self.interpreter.invoke()
                    inference_time = time.time() - inference_start
                    
                    # Get results
                    output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                    probabilities = output_data[0]
                    predicted_idx = np.argmax(probabilities)
                    confidence = probabilities[predicted_idx]
                    
                    # Calculate FPS
                    current_time = time.time()
                    self.fps = 1 / (current_time - self.last_prediction_time)
                    self.last_prediction_time = current_time
                    
                    self.log(f"Inference time: {inference_time*1000:.2f}ms | Confidence: {confidence:.4f}")
                    
                    if confidence >= self.CONFIDENCE_THRESHOLD:
                        return self.CLASS_LABELS[predicted_idx], confidence
                    else:
                        return "Low confidence", 0.0
                else:
                    self.log("Invalid landmark count", "WARNING")
                    return "Invalid hand", 0.0
            else:
                return "No hand", 0.0
            
        except Exception as e:
            self.log(f"Processing error: {str(e)}", "ERROR")
            return "Error", 0.0

    def run(self):
        self.log("Starting webcam capture")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            self.log("Failed to open webcam", "ERROR")
            return
            
        while True:
            ret, frame = cap.read()
            if not ret:
                self.log("Frame capture failed", "ERROR")
                break
                
            # Process frame
            prediction, confidence = self.process_frame(frame)
            
            # Display results
            cv2.putText(frame, f"Pred: {prediction} ({confidence:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('ASL Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.log("User requested exit")
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.log("Application shutdown")

if __name__ == "__main__":
    recognizer = ASLRecognizer()
    recognizer.run()