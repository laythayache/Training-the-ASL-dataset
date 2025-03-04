import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from datetime import datetime

def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def process_image(img_path, hands, LANDMARK_SIZE):
    try:
        log(f"Processing image: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            log(f"Could not read image {img_path}", "WARNING")
            return None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                log(f"Found hand {hand_idx+1} in {img_path}")
                for lm_idx, lm in enumerate(hand_landmarks.landmark):
                    landmarks.extend([lm.x, lm.y, lm.z])
            
            if len(landmarks) == LANDMARK_SIZE:
                log(f"Successfully processed {img_path}")
                return np.array(landmarks, dtype=np.float32)
            else:
                log(f"Invalid landmark count {len(landmarks)} in {img_path}", "WARNING")
                return None
        else:
            log(f"No hands detected in {img_path}", "WARNING")
            return None
    except Exception as e:
        log(f"Error processing {img_path}: {str(e)}", "ERROR")
        return None

def main():
    log("Starting data preparation pipeline")
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Configuration
    DATA_DIR = "shared_dataset/American Sign Language/dataset"
    LANDMARK_SIZE = 63
    LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
    
    X, y = [], []
    total_images = 0
    valid_images = 0
    
    log(f"Processing {len(LABELS)} labels in {DATA_DIR}")
    
    for label_idx, label in enumerate(LABELS):
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            log(f"Missing directory for label {label}", "WARNING")
            continue
            
        log(f"Processing label {label} ({label_idx+1}/{len(LABELS)})")
        label_count = 0
        
        for img_name in os.listdir(label_dir):
            total_images += 1
            img_path = os.path.join(label_dir, img_name)
            landmarks = process_image(img_path, hands, LANDMARK_SIZE)
            
            if landmarks is not None:
                X.append(landmarks)
                y.append(label_idx)
                valid_images += 1
                label_count += 1
        
        log(f"Label {label} completed with {label_count} valid samples")
    
    log(f"Data collection complete. Total images: {total_images}, Valid: {valid_images} ({valid_images/total_images:.2%})")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Filter small classes
    log("Filtering classes with insufficient samples")
    unique, counts = np.unique(y, return_counts=True)
    valid_classes = [cls for cls, cnt in zip(unique, counts) if cnt > 1]
    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]
    
    log(f"Final dataset shape: {X.shape}, Labels: {y.shape}")
    
    # Split dataset
    log("Splitting dataset")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y,
        random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )
    
    # Save datasets
    log("Saving processed data")
    np.save("asl_landmark_X_train.npy", X_train)
    np.save("asl_landmark_y_train.npy", y_train)
    np.save("asl_landmark_X_val.npy", X_val)
    np.save("asl_landmark_y_val.npy", y_val)
    np.save("asl_landmark_X_test.npy", X_test)
    np.save("asl_landmark_y_test.npy", y_test)
    
    log("Data preparation completed successfully")

if __name__ == "__main__":
    main()