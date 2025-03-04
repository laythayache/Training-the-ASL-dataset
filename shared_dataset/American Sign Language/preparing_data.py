import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split

def process_image(img_path, hands, LANDMARK_SIZE):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) == LANDMARK_SIZE:
                return np.array(landmarks, dtype=np.float32)
            else:
                return None
        else:
            return None
    except Exception as e:
        return None

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    DATA_DIR = "shared_dataset/American Sign Language/dataset"
    LANDMARK_SIZE = 63
    LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

    X, y = [], []
    for label_idx, label in enumerate(LABELS):
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            landmarks = process_image(img_path, hands, LANDMARK_SIZE)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    # Filter classes with <2 samples
    valid_classes = [cls for cls in np.unique(y) if np.sum(y == cls) > 1]
    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Save datasets
    np.save("asl_landmark_X_train.npy", X_train)
    np.save("asl_landmark_y_train.npy", y_train)
    np.save("asl_landmark_X_val.npy", X_val)
    np.save("asl_landmark_y_val.npy", y_val)
    np.save("asl_landmark_X_test.npy", X_test)
    np.save("asl_landmark_y_test.npy", y_test)

if __name__ == "__main__":
    main()