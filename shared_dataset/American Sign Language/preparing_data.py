import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split

def process_image(img_path, hands, LANDMARK_SIZE):
    try:
        print(f"Processing image: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return None
        # Optionally resize image if needed:
        # img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    # Use x, y, and z for consistency with inference
                    landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) == LANDMARK_SIZE:
                print(f"Successfully extracted landmarks for image: {img_path}")
                return np.array(landmarks, dtype=np.float32)
            else:
                print(f"Error: Incorrect landmark size in {img_path} (got {len(landmarks)} values)")
                return None
        else:
            print(f"No hand landmarks detected in {img_path}")
            return None
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def main():
    print("Starting landmark extraction from dataset...")

    # Initialize MediaPipe Hands with robust settings
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Configuration
    DATA_DIR = "shared_dataset/American Sign Language/dataset"
    LANDMARK_SIZE = 63  # 21 landmarks * 3 (x, y, z)
    LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

    X, y = [], []
    total_samples = 0
    valid_samples = 0

    # Process each label directory
    for label_idx, label in enumerate(LABELS):
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            print(f"Warning: Missing directory for label '{label}' at {label_dir}")
            continue

        print(f"\nProcessing label: {label} in directory: {label_dir}")
        label_count = 0
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            total_samples += 1
            landmarks = process_image(img_path, hands, LANDMARK_SIZE)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label_idx)
                valid_samples += 1
                label_count += 1

        print(f"Label '{label}': {label_count} valid samples processed.")

    print("\nLandmark extraction completed.")
    print(f"Total images checked: {total_samples}")
    print(f"Valid samples extracted: {valid_samples} ({(valid_samples/total_samples)*100:.2f}%)")

    if valid_samples == 0:
        print("Error: No valid samples were extracted. Exiting.")
        exit(1)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    print("Converted data to numpy arrays.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Check original class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nOriginal class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"Class {cls} (label: {LABELS[cls]}): {cnt} samples")

    # Filter out classes with fewer than 2 samples to avoid stratification issues
    valid_classes = [cls for cls, cnt in zip(unique, counts) if cnt > 1]
    if len(valid_classes) != len(unique):
        print("\nFiltering out classes with fewer than 2 samples...")
    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]

    # Print filtered class distribution
    unique_filt, counts_filt = np.unique(y, return_counts=True)
    print("\nFiltered class distribution:")
    for cls, cnt in zip(unique_filt, counts_filt):
        print(f"Class {cls} (label: {LABELS[cls]}): {cnt} samples")

    # Split dataset into training, validation, and test sets
    print("\nSplitting dataset into train, validation, and test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")

    # Save datasets
    np.save("asl_landmark_X_train.npy", X_train)
    np.save("asl_landmark_y_train.npy", y_train)
    np.save("asl_landmark_X_val.npy", X_val)
    np.save("asl_landmark_y_val.npy", y_val)
    np.save("asl_landmark_X_test.npy", X_test)
    np.save("asl_landmark_y_test.npy", y_test)

    print("Datasets saved successfully as .npy files.")

if __name__ == "__main__":
    main()
