import os
import cv2
import numpy as np
import mediapipe as mp
import torch
from sklearn.model_selection import train_test_split

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Configuration
DATA_DIR = r"C:\Users\Layth\Desktop\layth stuff\shared_dataset\American Sign Language\dataset"
IMG_SIZE = (224, 224)
LANDMARK_SIZE = 63  # 21 landmarks * 3 (x, y, visibility)
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

def process_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Move image to GPU if possible
        if device.type == "cuda":
            img_gpu = cv2.cuda_GpuMat()
            img_gpu.upload(img)
            img = img_gpu.download()
        
        results = hands.process(img)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.visibility])
            return np.array(landmarks, dtype=np.float32)
        return None
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

# Collect data with validation
X, y = [], []
valid_samples = 0
total_samples = 0

for label_idx, label in enumerate(LABELS):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        print(f"Warning: Missing directory for label {label}")
        continue
        
    print(f"\nProcessing {label}...")
    label_count = 0
    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)
        total_samples += 1
        landmarks = process_image(img_path)
        
        if landmarks is not None and len(landmarks) == LANDMARK_SIZE:
            X.append(landmarks)
            y.append(label_idx)
            valid_samples += 1
            label_count += 1
            
    print(f"Found {label_count} valid samples for {label}")

# Validate dataset before splitting
if len(X) == 0:
    print("\nCRITICAL ERROR: No valid samples found!")
    print("Possible reasons:")
    print("1. Incorrect dataset directory structure")
    print("2. Images don't contain visible hands")
    print("3. MediaPipe installation issues")
    print(f"Total images checked: {total_samples}")
    exit(1)

print(f"\nDataset summary:")
print(f"Total images processed: {total_samples}")
print(f"Valid samples with hands: {valid_samples} ({valid_samples/total_samples:.1%})")

# Convert to numpy arrays and move to GPU if available
X = np.array(X)
y = np.array(y)

if device.type == "cuda":
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Save datasets (move to CPU before saving if necessary)
np.save("asl_landmark_X_train.npy", X_train.cpu().numpy() if device.type == "cuda" else X_train)
np.save("asl_landmark_y_train.npy", y_train.cpu().numpy() if device.type == "cuda" else y_train)
np.save("asl_landmark_X_val.npy", X_val.cpu().numpy() if device.type == "cuda" else X_val)
np.save("asl_landmark_y_val.npy", y_val.cpu().numpy() if device.type == "cuda" else y_val)
np.save("asl_landmark_X_test.npy", X_test.cpu().numpy() if device.type == "cuda" else X_test)
np.save("asl_landmark_y_test.npy", y_test.cpu().numpy() if device.type == "cuda" else y_test)

print("\nDataset prepared successfully with GPU acceleration!")
