import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

print("Starting training for landmark-based ASL model...")

# Device configuration: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 5  # for early stopping
NUM_CLASSES = 29
INPUT_SIZE = 63  # 21 landmarks * 3

print("Loading landmark datasets...")
X_train = np.load("asl_landmark_X_train.npy")
y_train = np.load("asl_landmark_y_train.npy")
X_val = np.load("asl_landmark_X_val.npy")
y_val = np.load("asl_landmark_y_val.npy")
X_test = np.load("asl_landmark_X_test.npy")
y_test = np.load("asl_landmark_y_test.npy")
print(f"Data shapes - Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# Define custom dataset
class LandmarkDataset(Dataset):
    def __init__(self, landmarks, labels):
        self.landmarks = landmarks
        self.labels = labels

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        landmark = torch.tensor(self.landmarks[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return landmark, label

train_dataset = LandmarkDataset(X_train, y_train)
val_dataset = LandmarkDataset(X_val, y_val)
test_dataset = LandmarkDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Define the model
class LandmarkClassifier(nn.Module):
    def __init__(self):
        super(LandmarkClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.model(x)

model = LandmarkClassifier().to(device)
print("Model instantiated and moved to device.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float('inf')
best_model_wts = None
epochs_no_improve = 0

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_train = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        total_train += inputs.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}: Loss = {loss.item():.4f}")

    epoch_loss = running_loss / total_train
    epoch_acc = running_corrects / total_train
    print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data).item()
            total_val += inputs.size(0)

    epoch_val_loss = val_loss / total_val
    epoch_val_acc = val_corrects / total_val
    print(f"Validation Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_acc:.4f}")

    # Early stopping check
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_wts = model.state_dict()
        epochs_no_improve = 0
        print("Validation loss improved. Saving model weights.")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

if best_model_wts is not None:
    model.load_state_dict(best_model_wts)
print("Training complete. Evaluating on test set...")

model.eval()
test_loss = 0.0
test_corrects = 0
total_test = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum(preds == labels.data).item()
        total_test += inputs.size(0)

epoch_test_loss = test_loss / total_test
epoch_test_acc = test_corrects / total_test
print(f"Test Loss: {epoch_test_loss:.4f}, Accuracy: {epoch_test_acc:.4f}")

MODEL_SAVE_PATH = "asl_landmark_model.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
