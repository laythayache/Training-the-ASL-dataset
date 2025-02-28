import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Configuration
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
INPUT_SIZE = 63  # 21 landmarks * 3 values
NUM_CLASSES = 29

# Force GPU usage
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available - required for training")
DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")

# Dataset Class
class LandmarkDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.X = f['X'][:]
            self.y = f['y'][:]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)
        )

# Model Architecture
class LandmarkClassifier(nn.Module):
    def __init__(self):
        super().__init__()
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

# Training Function
def train_model():
    print("Loading datasets...")
    train_dataset = LandmarkDataset("asl_landmark_train.h5")
    val_dataset = LandmarkDataset("asl_landmark_val.h5")
    test_dataset = LandmarkDataset("asl_landmark_test.h5")
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = LandmarkClassifier().to(DEVICE)
    print("Model architecture:")
    print(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            
            if (batch_idx+1) % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Training Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "asl_landmark_model.pth")
            print(f"New best model saved with accuracy {val_acc:.2f}%")

    # Final test
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load("asl_landmark_model.pth"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    print(f"Final Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train_model()