#this file is optional and not necessary for the model to work

import os
import torch
from torchvision import transforms
from PIL import Image

# Define dataset path
data_dir = "shared_dataset\American Sign Language\dataset"

# Image settings
IMG_SIZE = (64, 64)  # Expected size for all images
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# Expected labels
expected_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

# Check dataset consistency
consistent_size = True
first_image_size = None

def load_image(image_path):
    try:
        print(f"Loading image: {image_path}")  # Debugging
        img = Image.open(image_path).convert("RGB")  # Ensure 3-channel RGB format
        img_tensor = transform(img)  # Apply transformations
        return img_tensor
    except Exception as e:
        print(f"Warning: Failed to load {image_path} - {e}")
        return None

for label in expected_labels:
    label_path = os.path.join(data_dir, label)
    
    if os.path.isdir(label_path):
        print(f"Processing label: {label} | Path: {label_path}")  # Log label processing
        for img_index, img_name in enumerate(os.listdir(label_path)):
            img_path = os.path.join(label_path, img_name)
            print(f"Attempting to open: {img_path}")  # Debugging print
            img_tensor = load_image(img_path)
            
            if img_tensor is None:
                print(f"Failed to process: {img_path}")
                continue
            
            img_shape = img_tensor.shape  # Get (channels, height, width)
            
            if first_image_size is None:
                first_image_size = img_shape  # Store first image size as reference
                print(f"Reference size set to {first_image_size}")
            elif img_shape != first_image_size:
                consistent_size = False
                print(f"Inconsistent size found in {label}/{img_name}: {img_shape}")
            
            print(f"Successfully processed {label}/{img_name} ({img_index + 1})")  # Log progress

if first_image_size is None:
    print("Error: No images were successfully processed. Check dataset path and image files.")
elif consistent_size:
    print("All images are cleaned and have the same size:", first_image_size)
else:
    print("Dataset contains images with inconsistent sizes.")
