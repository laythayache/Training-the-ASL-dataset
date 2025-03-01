Below is an example of a comprehensive README for your repository:

---

# American Sign Language Landmark Extraction and Dataset Preparation

This repository contains a Python script for extracting hand landmarks from images of American Sign Language (ASL) gestures using MediaPipe. The extracted landmarks are then filtered to remove underrepresented classes and split into training, validation, and test sets. The processed data is saved as NumPy arrays, ready for further machine learning model training and evaluation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Script Details](#script-details)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The goal of this project is to prepare an ASL dataset by:
- **Extracting Hand Landmarks:** Using MediaPipe to detect hand landmarks in static images.
- **Data Filtering:** Removing classes that have fewer than 2 samples to avoid issues during stratified splitting.
- **Dataset Splitting:** Dividing the dataset into training, validation, and test sets while maintaining class balance.
- **Saving Processed Data:** Saving the resulting NumPy arrays for further processing and model training.

## Features

- **MediaPipe Integration:** Utilizes MediaPipe Hands for robust hand landmark extraction.
- **OpenCV Image Processing:** Reads and processes images with OpenCV.
- **Stratified Dataset Splitting:** Ensures that each set has a representative distribution of all classes.
- **Data Persistence:** Saves the processed arrays as `.npy` files.

## Prerequisites

Ensure you have Python 3.6 or later installed. The following Python libraries are required:

- [OpenCV](https://opencv.org/) (`opencv-python`)
- [NumPy](https://numpy.org/)
- [MediaPipe](https://mediapipe.dev/)
- [scikit-learn](https://scikit-learn.org/)

## Installation


1. **Create a Virtual Environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. **Install the Required Packages:**

   ```bash
   pip install opencv-python numpy mediapipe scikit-learn
   ```

## Dataset Structure

The script expects the dataset to be organized as follows:

```
shared_dataset/
└── American Sign Language/
    └── dataset/
        ├── A/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── B/
        │   └── ...
        ├── ...
        ├── Z/
        │   └── ...
        ├── del/
        ├── nothing/
        └── space/
```

Each folder under `dataset/` should contain images corresponding to that ASL gesture.

## Usage

1. **Prepare Your Dataset:**  
   Ensure your dataset is structured correctly as shown above.

2. **Run the Preparation Script:**  
   Execute the script to extract landmarks, filter data, and split the dataset:

   ```bash
   python preparing_data.py
   ```

3. **Output:**  
   After processing, the following files will be created in your working directory:
   - `asl_landmark_X_train.npy`
   - `asl_landmark_y_train.npy`
   - `asl_landmark_X_val.npy`
   - `asl_landmark_y_val.npy`
   - `asl_landmark_X_test.npy`
   - `asl_landmark_y_test.npy`

   These files can be loaded later for training and evaluating your ASL recognition model.

## Script Details

- **Landmark Extraction:**  
  The script uses MediaPipe's Hands solution in static image mode to detect 21 hand landmarks (with x, y, and z coordinates), resulting in a 63-element feature vector per image.

- **Filtering Underrepresented Classes:**  
  Classes with fewer than 2 samples are filtered out to avoid issues with stratified splitting in scikit-learn.

- **Data Splitting:**  
  The filtered dataset is split into training (80%), validation (10%), and test (10%) sets with stratification to maintain a balanced class distribution.

- **Saving Processed Data:**  
  The NumPy arrays are saved to disk for further processing or model training.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **MediaPipe:** For providing a robust framework for hand landmark detection.
- **OpenCV:** For its extensive image processing functionalities.
- **scikit-learn:** For providing easy-to-use functions for data splitting and machine learning utilities.

---

Feel free to customize this README with additional project-specific details or instructions as needed. Enjoy building your ASL recognition system!