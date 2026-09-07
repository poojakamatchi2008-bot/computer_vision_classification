# Rock Paper Scissors - Computer Vision Classification

## Project Overview

This project is a Computer Vision classification system that identifies images as:

- Paper
- Rock
- Scissors

The project uses image preprocessing, feature extraction and an SVM (Support Vector Machine) classifier to classify the images.

## Dataset

The Rock Paper Scissors dataset contains:

- Training images: 2520
- Testing images: 372
- Number of classes: 3

### Classes

1. Paper
2. Rock
3. Scissors

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Methodology

The project follows these steps:

1. Load the training and testing images.
2. Resize the images to 64 × 64 pixels.
3. Convert images to grayscale.
4. Normalize pixel values.
5. Convert images into feature vectors.
6. Train an SVM classifier.
7. Predict the classes of test images.
8. Evaluate the model using accuracy and classification metrics.
9. Visualize the confusion matrix.
10. Perform single-image prediction.

## Machine Learning Model

### Support Vector Machine (SVM)

SVM is used as the classification algorithm.

The model uses:

- Kernel: RBF
- C: 10
- Gamma: Scale

## Model Evaluation

The model achieved an overall accuracy of:

**67.47%**

### Classification Report

| Class | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Paper | 0.55 | 0.69 | 0.61 |
| Rock | 0.95 | 0.60 | 0.73 |
| Scissors | 0.66 | 0.73 | 0.69 |

## Visualizations

The project includes:

- Sample training image visualization
- Confusion matrix
- Actual vs Predicted image visualization

## Single Image Prediction

The trained model can also predict the class of an individual image.

Example:

```text
Input Image
     ↓
Preprocessing
     ↓
Feature Extraction
     ↓
SVM Model
     ↓
Predicted Class

Example Prediction:
   Predicted Class: rock

Project structure:
   Computer_Vision_Classification/
│
├── dataset/
│   └── testing dataset/
│       └── rps-test-set/
│
├── training dataset/
│   └── rps/
│
├── rps_classification.py
│
└── README.md

Result

The Computer Vision classification model successfully classifies Rock, Paper and Scissors images with an accuracy of 67.47%.

Future Improvements

The model performance can be improved using:

Data augmentation
Better feature extraction techniques
CNN-based image classification
Hyperparameter tuning
Image preprocessing improvements
Conclusion

This project demonstrates a complete Computer Vision classification pipeline using OpenCV and SVM. It covers image loading, preprocessing, feature extraction, model training, prediction and evaluation.