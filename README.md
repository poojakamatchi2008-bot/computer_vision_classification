# Rock Paper Scissors Image Classification using CNN

## Project Overview
This project uses a Convolutional Neural Network (CNN) built with PyTorch to classify hand gesture images into three categories: **Rock, Paper, and Scissors**. A Streamlit web application allows users to upload an image and get a prediction.

## Features
- Image classification using a CNN model.
- Image preprocessing using Pillow (PIL) and NumPy.
- Grayscale conversion and resizing to 64 × 64 pixels.
- Model training and evaluation.
- Image upload and prediction through a Streamlit interface.

## Technologies Used
- **Python** – Programming language
- **PyTorch** – CNN model development and prediction
- **Pillow (PIL)** – Image processing
- **NumPy** – Array operations and numerical processing
- **Matplotlib** – Visualization
- **Scikit-learn** – Model evaluation metrics
- **Streamlit** – Web application interface

## Project Structure
```text
Computer_Vision_Classification/
├── app.py
├── cnn_classify.py
├── rps_cnn_model.pth
├── README.md
└── .gitignore
```

## How to Run the Project

### 1. Install the required libraries
```bash
pip install torch streamlit pillow numpy matplotlib scikit-learn
```

### 2. Run the Streamlit application
```bash
streamlit run app.py
```

### 3. Make a Prediction
- Open the local URL displayed in the terminal.
- Upload a Rock, Paper, or Scissors image.
- View the predicted class.

## Model
The project uses a CNN trained to classify images into three classes:
- Rock
- Paper
- Scissors

The trained model is saved as `rps_cnn_model.pth`.

## Learning Outcomes
- Understanding CNN-based image classification.
- Learning image preprocessing techniques.
- Training and evaluating a deep learning model.
- Building an interactive application using Streamlit.

## Author
**Pooja P**
