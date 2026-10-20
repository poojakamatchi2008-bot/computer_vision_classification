
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Rock Paper Scissors CNN",
    page_icon="✋",
    layout="wide"
)


# =========================================================
# 2. TITLE
# =========================================================

st.title("✋ Rock Paper Scissors - CNN Image Classification")

st.write(
    "Image classification using a "
    "Convolutional Neural Network (CNN)"
)


# =========================================================
# 3. CLASSES
# =========================================================

classes = ["paper", "rock", "scissors"]


# =========================================================
# 4. CNN MODEL
# =========================================================

class RPS_CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================================================
# 5. LOAD TRAINED MODEL
# =========================================================

device = torch.device("cpu")


@st.cache_resource
def load_model():

    model = RPS_CNN()

    model.load_state_dict(
        torch.load(
            "rps_cnn_model.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()
    return model


try:
    model = load_model()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# =========================================================
# 6. SIDEBAR
# =========================================================

st.sidebar.title("Project Information")

st.sidebar.write("### Model")
st.sidebar.write("Convolutional Neural Network")

st.sidebar.write("### Classes")
st.sidebar.write("Paper")
st.sidebar.write("Rock")
st.sidebar.write("Scissors")

st.sidebar.write("### Dataset")
st.sidebar.write("Training Images: 2520")
st.sidebar.write("Testing Images: 372")

st.sidebar.write("### Test Accuracy")
st.sidebar.success("86.56%")


# =========================================================
# 7. MODEL PERFORMANCE
# =========================================================

st.header("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Training Images", "2520")

with col2:
    st.metric("Testing Images", "372")

with col3:
    st.metric("Test Accuracy", "86.56%")


# =========================================================
# 8. CONFUSION MATRIX
# =========================================================

st.header("🔲 Confusion Matrix")

cm = np.array([
    [121, 0, 3],
    [21, 88, 15],
    [11, 0, 113]
])

fig, ax = plt.subplots()

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

ax.set_title("CNN Confusion Matrix")

st.pyplot(fig)
plt.close(fig)


# =========================================================
# 9. CLASSIFICATION REPORT
# =========================================================

st.header("📋 Classification Report")

report_data = {
    "Class": ["Paper", "Rock", "Scissors"],
    "Precision": [0.79, 1.00, 0.86],
    "Recall": [0.98, 0.71, 0.91],
    "F1 Score": [0.87, 0.83, 0.89]
}

st.table(report_data)


# =========================================================
# 10. IMAGE UPLOAD
# =========================================================

st.header("🖼️ Image Prediction")

st.write(
    "Upload a Rock, Paper, or Scissors image to classify it."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# 11. PREDICTION
# =========================================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(image, width=300)

        if st.button("🔍 Predict"):

            processed_image = image.convert("L")
            processed_image = processed_image.resize((64, 64))

            image_array = np.array(
                processed_image,
                dtype=np.float32
            ) / 255.0

            image_tensor = torch.tensor(
                image_array,
                dtype=torch.float32
            )

            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():

                outputs = model(image_tensor)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                confidence, predicted = torch.max(
                    probabilities,
                    dim=1
                )

            predicted_class = classes[predicted.item()]
            confidence_value = confidence.item() * 100

            st.success(
                f"Predicted Class: {predicted_class.upper()}"
            )

            st.info(
                f"Confidence: {confidence_value:.2f}%"
            )

            st.subheader("Class Probabilities")

            for i, class_name in enumerate(classes):

                probability = probabilities[0][i].item()

                st.write(
                    f"{class_name.title()}: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(probability)

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.write(
    "CNN-based Rock Paper Scissors Image Classification"
)
