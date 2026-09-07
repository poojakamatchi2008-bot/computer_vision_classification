import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ==============================
# 1. DATASET PATHS
# ==============================

train_path = "training dataset/rps"
test_path = "dataset/testing dataset/rps-test-set"


# ==============================
# 2. LOAD IMAGES
# ==============================

IMG_SIZE = (64, 64)

classes = ["paper", "rock", "scissors"]

X_train = []
y_train = []

X_test = []
y_test = []


def load_dataset(folder_path, X, y):

    for label, class_name in enumerate(classes):

        class_path = os.path.join(folder_path, class_name)

        for file_name in os.listdir(class_path):

            file_path = os.path.join(class_path, file_name)

            image = cv2.imread(file_path)

            if image is None:
                continue

            image = cv2.resize(image, IMG_SIZE)

            # Convert BGR to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            X.append(gray)
            y.append(label)


load_dataset(train_path, X_train, y_train)
load_dataset(test_path, X_test, y_test)


X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)


print("Training images:", len(X_train))
print("Testing images:", len(X_test))


# ==============================
# 3. VISUALIZE SAMPLE IMAGES
# ==============================

plt.figure(figsize=(10, 4))

for i in range(6):

    plt.subplot(2, 3, i + 1)

    plt.imshow(X_train[i], cmap="gray")

    plt.title(classes[y_train[i]])

    plt.axis("off")

plt.suptitle("Sample Training Images")

plt.show()


# ============================================
# 3. FEATURE EXTRACTION
# ============================================


def extract_features(images):
    features = []

    for image in images:

        # If image is already grayscale
        if len(image.shape) == 2:
            gray = image

        # If image is a color image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize image
        gray = cv2.resize(gray, (64, 64))

        # Normalize pixel values
        gray = gray.astype("float32") / 255.0

        # Convert image into 1D feature vector
        features.append(gray.flatten())

    return np.array(features)

print("\nExtracting features...")

X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

y_train_array = np.array(y_train)
y_test_array = np.array(y_test)

print("Training feature shape:", X_train_features.shape)
print("Testing feature shape:", X_test_features.shape)


# ============================================
# 4. TRAIN SVM CLASSIFIER
# ============================================


print("\nTraining SVM model...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train_features, y_train_array)

print("Model training completed!")


# ==============================
# 6. PREDICTION
# ==============================

y_pred = model.predict(X_test_features)


# ============================================
# 5. MODEL EVALUATION
# ============================================


accuracy = accuracy_score(y_test_array, y_pred)

print("\n================================")
print("MODEL EVALUATION")
print("================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test_array,
        y_pred,
        target_names=classes
    )
)

# Confusion Matrix

cm = confusion_matrix(y_test_array, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# 9. ACTUAL VS PREDICTED
# ==============================

plt.figure(figsize=(12, 7))

for i in range(9):

    plt.subplot(3, 3, i + 1)

    plt.imshow(X_test[i], cmap="gray")

    actual = classes[y_test_array[i]]
    predicted = classes[y_pred[i]]

    plt.title(
        f"Actual: {actual}\nPredicted: {predicted}"
    )

    plt.axis("off")

plt.suptitle("Actual vs Predicted - Rock Paper Scissors", fontsize=16)

plt.tight_layout()
plt.show()


# ==============================
# 10. FINAL RESULT
# ==============================

print("\n==============================")
print("FINAL RESULT")
print("==============================")

print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("Classes:", classes)
# ==============================
# 10. SINGLE IMAGE PREDICTION
# ==============================

image_path = input("\nEnter image path: ")

image = cv2.imread(image_path)

if image is None:
    print("Image not found. Check the image path.")
else:
    # Resize image
    image = cv2.resize(image, IMG_SIZE)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values
    gray = gray.astype("float32") / 255.0

    # Convert image into feature vector
    feature = gray.flatten().reshape(1, -1)

    # Predict class
    prediction = model.predict(feature)

    predicted_class = classes[prediction[0]]

    print("\n==============================")
    print("SINGLE IMAGE PREDICTION")
    print("==============================")

    print("Predicted Class:", predicted_class)

    # Display image
    plt.figure(figsize=(5, 5))

display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(display_image)
plt.title(f"Predicted: {predicted_class}")
plt.axis("off")
plt.show()

    # ==============================
# 11. FINAL RESULT
# ==============================

print("\n==============================")
print("FINAL RESULT")
print("==============================")

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classes:", classes)