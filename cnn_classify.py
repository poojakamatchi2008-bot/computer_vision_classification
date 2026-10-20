# ============================================================
# ROCK PAPER SCISSORS - CNN IMAGE CLASSIFICATION
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 2. SETTINGS
# ============================================================

train_path = "training dataset/rps"
test_path = "dataset/testing dataset/rps-test-set"

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

classes = ["paper", "rock", "scissors"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Classes:", classes)


# ============================================================
# 3. CUSTOM DATASET
# ============================================================

class RPSDataset(Dataset):

    def __init__(self, folder_path):

        self.images = []
        self.labels = []

        for label, class_name in enumerate(classes):

            class_folder = os.path.join(folder_path, class_name)

            if not os.path.exists(class_folder):
                print("Folder not found:", class_folder)
                continue

            for file_name in sorted(os.listdir(class_folder)):

                file_path = os.path.join(class_folder, file_name)

                try:

                    image = Image.open(file_path).convert("L")

                    image = image.resize(
                        (IMG_SIZE, IMG_SIZE)
                    )

                    image = np.array(image, dtype=np.float32)

                    image = image / 255.0

                    image = torch.tensor(
                        image,
                        dtype=torch.float32
                    )

                    # Add channel dimension
                    # 64 x 64 -> 1 x 64 x 64
                    image = image.unsqueeze(0)

                    self.images.append(image)
                    self.labels.append(label)

                except Exception:
                    continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        return self.images[index], self.labels[index]


# ============================================================
# 4. LOAD DATASET
# ============================================================

train_dataset = RPSDataset(train_path)
test_dataset = RPSDataset(test_path)

print()
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))


# ============================================================
# 5. DATA LOADERS
# ============================================================

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 6. DISPLAY SAMPLE TRAINING IMAGES
# ============================================================

print()
print("Displaying sample training images...")

fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for i, ax in enumerate(axes.flat):

    image, label = train_dataset[i]

    image = image.squeeze(0).numpy()

    ax.imshow(image, cmap="gray")

    # Actual label from dataset
    ax.set_title(
        f"Actual: {classes[label]}"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 7. CNN MODEL
# ============================================================

class RPS_CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # First convolution block
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # Second convolution block
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # Third convolution block
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            # 64 -> 32 -> 16 -> 8
            # final feature map = 128 x 8 x 8

            nn.Linear(
                128 * 8 * 8,
                128
            ),

            nn.ReLU(),

            # Helps reduce overfitting
            nn.Dropout(0.5),

            nn.Linear(
                128,
                3
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# 8. CREATE MODEL
# ============================================================

model = RPS_CNN().to(device)

print()
print("==============================")
print("CNN MODEL")
print("==============================")

print(model)


# ============================================================
# 9. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ============================================================
# 10. TRAINING
# ============================================================

print()
print("==============================")
print("STARTING CNN TRAINING")
print("==============================")


train_losses = []
test_losses = []

train_accuracies = []
test_accuracies = []


best_test_accuracy = 0.0
best_model_state = None

# Early stopping counter
patience = 4
counter = 0


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    train_predictions = []
    train_actuals = []

    for images, labels in train_loader:

        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_predictions.extend(
            predictions.cpu().numpy()
        )

        train_actuals.extend(
            labels.cpu().numpy()
        )


    train_loss = (
        running_loss /
        len(train_dataset)
    )

    train_accuracy = accuracy_score(
        train_actuals,
        train_predictions
    )


    # --------------------------------------------------------
    # TESTING
    # --------------------------------------------------------

    model.eval()

    test_running_loss = 0.0

    test_predictions = []
    test_actuals = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            labels = labels.to(device)

            
            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            test_running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            test_predictions.extend(
                predictions.cpu().numpy()
            )

            test_actuals.extend(
                labels.cpu().numpy()
            )


    test_loss = (
        test_running_loss /
        len(test_dataset)
    )

    test_accuracy = accuracy_score(
        test_actuals,
        test_predictions
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.4f} "
        f"Test Loss: {test_loss:.4f} "
        f"Test Accuracy: {test_accuracy:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if test_accuracy > best_test_accuracy:

        best_test_accuracy = test_accuracy

        best_model_state = {
            key: value.cpu().clone()
            for key, value in model.state_dict().items()
        }

        counter = 0

    else:

        counter += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if counter >= patience:

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# 11. RESTORE BEST MODEL
# ============================================================

if best_model_state is not None:

    model.load_state_dict(
        best_model_state
    )

    model = model.to(device)


# ============================================================
# 12. FINAL TEST
# ============================================================

model.eval()

final_predictions = []
final_actuals = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        final_predictions.extend(
            predictions.cpu().numpy()
        )

        final_actuals.extend(
            labels
        )


final_accuracy = accuracy_score(
    final_actuals,
    final_predictions
)


print()
print("==============================")
print("FINAL TEST RESULT")
print("==============================")

print(
    f"Test Accuracy: {final_accuracy * 100:.2f} %"
)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print()
print("==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        final_actuals,
        final_predictions,
        target_names=classes,
        zero_division=0
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    final_actuals,
    final_predictions
)

print()
print("==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)


# ============================================================
# 15. CONFUSION MATRIX VISUALIZATION
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(
    cmap="Blues"
)

plt.title(
    "CNN Confusion Matrix"
)

plt.tight_layout()
plt.show()


# ============================================================
# 16. TRAINING VS TESTING ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    test_accuracies,
    label="Testing Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training vs Testing Accuracy"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 17. TRAINING VS TESTING LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    test_losses,
    label="Testing Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training vs Testing Loss"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# 18. ACTUAL VS PREDICTED IMAGES
# ============================================================

print()
print(
    "Displaying Actual vs Predicted images..."
)

fig, axes = plt.subplots(
    3,
    4,
    figsize=(12, 9)
)

for i, ax in enumerate(axes.flat):

    if i >= len(test_dataset):
        ax.axis("off")
        continue

    image, actual_label = test_dataset[i]

    image_for_model = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(
            image_for_model
        )

        predicted_label = torch.argmax(
            output,
            dim=1
        ).item()

    image_display = (
        image.squeeze(0).numpy()
    )

    ax.imshow(
        image_display,
        cmap="gray"
    )

    ax.set_title(
        f"Actual: {classes[actual_label]}\n"
        f"Predicted: {classes[predicted_label]}"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 19. SAVE TRAINED MODEL
# ============================================================

model_path = "rps_cnn_model.pth"

torch.save(
    model.state_dict(),
    model_path
)

print()
print(
    "Trained model saved as:",
    model_path
)


# ============================================================
# 20. SINGLE IMAGE PREDICTION
# ============================================================

print()
print("==============================")
print("SINGLE IMAGE PREDICTION")
print("==============================")


unknown_path = input(
    "Enter image path: "
).strip()


try:

    unknown_image = Image.open(
        unknown_path
    ).convert("L")

except Exception:

    print()
    print(
        "ERROR: Please enter a valid "
        "image file (.jpg, .jpeg, .png)"
    )

    exit()


# Resize
unknown_image = unknown_image.resize(
    (IMG_SIZE, IMG_SIZE)
)


# Convert to NumPy
unknown_array = np.array(
    unknown_image,
    dtype=np.float32
)


# Normalize
unknown_array = (
    unknown_array / 255.0
)


# Convert to tensor
unknown_tensor = torch.tensor(
    unknown_array,
    dtype=torch.float32
)


# Add channel dimension
unknown_tensor = unknown_tensor.unsqueeze(0)


# Add batch dimension
unknown_tensor = unknown_tensor.unsqueeze(0)


unknown_tensor = unknown_tensor.to(
    device
)


# ============================================================
# PREDICTION
# ============================================================

model.eval()

with torch.no_grad():

    output = model(
        unknown_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = (
        probabilities[0][predicted_class]
        .item() * 100
    )


print()
print("==============================")
print(
    "Predicted Class:",
    classes[predicted_class]
)

print(
    f"Confidence: {confidence:.2f} %"
)

print("==============================")

# Display the input image with prediction
plt.figure(figsize=(5, 5))

plt.imshow(unknown_image, cmap="gray")

plt.title(
    f"Predicted: {classes[predicted_class]}\n"
    f"Confidence: {confidence:.2f}%"
)

plt.axis("off")

plt.show()