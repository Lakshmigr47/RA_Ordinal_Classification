import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import RAOrdinalDataset
from model import EfficientNetOrdinal, coral_loss

# ------------------------------
# Training Configuration
# ------------------------------
DATA_DIR = "data/RA"
NUM_CLASSES = 5
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "saved_models/efficientnet_ordinal.pth"


# ------------------------------
# Data Transforms
# ------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])


# ------------------------------
# Load Datasets
# ------------------------------
def load_data():
    train_set = RAOrdinalDataset(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_set = RAOrdinalDataset(os.path.join(DATA_DIR, "val"), transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# ------------------------------
# Training Loop
# ------------------------------
def train_model():
    train_loader, val_loader = load_data()

    model = EfficientNetOrdinal(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            loss = coral_loss(outputs, labels, NUM_CLASSES)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ------------------------------
        # Validation
        # ------------------------------
        model.eval()
        running_val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = coral_loss(outputs, labels, NUM_CLASSES)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Save training curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("results/training_curves.png")

    print(f"\nTraining complete. Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
