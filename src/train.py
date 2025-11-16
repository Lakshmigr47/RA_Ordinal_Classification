import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RAOrdinalDataset
from model import EfficientNetOrdinal

# ======================
# CONFIG
# ======================
DATA_DIR = "data/RA"
MODEL_SAVE_PATH = "saved_models/efficientnet_ordinal.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 25   # You can reduce to 12–15 if overfitting continues
PATIENCE = 5  # Early stopping patience


# ======================
# LOADERS
# ======================
def get_loaders():
    train_set = RAOrdinalDataset(os.path.join(DATA_DIR, "train"))
    val_set = RAOrdinalDataset(os.path.join(DATA_DIR, "val"))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# ======================
# TRAINING
# ======================
def train_model():

    train_loader, val_loader = get_loaders()

    model = EfficientNetOrdinal(num_classes=4).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Training on device: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        # -------- TRAIN LOOP --------
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)

        # -------- VALIDATION LOOP --------
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

        val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # -------- CHECKPOINT: SAVE BEST MODEL --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🔥 Saved BEST model (val_loss={val_loss:.4f})")

        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

            # -------- EARLY STOPPING --------
            if patience_counter >= PATIENCE:
                print("⛔ Early stopping triggered.")
                break

    print("Training complete.")


if __name__ == "__main__":
    train_model()
