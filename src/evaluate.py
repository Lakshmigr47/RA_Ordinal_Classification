import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
import seaborn as sns
import pandas as pd

from dataset import RAOrdinalDataset
from model import EfficientNetOrdinal, coral_predict

# ------------------------------
# Config
# ------------------------------
DATA_DIR = "data/RA"
NUM_CLASSES = 5
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "saved_models/efficientnet_ordinal.pth"

# ------------------------------
# Test transforms
# ------------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# Load Test Dataset
# ------------------------------
def load_test_data():
    test_set = RAOrdinalDataset(os.path.join(DATA_DIR, "test"), transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_model():
    test_loader = load_test_data()

    model = EfficientNetOrdinal(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = coral_predict(outputs)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # ------------------------------
    # Metrics
    # ------------------------------
    accuracy = np.mean(all_labels == all_preds)
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    mae = mean_absolute_error(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    print("\n===== Evaluation Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"QWK: {qwk:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")

    # ------------------------------
    # Confusion Matrix Plot
    # ------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[0,1,2,3,4],
                yticklabels=[0,1,2,3,4])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("../results/confusion_matrix.png")
    print("\nConfusion matrix saved to results/confusion_matrix.png")

    # ------------------------------
    # Save Predictions CSV
    # ------------------------------
    df = pd.DataFrame({
        "true_label": all_labels,
        "predicted_label": all_preds
    })
    df.to_csv("../results/predictions.csv", index=False)
    print("Predictions saved to results/predictions.csv")


if __name__ == "__main__":
    evaluate_model()
