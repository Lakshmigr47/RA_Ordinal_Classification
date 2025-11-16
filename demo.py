import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from src.model import EfficientNetOrdinal, coral_predict

# ------------------------------
# Configuration
# ------------------------------
MODEL_PATH = "saved_models/efficientnet_ordinal.pth"
NUM_CLASSES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# Image Transform
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ------------------------------
# Inference Function
# ------------------------------
def predict_image(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Load trained model
    model = EfficientNetOrdinal(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        pred_stage = coral_predict(outputs)[0].item()

    return image, pred_stage, probabilities


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RA Staging Demo Script")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the X-ray image")
    args = parser.parse_args()

    img, stage, probs = predict_image(args.image)

    print("\n===== Rheumatoid Arthritis Stage Prediction =====")
    print(f"Input Image: {args.image}")
    print(f"Predicted Stage: {stage}")
    print("\nThreshold Probabilities (sigmoid outputs):")
    for i, p in enumerate(probs):
        print(f"  p(stage > {i}): {p:.4f}")

    # Display image
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.title(f"Predicted Stage: {stage}", fontsize=16)
    plt.axis("off")
    plt.show()
