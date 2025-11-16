import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ------------------------------
# Ordinal Regression (CORAL)
# ------------------------------

class CoralOrdinalHead(nn.Module):
    """
    Implements CORAL ordinal head:
    For K classes → K-1 sigmoid outputs.
    """
    def __init__(self, in_features, num_classes):
        super(CoralOrdinalHead, self).__init__()
        self.num_classes = num_classes
        
        # For 5 classes → 4 logits
        self.linear = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        logits = self.linear(x)
        prob = torch.sigmoid(logits)
        return prob  # shape → (batch, num_classes-1)


def coral_label_transform(labels, num_classes):
    """
    Convert class labels (0,1,2,3,4) to ordinal binary matrix:
    Example: label=2 → [1,1,0,0]
    """
    batch_size = labels.size(0)
    label_matrix = torch.zeros((batch_size, num_classes - 1)).float()

    for i, label in enumerate(labels):
        label_matrix[i, :label] = 1

    return label_matrix


def coral_loss(preds, labels, num_classes):
    """
    CORAL Loss = Binary cross entropy applied across K-1 ordinal outputs.
    preds: (B, K-1)
    labels: (B,) raw labels (0–K-1)
    """
    device = preds.device
    ordinal_labels = coral_label_transform(labels, num_classes).to(device)
    
    bce = nn.BCELoss()
    return bce(preds, ordinal_labels)


def coral_predict(outputs):
    """
    Convert sigmoid outputs → predicted ordinal class.
    Count number of thresholds passed.
    Example: [0.9, 0.8, 0.3, 0.1] → predicted class = 2
    """
    preds = torch.round(outputs)
    return torch.sum(preds, dim=1).long()


# ------------------------------
# Full Model: EfficientNet-B0 + CORAL head
# ------------------------------

class EfficientNetOrdinal(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNetOrdinal, self).__init__()

        # Load EfficientNet-B0 pretrained on ImageNet
        self.base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Extract number of features from last layer
        in_features = self.base.classifier[1].in_features

        # Remove final classification layer
        self.base.classifier = nn.Identity()

        # Add CORAL ordinal head
        self.ordinal_head = CoralOrdinalHead(in_features, num_classes)

        self.num_classes = num_classes

    def forward(self, x):
        features = self.base(x)
        outputs = self.ordinal_head(features)
        return outputs
