"""CNN model for mel-spectrogram based depression classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepressionCNN(nn.Module):
    """2D CNN that learns spatial-temporal patterns from mel spectrograms."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 1)
        self._grad_cam_activations = None
        self._grad_cam_gradients = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.conv3(x)
        if not self.training and x.requires_grad:
            x.register_hook(self._save_gradients)
        self._grad_cam_activations = x
        x = self.pool(F.relu(self.bn3(x)))
        x = self.dropout(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def _save_gradients(self, grad: torch.Tensor) -> None:
        self._grad_cam_gradients = grad

    def predict_proba(self, x: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).item()


class FeatureClassifier(nn.Module):
    """MLP on acoustic features for SHAP-based explanations."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
