import torch.nn as nn


class LinearHead(nn.Module):
    """Baseline: single linear layer (original paper head)."""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
