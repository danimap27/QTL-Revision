import torch.nn as nn


class MLPBHead(nn.Module):
    """Standard MLP head (upper-bound reference).

    Default: 128 -> ReLU -> 64 -> ReLU -> C
    """

    def __init__(self, feature_dim, num_classes, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        in_dim = feature_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
