import torch.nn as nn


class MLPAHead(nn.Module):
    """Parameter-matched MLP (~12 trainable params).

    Architecture: Linear(feat_dim -> hidden_dim, no bias) -> Tanh -> Linear(hidden_dim -> C)
    With hidden_dim=4, num_classes=2: 4*4 + 4*2+2 = 16+10 = 26 params from head alone,
    but the projection is feature_dim*hidden_dim (no bias). Total trainable = proj + fc.
    """

    def __init__(self, feature_dim, num_classes, hidden_dim=4):
        super().__init__()
        self.proj = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.act = nn.Tanh()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        return self.fc(x)
