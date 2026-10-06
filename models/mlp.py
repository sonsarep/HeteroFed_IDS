"""
models/mlp.py
─────────────
3-layer MLP micro-controller model for ESP32-class devices.

Architecture (Section V-C / Table A1):
  Input (batch, 47) → FC(256) → BN → ReLU → Dropout
                    → FC(128) → BN → ReLU → Dropout
                    → FC(64)  → BN → ReLU → Dropout
                    → FC(num_classes)

Measured: 0.9 ms/flow, ~0.38 MB parameters (float32)
"""
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=47, num_classes=33,
                 hidden_dims=(256, 128, 64), dropout=0.20):
        super().__init__()
        layers, in_d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        self.backbone   = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_d, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def get_embedding(self, x):
        return self.backbone(x)

    @property
    def param_mb(self):
        return sum(p.numel() for p in self.parameters()) * 4 / 1024**2


def build_microcontroller_model(num_features=47, num_classes=33):
    return MLP(input_dim=num_features, num_classes=num_classes)
