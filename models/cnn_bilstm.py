"""
models/cnn_bilstm.py
────────────────────
CNN-BiLSTM gateway model for Raspberry Pi 5 class devices.

Architecture (Section V-C / Table A1):
  Input (batch, 47)
  → unsqueeze → Conv1d(1→64, k=3) → BN → ReLU
  → Conv1d(64→128, k=3) → BN → ReLU
  → BiLSTM(128, 2 layers, bidirectional)
  → take last timestep → FC(256) → Dropout → FC(num_classes)

Measured: 1.6 ms/flow on Raspberry Pi 5 (BCM2712 @ 2.4 GHz)
          ~4.81 MB parameters (float32)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBiLSTM(nn.Module):
    def __init__(self, input_dim=47, num_classes=33,
                 conv_channels=(64, 128), lstm_hidden=128,
                 lstm_layers=2, dropout=0.30):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[0]), nn.ReLU(), nn.Dropout(dropout*0.5))
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[1]), nn.ReLU(), nn.Dropout(dropout*0.5))
        self.bilstm = nn.LSTM(
            input_size=conv_channels[1], hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        """x: (batch, input_dim) → logits: (batch, num_classes)"""
        x = x.unsqueeze(1)               # (B, 1, F)
        x = self.conv2(self.conv1(x))    # (B, 128, F)
        x = x.permute(0, 2, 1)          # (B, F, 128)
        x, _ = self.bilstm(x)            # (B, F, 256)
        x = x[:, -1, :]                  # (B, 256)
        return self.classifier(x)

    def get_embedding(self, x):
        """Penultimate layer — used by DBEC if needed."""
        x = x.unsqueeze(1)
        x = self.conv2(self.conv1(x))
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = x[:, -1, :]
        return F.relu(self.classifier[0](x))

    @property
    def param_mb(self):
        return sum(p.numel() for p in self.parameters()) * 4 / 1024**2


def build_gateway_model(num_features=47, num_classes=33):
    return CNNBiLSTM(input_dim=num_features, num_classes=num_classes)
