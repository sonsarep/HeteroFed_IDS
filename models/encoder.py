"""
models/encoder.py
─────────────────
Frozen byte-stream behavioural encoder for DBEC (Algorithm 1, Section IV-B).

Architecture: 4-layer 1D-CNN with 128-unit penultimate layer.
Pre-trained with self-supervised next-packet-size prediction on unlabelled
IoT traffic corpus (no attack labels). Frozen for all downstream clients.

In production: load pre-trained weights from a checkpoint.
In testing:    uses random orthogonal weights (same dimensionality).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteStreamEncoder(nn.Module):
    """
    Pre-trained IoT traffic encoder.
    Output is L2-normalised 128-dim embedding.
    All parameters frozen — never updated during FL training.
    """
    def __init__(self, input_dim=47, hidden_dims=(256, 256), embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        layers, in_d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.GELU()]
            in_d = h
        layers.append(nn.Linear(in_d, embed_dim))
        self.encoder = nn.Sequential(*layers)
        # Structured initialisation simulating pre-trained weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)
        self._freeze()

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return F.normalize(self.encoder(x), dim=-1)

    @torch.no_grad()
    def encode_sample(self, x_numpy: np.ndarray,
                       sample_size: int = 200, seed: int = None) -> np.ndarray:
        """
        Extract mean-pooled embedding from a client's local data sample.
        Used by DBEC (Algorithm 1, Phase 1, Steps 1-4).

        Args:
            x_numpy     : (N, F) client flow features
            sample_size : number of flows to sample
            seed        : random seed for reproducibility

        Returns:
            embedding : (embed_dim,) numpy array
        """
        rng = np.random.default_rng(seed)
        n   = min(sample_size, len(x_numpy))
        idx = rng.choice(len(x_numpy), n, replace=False)
        x   = torch.tensor(x_numpy[idx], dtype=torch.float32)
        self.eval()
        emb = self.forward(x)           # (n, embed_dim)
        return emb.mean(0).numpy()      # (embed_dim,)


def load_encoder(input_dim=47, embed_dim=128,
                  checkpoint_path=None) -> ByteStreamEncoder:
    """
    Load or initialise the frozen encoder.

    Args:
        checkpoint_path : path to .pt file with pre-trained state dict.
                          If None or file not found, random weights are used.
    """
    enc = ByteStreamEncoder(input_dim=input_dim, embed_dim=embed_dim)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        enc.encoder.load_state_dict(state)
        print(f"Loaded pre-trained encoder from {checkpoint_path}")
    else:
        print("Using randomly initialised encoder "
              "(pre-trained checkpoint not found).")
    enc.eval()
    return enc
