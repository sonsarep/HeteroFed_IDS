"""
experiments/cckd.py
────────────────────
Algorithm 3: Cross-Client Knowledge Distillation (CCKD)
(Paper Section IV-C, Equations 3-5)

Equations:
  p̄_c(x) = softmax( mean_k(L_k(x,c)) / T )    Eq. 3
  L_KD    = -Σ_x Σ_c p̄_c(x) · log f_k(x)_c   Eq. 4
  L       = (1-λ)·L_CE + λ·L_KD                Eq. 5

All logit values transmitted in float32.
DP noise N(0, σ²·C_clip²·I) added before upload (Section IV-F).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def compute_client_logits(model: nn.Module, X_probe: torch.Tensor,
                           sigma: float = 0.80,
                           device: str = "cpu") -> torch.Tensor:
    """
    Client-side: forward pass over Xprobe + Gaussian DP noise.
    Algorithm 3, Steps 1-4.

    Args:
        model   : local model (any architecture — CCKD's key feature)
        X_probe : (probe_size, F) float32 tensor
        sigma   : DP noise std dev (C_clip=1.0 so noise = N(0, sigma²·I))
        device  : torch device

    Returns:
        noisy_logits : (probe_size, C) float32 tensor (L2-clipped + DP noise)
    """
    model.eval().to(device)
    with torch.no_grad():
        logits = model(X_probe.to(device))
        # L2 clip (C_clip = 1.0 per Section IV-F)
        norms  = logits.norm(dim=-1, keepdim=True).clamp(min=1.0)
        logits = logits / norms
        # Gaussian DP noise
        noise  = torch.randn_like(logits) * sigma
    return (logits + noise).cpu()


def compute_soft_labels(logits_list: list, temperature: float = 3.0) -> torch.Tensor:
    """
    Fog-side: cluster-averaged temperature-softened soft labels.
    Algorithm 3, Steps 6-7 / Equation 3.

    Args:
        logits_list : list of (probe_size, C) tensors from cluster members
        temperature : KD temperature T

    Returns:
        soft_labels : (probe_size, C) tensor
    """
    mean_logits = torch.stack(logits_list, dim=0).mean(dim=0)  # (P, C)
    return F.softmax(mean_logits / temperature, dim=-1)


def kd_loss(student_logits: torch.Tensor, soft_labels: torch.Tensor,
             temperature: float = 3.0) -> torch.Tensor:
    """L_KD — Equation 4."""
    log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return -(soft_labels * log_probs).sum(dim=-1).mean()


def combined_loss(logits: torch.Tensor, hard_labels: torch.Tensor,
                   soft_labels: torch.Tensor,
                   temperature: float = 3.0, lambda_kd: float = 0.5):
    """L = (1-λ)·L_CE + λ·L_KD — Equation 5."""
    ce    = F.cross_entropy(logits, hard_labels)
    kd    = kd_loss(logits, soft_labels, temperature)
    total = (1 - lambda_kd) * ce + lambda_kd * kd
    return total, ce.item(), kd.item()


def local_kd_epoch(model: nn.Module, train_loader: DataLoader,
                    soft_labels: torch.Tensor, X_probe: torch.Tensor,
                    optimizer: torch.optim.Optimizer,
                    device: str = "cpu",
                    temperature: float = 3.0, lambda_kd: float = 0.5) -> float:
    """
    One local KD training epoch.
    Algorithm 3, Steps 8-15.
    """
    model.train().to(device)
    X_probe     = X_probe.to(device)
    soft_labels = soft_labels.to(device)
    total_loss  = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        B = len(X_batch)
        optimizer.zero_grad()

        # Hard-label CE on local data
        logits_hard = model(X_batch)
        ce = F.cross_entropy(logits_hard, y_batch)

        # KD loss on probe set subsample
        pidx         = torch.randint(0, len(X_probe), (min(B, len(X_probe)),))
        logits_probe = model(X_probe[pidx])
        kd           = kd_loss(logits_probe, soft_labels[pidx], temperature)

        loss = (1 - lambda_kd) * ce + lambda_kd * kd
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(train_loader), 1)


class CCKD:
    """Full CCKD module (Algorithm 3)."""
    def __init__(self, X_probe: torch.Tensor, sigma=0.80,
                 temperature=3.0, lambda_kd=0.50, device="cpu"):
        self.X_probe     = X_probe
        self.sigma       = sigma
        self.temperature = temperature
        self.lambda_kd   = lambda_kd
        self.device      = device

    def collect_logits(self, models_list):
        """Step 1: collect noisy logits from cluster member models."""
        return [compute_client_logits(m, self.X_probe,
                                       self.sigma, self.device)
                for m in models_list]

    def make_soft_labels(self, logits_list):
        """Step 2: cluster-average and soften."""
        return compute_soft_labels(logits_list, self.temperature)

    def distill(self, model, train_loader, soft_labels, optimizer):
        """Step 3: one local KD epoch."""
        return local_kd_epoch(model, train_loader, soft_labels,
                               self.X_probe, optimizer, self.device,
                               self.temperature, self.lambda_kd)
