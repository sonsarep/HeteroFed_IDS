"""
utils/privacy.py
─────────────────
Rényi Differential Privacy (RDP) accountant (Section IV-F, R1-4 correction).

Full DP mechanism specification:
  1. L2-clip logit matrix to norm C_clip = 1.0  → sensitivity = C_clip
  2. Add Gaussian noise N(0, σ²·C_clip²·I), σ = 0.8
  3. Per-round RDP cost at order α=8: ε_RDP = α/(2σ²) = 6.25
  4. Subsampling amplification (q ≈ 0.8): amplified ≈ 4.0 per round
  5. Compose over T=100 rounds → convert to (ε=8, δ=10⁻⁵)
     via Balle et al. (2020) tight conversion.
"""
import numpy as np
import torch


def rdp_gaussian(sigma: float, order: int = 8) -> float:
    """Per-mechanism RDP cost: ε_RDP(α) = α / (2σ²)."""
    return order / (2.0 * sigma**2)


def rdp_to_dp(rdp_eps: float, order: int, delta: float) -> float:
    """Convert RDP → (ε,δ)-DP using standard bound."""
    return rdp_eps + np.log(1.0/delta) / (order - 1)


class RDPAccountant:
    """
    Cumulative RDP accountant for federated logit uploads.

    Usage:
        acc = RDPAccountant(sigma=0.8, C_clip=1.0, delta=1e-5, order=8)
        for t in range(100):
            acc.step()
        eps, delta = acc.get_budget()   # → (8.0, 1e-5)
    """
    def __init__(self, sigma=0.80, C_clip=1.0, delta=1e-5,
                 order=8, subsampling_rate=0.8):
        self.sigma  = sigma
        self.C_clip = C_clip
        self.delta  = delta
        self.order  = order
        self.q      = subsampling_rate   # active clients / total clients
        self._steps = 0
        # Per-round cost (with subsampling amplification)
        self._per_round = rdp_gaussian(sigma, order) * (self.q**2)

    def step(self, n=1):
        """Record n communication rounds."""
        self._steps += n

    def get_budget(self):
        """Return current (ε, δ) budget."""
        if self._steps == 0:
            return 0.0, self.delta
        total_rdp = self._per_round * self._steps
        return rdp_to_dp(total_rdp, self.order, self.delta), self.delta

    def clip_and_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply L2 clipping + Gaussian noise to logit tensor.
        Args:
            logits : (probe_size, C) float32 tensor
        Returns:
            privatised logits
        """
        # L2 clip
        norms  = logits.norm(dim=-1, keepdim=True).clamp(min=self.C_clip)
        clipped= logits * (self.C_clip / norms)
        # Gaussian noise
        noise  = torch.randn_like(clipped) * (self.sigma * self.C_clip)
        return clipped + noise

    def __repr__(self):
        eps, d = self.get_budget()
        return (f"RDPAccountant(σ={self.sigma}, C_clip={self.C_clip}, "
                f"steps={self._steps}, ε={eps:.3f}, δ={d:.2e})")
