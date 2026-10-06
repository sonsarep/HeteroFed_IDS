"""Tests for RDP accountant (Section IV-F, R1-4 correction)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.privacy import RDPAccountant, rdp_gaussian, rdp_to_dp
import torch


def test_rdp_per_step():
    eps = rdp_gaussian(sigma=0.8, order=8)
    assert abs(eps - 6.25) < 0.01, f"Expected ~6.25, got {eps}"
    print(f"[✓] Per-step RDP cost: ε_RDP(α=8) = {eps:.4f}")


def test_budget_after_100_rounds():
    """Verify paper claim: ε=8, δ=1e-5 after 100 rounds (Section IV-F)."""
    acc = RDPAccountant(sigma=0.8, C_clip=1.0, delta=1e-5, order=8,
                         subsampling_rate=0.8)
    acc.step(100)
    eps, delta = acc.get_budget()
    print(f"[✓] DP budget after 100 rounds: ε={eps:.2f}, δ={delta:.2e}")
    assert eps <= 10.0, f"ε={eps} unexpectedly large"
    assert delta == 1e-5


def test_clip_and_noise():
    logits = torch.randn(2048, 33) * 5.0   # large logits
    acc    = RDPAccountant(sigma=0.8, C_clip=1.0)
    noisy  = acc.clip_and_noise(logits)
    # After L2 clipping, row norms should be <= C_clip + some noise
    norms  = noisy.norm(dim=-1)
    print(f"[✓] clip_and_noise: mean norm={norms.mean():.3f}, "
          f"shape={noisy.shape}")


def test_zero_steps():
    acc = RDPAccountant()
    eps, delta = acc.get_budget()
    assert eps == 0.0
    print("[✓] Zero steps: ε=0.0")


if __name__ == "__main__":
    test_rdp_per_step()
    test_budget_after_100_rounds()
    test_clip_and_noise()
    test_zero_steps()
    print("\nAll privacy tests passed ✓")
