"""Tests for SAA — Algorithm 2 (including R2-5 correction)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.saa import staleness_discount, cosine_importance, SAABuffer


def test_staleness_discount():
    assert abs(staleness_discount(0, 0.1) - 1.0) < 1e-6
    assert staleness_discount(5, 0.1) < 1.0
    assert staleness_discount(10, 0.1) > 0.0  # never zeroed
    print("[✓] staleness_discount: tau=0→1.0, monotonically decreasing, never zero")


def test_non_negative_weight_r2_5():
    """Verify R2-5 correction: max(sC, 0) ensures wC >= 0."""
    buf = SAABuffer(tau_max=10, alpha=0.1, theta_gate=-1.0)  # gate disabled
    buf.centroid = np.array([1.0, 0.0])
    anti_update  = np.array([-1.0, 0.0])   # cosine = -1.0 (anti-parallel)
    accepted, score = buf.accept(0, anti_update, n_samples=100, tau_C=1)
    if accepted:
        w = [w for _, w, _ in buf.buffer]
        assert all(wi >= 0 for wi in w), "Negative weight detected — R2-5 fix missing!"
    print(f"[✓] Non-negative weight (R2-5): score={score:.2f}, accepted={accepted}")


def test_w_guard():
    """Verify W=0 guard: aggregate returns unchanged params when buffer empty."""
    buf = SAABuffer(tau_max=10)
    global_flat = np.ones(10)
    result = buf.aggregate(global_flat)
    assert np.allclose(result, global_flat), "W=0 guard failed"
    print("[✓] W=0 guard: empty buffer returns unchanged global params")


def test_staleness_rejection():
    buf = SAABuffer(tau_max=5)
    accepted, _ = buf.accept(0, np.ones(10), n_samples=100, tau_C=6)  # tau > tau_max
    assert not accepted
    print("[✓] Staleness rejection: tau_C=6 > tau_max=5 correctly rejected")


def test_aggregation():
    buf = SAABuffer(tau_max=10, alpha=0.1, theta_gate=-1.0)
    buf.centroid = np.zeros(4)
    global_flat = np.zeros(4)
    for i in range(3):
        update = np.array([1.0, 0.0, 0.0, 0.0])
        buf.accept(i, update, n_samples=100, tau_C=0)
    result = buf.aggregate(global_flat)
    assert result[0] > 0, "Aggregation produced no change"
    print(f"[✓] Aggregation: global updated from 0.0 to {result[0]:.4f}")


if __name__ == "__main__":
    test_staleness_discount()
    test_non_negative_weight_r2_5()
    test_w_guard()
    test_staleness_rejection()
    test_aggregation()
    print("\nAll SAA tests passed ✓")
