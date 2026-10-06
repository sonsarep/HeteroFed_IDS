"""
experiments/saa.py
──────────────────
Algorithm 2: Staleness-Aware Asynchronous Aggregation (SAA)
(Paper Section IV-D, Equation 2 [REVISED R2-5])

Corrected weight formula:
  w_C = n_C · φ(τ_C) · max(s_C, 0)

where:
  φ(τ) = 1/(1 + α·τ²)           [staleness discount]
  s_C  = cos(u_C, μ_cluster)    [importance score]
  max(·, 0)                      [non-negativity clipping — R2-5 correction]

Guard: if W = Σ w_C < 1e-6, skip aggregation for that round.
"""
import numpy as np
import torch
import copy


def staleness_discount(tau: int, alpha: float = 0.10) -> float:
    """φ(τ) = 1 / (1 + α·τ²)  — Equation 1 / Algorithm 2 line 6."""
    return 1.0 / (1.0 + alpha * tau**2)


def cosine_importance(update_vec: np.ndarray,
                       centroid_vec: np.ndarray) -> float:
    """
    s_C = cos(u_C, μ_cluster)  — Algorithm 2 line 7.
    Clipped to max(s_C, 0) before use in weight computation (R2-5 correction).
    """
    u = np.linalg.norm(update_vec)
    c = np.linalg.norm(centroid_vec)
    if u < 1e-9 or c < 1e-9:
        return 0.0
    return float(np.dot(update_vec, centroid_vec) / (u * c))


def flatten_params(model: torch.nn.Module) -> np.ndarray:
    return np.concatenate([p.data.cpu().numpy().flatten()
                           for p in model.parameters()])


def unflatten_params(model: torch.nn.Module, flat: np.ndarray):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(torch.tensor(flat[offset:offset+n],
                                   dtype=p.dtype).reshape(p.shape))
        offset += n


class SAABuffer:
    """
    Asynchronous update buffer for one cluster.
    Implements Algorithm 2 lines 1-15.
    """
    W_MIN_GUARD = 1e-6   # If W < this, skip aggregation

    def __init__(self, tau_max=10, alpha=0.10, ema_gamma=0.10,
                 theta_gate=0.30):
        self.tau_max   = tau_max
        self.alpha     = alpha
        self.gamma     = ema_gamma
        self.theta_gate= theta_gate
        self.centroid  = None      # EMA cluster centroid in gradient space
        self.buffer    = []        # [(update_vec, weight, client_id)]

    def accept(self, client_id, update_vec, n_samples, tau_C,
               byzantine_ids=None):
        """
        Try to accept an incoming update.
        Returns (accepted: bool, score: float).
        """
        # Staleness check (line 4)
        if tau_C > self.tau_max:
            return False, 0.0
        # Known Byzantine pre-filter
        if byzantine_ids and client_id in byzantine_ids:
            return False, 0.0

        # Importance score (line 7)
        if self.centroid is None:
            self.centroid = update_vec.copy()
            s_C = 1.0
        else:
            s_C = cosine_importance(update_vec, self.centroid)

        # Cosine-similarity Byzantine gate (line 8 in Alg 4)
        if s_C < self.theta_gate:
            return False, s_C

        # ── CORRECTED weight formula (R2-5) ──────────────────────────────
        phi = staleness_discount(tau_C, self.alpha)
        w_C = n_samples * phi * max(s_C, 0.0)   # max(s_C, 0) clipping
        # ─────────────────────────────────────────────────────────────────

        self.buffer.append((update_vec, w_C, client_id))
        return True, s_C

    def aggregate(self, global_flat: np.ndarray) -> np.ndarray:
        """
        Normalise weights and aggregate (lines 12-14).
        Returns new flat parameter vector.
        """
        if not self.buffer:
            return global_flat.copy()

        updates = np.array([u for u, _, _ in self.buffer])
        weights = np.array([w for _, w, _ in self.buffer])
        W = weights.sum()

        # Guard: W too small — skip this round
        if W < self.W_MIN_GUARD:
            self.buffer.clear()
            return global_flat.copy()

        agg_delta = (updates * weights[:, None]).sum(axis=0) / W
        new_flat  = global_flat + agg_delta

        # EMA centroid update (line 14)
        self.centroid = ((1 - self.gamma) * self.centroid
                         + self.gamma * updates.mean(axis=0))
        self.buffer.clear()
        return new_flat


class SAA:
    """Full SAA module (Algorithm 2) for one cluster."""
    def __init__(self, tau_max=10, alpha=0.10, ema_gamma=0.10, theta_gate=0.30):
        self.buf = SAABuffer(tau_max, alpha, ema_gamma, theta_gate)

    def receive_update(self, client_id, model_before, model_after,
                        n_samples, tau_C, byzantine_ids=None):
        delta = flatten_params(model_after) - flatten_params(model_before)
        return self.buf.accept(client_id, delta, n_samples, tau_C, byzantine_ids)

    def aggregate(self, cluster_model):
        flat     = flatten_params(cluster_model)
        new_flat = self.buf.aggregate(flat)
        unflatten_params(cluster_model, new_flat)
        return cluster_model
