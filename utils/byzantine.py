"""
utils/byzantine.py
───────────────────
Byzantine attack simulation (Paper Section VI-D, R1-5/R2-7 extended evaluation).

Five attack types evaluated in the paper:
  1. label_flipping          — random label permutation (original)
  2. sign_flipping           — all gradient signs reversed
  3. gaussian_noise          — random gradient direction N(0,I)
  4. model_replacement       — gradient scaled ×10 to dominate aggregate
  5. adaptive_centroid_mimic — mimics cluster centroid then embeds backdoor
"""
import numpy as np
import torch


def label_flip(y: np.ndarray, num_classes: int, seed=42) -> np.ndarray:
    """Replace labels with random incorrect classes."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    for i in range(len(out)):
        choices = [c for c in range(num_classes) if c != out[i]]
        out[i]  = rng.choice(choices)
    return out


def sign_flip(gradient: np.ndarray) -> np.ndarray:
    """Reverse all gradient signs."""
    return -gradient


def gaussian_noise_attack(gradient: np.ndarray, seed=42) -> np.ndarray:
    """Replace gradient with random Gaussian noise of same norm."""
    rng  = np.random.default_rng(seed)
    norm = np.linalg.norm(gradient)
    rand = rng.standard_normal(gradient.shape)
    return rand / (np.linalg.norm(rand) + 1e-9) * norm


def model_replacement(gradient: np.ndarray, scale=10.0) -> np.ndarray:
    """Scale gradient to dominate the aggregate."""
    return gradient * scale


def adaptive_centroid_mimic(gradient: np.ndarray,
                              centroid: np.ndarray,
                              target_class_idx: int,
                              embed_strength=0.1) -> np.ndarray:
    """
    Craft update that (a) lies close to centroid direction and
    (b) embeds a small backdoor perturbation toward target_class_idx.

    Strategy: project gradient onto centroid direction (passes cosine gate),
    then add a small perturbation orthogonal to centroid targeting rare class.
    """
    c_norm   = centroid / (np.linalg.norm(centroid) + 1e-9)
    proj     = np.dot(gradient, c_norm) * c_norm          # component ‖ centroid
    orth     = gradient - proj                              # orthogonal component
    backdoor = orth * embed_strength                        # small orthogonal perturbation
    return proj + backdoor


def assign_byzantine(num_clients: int, fraction: float, seed=42):
    """
    Randomly designate Byzantine clients.
    Returns (byzantine_ids, is_byzantine_array).
    """
    rng   = np.random.default_rng(seed)
    n_byz = int(num_clients * fraction)
    ids   = rng.choice(num_clients, n_byz, replace=False)
    mask  = np.zeros(num_clients, dtype=bool)
    mask[ids] = True
    return set(ids.tolist()), mask


def cosine_gate(update: np.ndarray, centroid: np.ndarray,
                 threshold=0.30):
    """
    First-stage Byzantine filter: cosine-similarity gate.
    Returns (accepted: bool, score: float).
    """
    u = np.linalg.norm(update); c = np.linalg.norm(centroid)
    if u < 1e-9 or c < 1e-9:
        return False, 0.0
    score = float(np.dot(update, centroid) / (u * c))
    return score >= threshold, score


def trimmed_mean(updates: list, trim=0.10) -> np.ndarray:
    """
    Second-stage Byzantine filter: coordinate-wise trimmed mean.
    Removes top and bottom `trim` fraction per coordinate.
    """
    arr = np.stack(updates, axis=0)   # (K, D)
    K   = arr.shape[0]
    n   = max(1, int(K * trim))
    s   = np.sort(arr, axis=0)
    return s[n: K-n].mean(axis=0)
