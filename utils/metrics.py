"""
utils/metrics.py
─────────────────
Evaluation metrics used in the paper (Section V-C).
Includes statistical testing (Wilcoxon signed-rank) per R1-6.
"""
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
import torch
from torch.utils.data import DataLoader, TensorDataset


def evaluate_model(model, X_test, y_test, device="cpu", batch_size=512):
    """
    Evaluate model on test data.

    Returns:
        macro_f1     : float (%)
        accuracy     : float (%)
        per_class_f1 : numpy array (%)
        y_pred       : numpy array
    """
    model.eval().to(device)
    loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.extend(model(X.to(device)).argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    y_pred, y_true = np.array(preds), np.array(labels)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    accuracy  = (y_pred == y_true).mean() * 100
    per_class = f1_score(y_true, y_pred, average=None,
                          labels=np.arange(y_true.max()+1),
                          zero_division=0) * 100
    return macro_f1, accuracy, per_class, y_pred


def minority_recall(y_true, y_pred, minority_indices):
    """Per-class recall (%) for minority attack classes."""
    return {c: (y_pred[y_true==c]==c).mean()*100 if (y_true==c).sum()>0 else 0.0
            for c in minority_indices}


def wilcoxon_test(scores_a, scores_b, alpha=0.05):
    """
    Wilcoxon signed-rank test for paired results (R1-6).
    Returns (statistic, p_value, significant).
    """
    stat, p = wilcoxon(scores_a, scores_b)
    return stat, p, p < alpha


class ConvergenceTracker:
    """Track macro-F1 per round for convergence curves (Figure 7)."""
    def __init__(self, name, threshold=95.0):
        self.name      = name
        self.threshold = threshold
        self.rounds, self.f1s = [], []
        self.threshold_round  = None

    def update(self, rnd, f1):
        self.rounds.append(rnd); self.f1s.append(f1)
        if self.threshold_round is None and f1 >= self.threshold:
            self.threshold_round = rnd

    def to_dict(self):
        return {"method": self.name, "rounds": self.rounds,
                "f1_scores": self.f1s, "threshold_round": self.threshold_round}


class CommunicationTracker:
    """Track per-round and cumulative communication (Table V)."""
    def __init__(self, name, upload_mb, num_clients):
        self.name = name; self.upload_mb = upload_mb
        self.K    = num_clients; self._cum = 0.0
        self.rounds, self.cumulative_gb = [], []

    def update(self, rnd, active=None):
        n = active or self.K
        self._cum += self.upload_mb * n / 1024
        self.rounds.append(rnd); self.cumulative_gb.append(self._cum)
