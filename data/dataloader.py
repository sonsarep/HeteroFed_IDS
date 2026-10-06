"""
data/dataloader.py
──────────────────
Dirichlet non-IID partitioner and FL DataLoader for HeteroFed-IDS.

Key functions:
  dirichlet_partition()   — partition indices into K non-IID subsets
  assign_device_types()   — label each client as gateway or MCU straggler
  get_client_dataloaders()— combined setup returning loaders + device info
  sample_probe_set()      — sample unlabelled Xprobe for CCKD
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class IoTFlowDataset(Dataset):
    """PyTorch Dataset wrapping numpy feature/label arrays."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def dirichlet_partition(y: np.ndarray, num_clients: int,
                         alpha: float, seed: int = 42):
    """
    Partition dataset indices using Dirichlet distribution (Algorithm per paper Sec V-A).

    Args:
        y           : label array
        num_clients : number of FL clients K
        alpha       : Dirichlet concentration (0.1 = strong non-IID)
        seed        : random seed

    Returns:
        client_indices : list of K numpy index arrays
    """
    rng     = np.random.default_rng(seed)
    classes = np.unique(y)
    class_idx = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(class_idx[c])

    client_indices = [[] for _ in range(num_clients)]
    for c in classes:
        idxs       = class_idx[c]
        props      = rng.dirichlet(np.repeat(alpha, num_clients))
        splits     = (np.cumsum(props) * len(idxs)).astype(int)[:-1]
        splits     = np.clip(splits, 0, len(idxs))
        for k, chunk in enumerate(np.split(idxs, splits)):
            client_indices[k].extend(chunk.tolist())

    for k in range(num_clients):
        rng.shuffle(client_indices[k])
        client_indices[k] = np.array(client_indices[k])
    return client_indices


def assign_device_types(num_clients: int, straggler_fraction: float,
                         seed: int = 42):
    """
    Assign gateway (CNN-BiLSTM) or MCU straggler (MLP) to each client.

    Returns:
        device_types : list of str per client
        is_straggler : bool array
        slowdown     : float array (1.0 or 4.0)
    """
    rng          = np.random.default_rng(seed)
    n_strag      = int(num_clients * straggler_fraction)
    strag_ids    = rng.choice(num_clients, n_strag, replace=False)
    is_straggler = np.zeros(num_clients, dtype=bool)
    is_straggler[strag_ids] = True
    device_types = ["microcontroller" if s else "gateway" for s in is_straggler]
    slowdown     = np.where(is_straggler, 4.0, 1.0)
    return device_types, is_straggler, slowdown


def sample_probe_set(X: np.ndarray, probe_size: int = 2048,
                      seed: int = 42) -> torch.Tensor:
    """Sample unlabelled probe set Xprobe for CCKD (Section IV-C)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(probe_size, len(X)), replace=False)
    return torch.tensor(X[idx], dtype=torch.float32)


def get_client_dataloaders(X: np.ndarray, y: np.ndarray,
                            num_clients: int, alpha: float,
                            straggler_fraction: float,
                            batch_size: int = 64, seed: int = 42):
    """
    Full FL data setup: partition + DataLoaders + device assignment.

    Returns:
        client_loaders  : list of DataLoader (one per client, None if empty)
        device_types    : list of str
        is_straggler    : bool array
        slowdown        : float array
        client_indices  : list of index arrays
    """
    client_indices               = dirichlet_partition(y, num_clients, alpha, seed)
    device_types, is_strag, slow = assign_device_types(num_clients, straggler_fraction, seed)
    dataset = IoTFlowDataset(X, y)
    loaders = []
    for k in range(num_clients):
        if len(client_indices[k]) == 0:
            loaders.append(None)
        else:
            sub = Subset(dataset, client_indices[k])
            loaders.append(DataLoader(sub, batch_size=batch_size,
                                      shuffle=True, num_workers=0))
    return loaders, device_types, is_strag, slow, client_indices


def train_test_split_chronological(X, y, train_fraction=0.70):
    """
    Chronological train/test split (70/30 as used in paper Section V-A).
    No shuffling — preserves temporal order to prevent future-data leakage.
    """
    split = int(len(y) * train_fraction)
    return (X[:split], y[:split]), (X[split:], y[split:])
