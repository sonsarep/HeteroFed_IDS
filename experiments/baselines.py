"""
experiments/baselines.py
─────────────────────────
Implementations of all 7 baseline FL algorithms (Paper Section V-C).
All use CNN-BiLSTM backbone for fair comparison.

  1. FedAvg    — McMahan et al. (2017) [4]
  2. FedProx   — Li et al. (2020) [5]
  3. SCAFFOLD  — Karimireddy et al. (2020) [6]
  4. FedNova   — Wang et al. (2020) [7]
  5. MOON      — Li et al. (2021) [15]
  6. Ditto     — Li et al. (2021) [16]
  7. FedMADE   — Sun et al. (2024) [14]
"""
import torch, copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam


# ── Shared utilities ──────────────────────────────────────────────────────────

def local_train_one_epoch(model, loader, optimizer, device, extra_loss_fn=None):
    model.train(); total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = F.cross_entropy(logits, y)
        if extra_loss_fn:
            loss = loss + extra_loss_fn(model, X, logits)
        loss.backward(); optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def fed_avg(models, weights):
    """Weighted average of model state dicts."""
    W   = sum(weights)
    avg = {}
    for k in models[0].state_dict():
        avg[k] = sum(models[i].state_dict()[k].float() * (weights[i]/W)
                     for i in range(len(models)))
    return avg


# ── 1. FedAvg ─────────────────────────────────────────────────────────────────
class FedAvgTrainer:
    name = "FedAvg"; upload_mb = 4.82
    def __init__(self, global_model, lr=0.001, device="cpu"):
        self.gm = global_model; self.lr = lr; self.device = device

    def client_update(self, model, loader, epochs):
        model.load_state_dict(copy.deepcopy(self.gm.state_dict()))
        opt = Adam(model.parameters(), lr=self.lr)
        for _ in range(epochs): local_train_one_epoch(model, loader, opt, self.device)
        return model

    def aggregate(self, models, sizes):
        self.gm.load_state_dict(fed_avg(models, sizes))


# ── 2. FedProx (μ=0.01) ──────────────────────────────────────────────────────
class FedProxTrainer:
    name = "FedProx"; upload_mb = 4.82
    def __init__(self, global_model, lr=0.001, mu=0.01, device="cpu"):
        self.gm = global_model; self.lr = lr; self.mu = mu; self.device = device

    def client_update(self, model, loader, epochs):
        model.load_state_dict(copy.deepcopy(self.gm.state_dict()))
        opt = Adam(model.parameters(), lr=self.lr)
        for _ in range(epochs):
            model.train()
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = F.cross_entropy(model(X), y)
                prox = sum((p - gp.detach()).norm()**2
                           for p, gp in zip(model.parameters(),
                                             self.gm.parameters()))
                (loss + (self.mu/2)*prox).backward(); opt.step()
        return model

    def aggregate(self, models, sizes):
        self.gm.load_state_dict(fed_avg(models, sizes))


# ── 3. SCAFFOLD ───────────────────────────────────────────────────────────────
class SCAFFOLDTrainer:
    name = "SCAFFOLD"; upload_mb = 9.64   # double — model + control variate
    def __init__(self, global_model, num_clients, lr=0.001, device="cpu"):
        self.gm  = global_model; self.lr = lr; self.device = device
        self.s_cv = {k: torch.zeros_like(v)
                     for k, v in global_model.named_parameters()}
        self.c_cv = [{k: torch.zeros_like(v)
                      for k, v in global_model.named_parameters()}
                     for _ in range(num_clients)]

    def client_update(self, cid, model, loader, epochs):
        model.load_state_dict(copy.deepcopy(self.gm.state_dict()))
        opt = Adam(model.parameters(), lr=self.lr)
        model.train()
        for _ in range(epochs):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                opt.zero_grad()
                F.cross_entropy(model(X), y).backward()
                with torch.no_grad():
                    for (nm, p), scv, ccv in zip(model.named_parameters(),
                                                   self.s_cv.values(),
                                                   self.c_cv[cid].values()):
                        if p.grad is not None:
                            p.grad += scv.to(self.device) - ccv.to(self.device)
                opt.step()
        with torch.no_grad():
            for nm, p in model.named_parameters():
                gp = self.gm.state_dict()[nm].to(self.device)
                self.c_cv[cid][nm] = (self.c_cv[cid][nm].to(self.device)
                                      - self.s_cv[nm].to(self.device)
                                      + (gp - p) / (epochs * self.lr))
        return model

    def aggregate(self, models, client_ids, sizes):
        n = len(client_ids)
        for nm in self.s_cv:
            self.s_cv[nm] += sum(self.c_cv[i][nm]
                                  for i in client_ids) / n * (n / sum(sizes))
        self.gm.load_state_dict(fed_avg(models, sizes))


# ── 4. FedNova ────────────────────────────────────────────────────────────────
class FedNovaTrainer:
    name = "FedNova"; upload_mb = 4.82
    def __init__(self, global_model, lr=0.001, device="cpu"):
        self.gm = global_model; self.lr = lr; self.device = device

    def client_update(self, model, loader, epochs):
        model.load_state_dict(copy.deepcopy(self.gm.state_dict()))
        init  = copy.deepcopy(model.state_dict())
        opt   = Adam(model.parameters(), lr=self.lr)
        steps = 0
        for _ in range(epochs):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                opt.zero_grad(); F.cross_entropy(model(X), y).backward()
                opt.step(); steps += 1
        norm_update = {k: (model.state_dict()[k].float() - init[k].float()) / max(steps,1)
                       for k in init}
        return model, norm_update, steps

    def aggregate(self, norm_updates, steps_list, sizes):
        W = sum(sizes)
        st = copy.deepcopy(self.gm.state_dict())
        for k in st:
            delta = sum(norm_updates[i][k] * steps_list[i] * (sizes[i]/W)
                        for i in range(len(norm_updates)))
            st[k] = st[k].float() + delta
        self.gm.load_state_dict(st)


# ── 5. MOON ───────────────────────────────────────────────────────────────────
class MOONTrainer:
    name = "MOON"; upload_mb = 4.82
    def __init__(self, global_model, lr=0.001, mu=1.0, tau=0.5, device="cpu"):
        self.gm   = global_model; self.lr = lr
        self.mu   = mu; self.tau = tau; self.device = device
        self.prev = {}   # client_id -> previous state

    def client_update(self, cid, model, loader, epochs):
        model.load_state_dict(copy.deepcopy(self.gm.state_dict()))
        opt = Adam(model.parameters(), lr=self.lr)
        for _ in range(epochs):
            model.train()
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                opt.zero_grad()
                ce   = F.cross_entropy(model(X), y)
                z_l  = F.normalize(model.get_embedding(X), dim=-1)
                with torch.no_grad():
                    z_g  = F.normalize(self.gm.get_embedding(X.cpu()).to(self.device), dim=-1)
                    z_p  = None
                    if cid in self.prev:
                        pm = copy.deepcopy(model)
                        pm.load_state_dict(self.prev[cid]); pm.to(self.device)
                        z_p = F.normalize(pm.get_embedding(X), dim=-1)
                pos = (z_l * z_g).sum(-1) / self.tau
                neg = (z_l * z_p).sum(-1) / self.tau if z_p is not None else None
                if neg is not None:
                    logits_c = torch.stack([pos, neg], -1)
                    labels_c = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
                    con = F.cross_entropy(logits_c, labels_c)
                else:
                    con = -pos.mean()
                (ce + self.mu * con).backward(); opt.step()
        self.prev[cid] = copy.deepcopy(model.state_dict())
        return model

    def aggregate(self, models, sizes):
        self.gm.load_state_dict(fed_avg(models, sizes))


# ── 6. Ditto ──────────────────────────────────────────────────────────────────
class DittoTrainer:
    name = "Ditto"; upload_mb = 9.64
    def __init__(self, global_model, num_clients, lr=0.001, lam=0.1, device="cpu"):
        self.gm    = global_model; self.lr = lr; self.lam = lam; self.device = device
        self.pm    = [copy.deepcopy(global_model) for _ in range(num_clients)]

    def client_update(self, cid, model, loader, epochs):
        model.load_state_dict(copy.deepcopy(self.gm.state_dict()))
        g_opt = Adam(model.parameters(), lr=self.lr)
        for _ in range(epochs): local_train_one_epoch(model, loader, g_opt, self.device)
        p_model = self.pm[cid].to(self.device)
        p_opt   = Adam(p_model.parameters(), lr=self.lr)
        for _ in range(epochs):
            p_model.train()
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                p_opt.zero_grad()
                ce   = F.cross_entropy(p_model(X), y)
                prox = sum((pp-gp.detach()).norm()**2
                           for pp, gp in zip(p_model.parameters(), self.gm.parameters()))
                (ce + (self.lam/2)*prox).backward(); p_opt.step()
        self.pm[cid] = p_model.cpu()
        return model, p_model

    def aggregate(self, models, sizes):
        self.gm.load_state_dict(fed_avg(models, sizes))


# ── 7. FedMADE ────────────────────────────────────────────────────────────────
class FedMADETrainer:
    name = "FedMADE"; upload_mb = 4.82
    def __init__(self, global_model, num_clients, num_classes=33, lr=0.001, device="cpu"):
        self.gm   = global_model; self.lr = lr; self.device = device
        self.nc   = num_classes
        self.pvec = np.zeros((num_clients, num_classes))

    def client_update(self, cid, model, loader, epochs):
        model.load_state_dict(copy.deepcopy(self.gm.state_dict()))
        opt = Adam(model.parameters(), lr=self.lr)
        for _ in range(epochs): local_train_one_epoch(model, loader, opt, self.device)
        model.eval()
        probs = []
        with torch.no_grad():
            for X, _ in loader:
                probs.append(F.softmax(model(X.to(self.device)), -1).cpu().numpy())
        self.pvec[cid] = np.vstack(probs).mean(0)
        return model

    def _cluster_clients(self, active_ids, k=5):
        from sklearn.cluster import KMeans
        vecs = self.pvec[active_ids]
        if len(active_ids) < k:
            return {i: [cid] for i, cid in enumerate(active_ids)}
        km = KMeans(n_clusters=k, n_init=3, random_state=42).fit(vecs)
        cl = {}
        for i, cid in enumerate(active_ids):
            cl.setdefault(km.labels_[i], []).append(cid)
        return cl

    def aggregate(self, models, client_ids, sizes):
        W  = sum(sizes)
        st = {k: torch.zeros_like(v) for k, v in self.gm.state_dict().items()}
        for i, cid in enumerate(client_ids):
            w = sizes[i] / W
            for k, v in models[i].state_dict().items():
                st[k] += v.float() * w
        self.gm.load_state_dict(st)
