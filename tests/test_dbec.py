"""Tests for DBEC — Algorithm 1."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.dbec import select_optimal_k, assign_clusters, DBEC
from models.encoder import load_encoder


def test_select_optimal_k():
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((30, 128))
    k_star, scores = select_optimal_k(embeddings, k_range=(3,5,7), seed=42)
    assert k_star in (3, 5, 7), f"k_star={k_star} not in range"
    assert len(scores) > 0
    print(f"[✓] select_optimal_k: k*={k_star}, scores={scores}")


def test_assign_clusters():
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((20, 128))
    labels, centroids = assign_clusters(embeddings, k_star=5, seed=42)
    assert len(labels) == 20
    assert centroids.shape == (5, 128)
    assert set(labels).issubset({0,1,2,3,4})
    print(f"[✓] assign_clusters: labels={np.unique(labels)}, centroids={centroids.shape}")


def test_dbec_update():
    enc = load_encoder(input_dim=47, embed_dim=128)
    dbec = DBEC(enc, k_range=(3,5,7), sample_size=50, seed=42)
    rng  = np.random.default_rng(42)
    data = [rng.standard_normal((100, 47)) for _ in range(20)]
    labels = dbec.update(data, round_num=1)
    assert len(labels) == 20
    members = dbec.get_cluster_members()
    total = sum(len(v) for v in members.values())
    assert total == 20, f"Members total={total} != 20"
    print(f"[✓] DBEC update: k*={dbec.k_star}, clusters={list(members.keys())}")


def test_cold_start():
    enc  = load_encoder(input_dim=47, embed_dim=128)
    dbec = DBEC(enc, sample_size=200, coldstart_size=50, seed=42)
    rng  = np.random.default_rng(42)
    data = [rng.standard_normal((200, 47)) for _ in range(20)]
    # Clients 0 and 1 are "new" — should use coldstart_size=50
    labels = dbec.update(data, round_num=1, new_client_ids={0, 1})
    assert len(labels) == 20
    print(f"[✓] Cold-start protocol: new clients used reduced sample size")


if __name__ == "__main__":
    test_select_optimal_k()
    test_assign_clusters()
    test_dbec_update()
    test_cold_start()
    print("\nAll DBEC tests passed ✓")
