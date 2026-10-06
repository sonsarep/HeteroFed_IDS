"""
experiments/dbec.py
────────────────────
Algorithm 1: Device-Behavioural Embedding Clustering (DBEC)
(Paper Section IV-B)

Steps:
  Phase 1 — Each client uploads 128-dim embedding (512 bytes/epoch)
  Phase 2 — Fog runs silhouette-guided mini-batch k-means
  Phase 3 — Clients assigned to closest cluster centroid
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


def select_optimal_k(embeddings: np.ndarray, k_range=(3,5,7),
                      n_iter=100, seed=42):
    """
    Silhouette-guided k selection (Algorithm 1, Phase 2, Steps 6-9).

    Returns:
        k_star  : optimal cluster count
        scores  : {k: silhouette_score}
    """
    scores = {}
    for k in k_range:
        if k >= len(embeddings):
            continue
        km     = MiniBatchKMeans(n_clusters=k, max_iter=n_iter,
                                  random_state=seed, n_init=3)
        labels = km.fit_predict(embeddings)
        if len(np.unique(labels)) < 2:
            scores[k] = -1.0
        else:
            scores[k] = silhouette_score(
                embeddings, labels,
                sample_size=min(1000, len(embeddings)))
    k_star = max(scores, key=scores.get)
    return k_star, scores


def assign_clusters(embeddings: np.ndarray, k_star: int,
                     n_iter=100, seed=42):
    """
    Cluster assignment (Algorithm 1, Phase 3, Steps 10-11).

    Returns:
        labels    : (K,) cluster assignment per client
        centroids : (k_star, embed_dim) cluster centroids
    """
    km = MiniBatchKMeans(n_clusters=k_star, max_iter=n_iter,
                          random_state=seed, n_init=5)
    labels    = km.fit_predict(embeddings)
    centroids = km.cluster_centers_
    return labels, centroids


class DBEC:
    """
    Full DBEC module (Algorithm 1).
    Maintains cluster state across rounds.
    """
    def __init__(self, encoder, k_range=(3,5,7), sample_size=200,
                 coldstart_size=50, n_iter=100, hierarchical_thresh=200, seed=42):
        self.encoder             = encoder
        self.k_range             = k_range
        self.sample_size         = sample_size
        self.coldstart_size      = coldstart_size
        self.n_iter              = n_iter
        self.hierarchical_thresh = hierarchical_thresh
        self.seed                = seed
        # State
        self.cluster_labels  = None
        self.centroids       = None
        self.k_star          = None
        self.embeddings      = None

    def update(self, client_data_list, round_num=0, new_client_ids=None):
        """
        Run full DBEC update for one round.

        Args:
            client_data_list : list of numpy arrays (or None if client inactive)
            round_num        : current communication round
            new_client_ids   : set of client indices that joined this round
                               (uses cold-start sample_size=50)
        Returns:
            cluster_labels : (K,) assignment
        """
        K         = len(client_data_list)
        embed_dim = self.encoder.embed_dim
        embeddings= np.zeros((K, embed_dim))
        new_ids   = set(new_client_ids or [])

        for k, data in enumerate(client_data_list):
            if data is None or len(data) == 0:
                continue
            ssize = self.coldstart_size if k in new_ids else self.sample_size
            embeddings[k] = self.encoder.encode_sample(
                data, sample_size=ssize,
                seed=self.seed + round_num * K + k)

        self.embeddings = embeddings

        # Scale k-range for large deployments
        k_range = (self._large_k_range() if K > 100 else self.k_range)
        self.k_star, _ = select_optimal_k(embeddings, k_range,
                                           self.n_iter, self.seed + round_num)
        self.cluster_labels, self.centroids = assign_clusters(
            embeddings, self.k_star, self.n_iter, self.seed + round_num)
        return self.cluster_labels

    def _large_k_range(self):
        return [3, 5, 7, 10, 15, 20]

    def get_cluster_members(self):
        """Return {cluster_id: [client_indices]}."""
        if self.cluster_labels is None:
            return {}
        clusters = {}
        for k, c in enumerate(self.cluster_labels):
            clusters.setdefault(int(c), []).append(k)
        return clusters

    def embedding_upload_bytes(self, num_clients):
        """Total upload cost for DBEC per epoch (512 bytes per client)."""
        return num_clients * 512
