"""# 5.(clustring.py) Unsupervised Learning (Clustering)

Important: DO NOT USE ORIGINAL LABELS. The source DDI data has interaction text
but no supervised risk labels, so we generate pseudo-labels using DBSCAN + FCM.
DBSCAN identifies dense regions and outliers; Fuzzy C-Means converts cluster
membership into soft Low Risk, Medium Risk, and High Risk labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


RISK_LABELS = np.array(["Low Risk", "Medium Risk", "High Risk"])


def fuzzy_membership_for_row(
    row_scaled: pd.DataFrame | np.ndarray,
    clustering_metadata: dict[str, object],
    m: float = 2.0,
) -> dict[str, float]:
    """Compute FCM membership for one inference row from saved FCM centers.

    mu_ik = 1 / sum_j((||x-c_i|| / ||x-c_j||) ** (2/(m-1)))

    The saved ``risk_map`` converts raw FCM cluster IDs to clinical labels, so
    callers can directly read mu(predicted_label).
    """
    centers = np.asarray(clustering_metadata.get("fcm_centers"), dtype=float)
    if centers.ndim != 2 or centers.size == 0:
        return {str(label): 0.0 for label in RISK_LABELS}

    values = row_scaled.to_numpy() if isinstance(row_scaled, pd.DataFrame) else np.asarray(row_scaled)
    x = np.asarray(values, dtype=float).reshape(1, -1)
    distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2).reshape(-1)
    distances = np.maximum(distances, 1e-12)
    inv = distances ** (-2 / (m - 1))
    raw_membership = inv / np.maximum(inv.sum(), 1e-12)

    risk_map = clustering_metadata.get("risk_map") or {}
    by_label: dict[str, float] = {str(label): 0.0 for label in RISK_LABELS}
    for cluster_id, membership_value in enumerate(raw_membership):
        label = risk_map.get(cluster_id, risk_map.get(str(cluster_id), str(cluster_id)))
        by_label[str(label)] = float(membership_value)

    return by_label


def tune_dbscan(scaled_features: pd.DataFrame) -> DBSCAN:
    """Tune DBSCAN eps/min_samples without using labels."""
    
    n_samples = len(scaled_features)

    min_samples_grid = [4, 6, 8, 10]

    best_model = None
    best_score = -np.inf

    for min_samples in min_samples_grid:

        k = min(min_samples, max(2, n_samples - 1))

        distances, _ = (
            NearestNeighbors(n_neighbors=k)
            .fit(scaled_features)
            .kneighbors(scaled_features)
        )

        candidates = np.quantile(
            distances[:, -1],
            [0.55, 0.65, 0.75, 0.85, 0.95],
        )

        for eps in candidates:

            safe_eps = max(float(eps), 0.1)

            model = DBSCAN(
                eps=safe_eps,
                min_samples=min_samples,
            )

            labels = model.fit_predict(scaled_features)

            clusters = set(labels) - {-1}

            noise_ratio = float(np.mean(labels == -1))

            if len(clusters) < 2 or noise_ratio > 0.7:
                score = -noise_ratio
            else:
                score = (
                    silhouette_score(scaled_features, labels)
                    - (0.25 * noise_ratio)
                )

            if score > best_score:
                best_score = score
                best_model = model

    return best_model or DBSCAN(eps=0.8,min_samples=6,).fit(scaled_features)


def fuzzy_c_means(
    x: np.ndarray,
    n_clusters: int = 3,
    m: float = 2.0,
    max_iter: int = 200,
    error: float = 1e-5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Small, dependency-free Fuzzy C-Means implementation."""
    rng = np.random.default_rng(seed)
    n_samples = x.shape[0]
    membership = rng.random((n_samples, n_clusters))
    membership = membership / membership.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        previous = membership.copy()
        um = membership**m
        centers = (um.T @ x) / np.maximum(um.sum(axis=0)[:, None], 1e-12)
        distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        distances = np.maximum(distances, 1e-12)
        inv = distances ** (-2 / (m - 1))
        membership = inv / inv.sum(axis=1, keepdims=True)
        if np.linalg.norm(membership - previous) < error:
            break

    return centers, membership


def generate_pseudo_labels(
    scaled_features: pd.DataFrame,
    original_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Generate Low/Medium/High pseudo-labels with DBSCAN + FCM."""
    dbscan = tune_dbscan(scaled_features)
    dbscan_labels = dbscan.fit_predict(scaled_features)
    centers, membership = fuzzy_c_means(scaled_features.to_numpy(), n_clusters=3)
    hard_fcm_clusters = membership.argmax(axis=1)

    # Rank FCM clusters by real feature risk intensity to name them as risk tiers.
    cluster_risk = {}
    risk_columns = [
        "interaction_frequency",
        "co_occurrence",
        "similarity_score",
        "description_severity",
        "description_length",
        "clinical_prior_mean",
    ]
    for cluster_id in range(3):
        rows = original_features.iloc[np.where(hard_fcm_clusters == cluster_id)[0]]
        cluster_risk[cluster_id] = float(rows[risk_columns].mean().mean()) if len(rows) else 0.0

    ordered_clusters = [c for c, _ in sorted(cluster_risk.items(), key=lambda item: item[1])]
    risk_map = {cluster_id: RISK_LABELS[i] for i, cluster_id in enumerate(ordered_clusters)}
    labels = np.array([risk_map[c] for c in hard_fcm_clusters], dtype=object)

    # DBSCAN outliers are treated as high-risk anomalies because unseen patterns
    # in pharmacological interactions deserve conservative attention.
    labels[dbscan_labels == -1] = "High Risk"

    result = pd.DataFrame(
        {
            "dbscan_cluster": dbscan_labels,
            "fcm_cluster": hard_fcm_clusters,
            "pseudo_label": labels,
            "low_membership": membership[:, ordered_clusters[0]],
            "medium_membership": membership[:, ordered_clusters[1]],
            "high_membership": membership[:, ordered_clusters[2]],
            "pseudo_label_confidence": membership.max(axis=1),
        }
    )
    metadata = {
        "dbscan_eps": float(dbscan.eps),
        "dbscan_min_samples": int(dbscan.min_samples),
        "dbscan_outlier_count": int(np.sum(dbscan_labels == -1)),
        "fcm_centers": centers,
        "risk_map": risk_map,
        "message": "We generate pseudo-labels using DBSCAN + FCM",
    }
    return result, metadata

