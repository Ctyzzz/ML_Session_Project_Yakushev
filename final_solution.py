from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
DATA_PATH = Path("Run200_Wave_0_1.txt")


def load_raw(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path, sep=" ", header=None, skipinitialspace=True)
    metadata = data.iloc[:, :4].to_numpy(dtype=float)
    signals = data.drop(columns=[0, 1, 2, 3, 504]).to_numpy(dtype=float)
    return metadata, signals


def build_features(signals: np.ndarray, metadata: np.ndarray) -> pd.DataFrame:
    baseline = np.median(signals[:, :120], axis=1)
    centered = baseline[:, None] - signals
    positive = np.clip(centered, 0, None)

    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, np.ones(5) / 5, mode="same"),
        1,
        centered,
    )
    peak_idx = np.argmax(smoothed[:, 120:240], axis=1) + 120
    peak = smoothed[np.arange(len(signals)), peak_idx]

    sample_idx = np.arange(signals.shape[1])[None, :]
    rel_idx = sample_idx - peak_idx[:, None]

    charge_full = positive[:, 120:420].sum(axis=1)
    charge_main = positive[:, 120:260].sum(axis=1)
    tail_fixed = positive[:, 170:360].sum(axis=1)
    fast_fixed = positive[:, 120:170].sum(axis=1)

    rel_full = (rel_idx >= -10) & (rel_idx <= 180)
    rel_tail = (rel_idx >= 25) & (rel_idx <= 180)
    rel_fast = (rel_idx >= -10) & (rel_idx < 25)

    rel_full_charge = (positive * rel_full).sum(axis=1)
    rel_tail_charge = (positive * rel_tail).sum(axis=1)
    rel_fast_charge = (positive * rel_fast).sum(axis=1)

    features = pd.DataFrame(
        {
            "log_charge": np.log1p(charge_full),
            "log_charge_main": np.log1p(charge_main),
            "log_peak": np.log1p(np.maximum(peak, 1)),
            "peak": peak,
            "peak_idx": peak_idx,
            "psd_fixed": tail_fixed / np.maximum(charge_full, 1),
            "psd_rel": rel_tail_charge / np.maximum(rel_full_charge, 1),
            "fast_rel": rel_fast_charge / np.maximum(rel_full_charge, 1),
            "centroid": (
                positive[:, 120:420] * np.arange(120, 420)
            ).sum(axis=1)
            / np.maximum(positive[:, 120:420].sum(axis=1), 1),
            "noise": signals[:, :120].std(axis=1),
            "baseline": baseline,
            "expected_frac": positive[:, 135:230].sum(axis=1)
            / np.maximum(positive.sum(axis=1), 1),
            "meta_1": metadata[:, 1],
            "meta_2": metadata[:, 2],
        }
    )

    rel_window = np.arange(-10, 170)
    aligned = np.zeros((len(signals), len(rel_window)))
    for row_id, peak_position in enumerate(peak_idx):
        positions = peak_position + rel_window
        valid = (positions >= 0) & (positions < signals.shape[1])
        aligned[row_id, valid] = positive[row_id, positions[valid]]

    aligned = aligned / np.maximum(aligned.sum(axis=1, keepdims=True), 1)
    binned_shape = aligned.reshape(len(aligned), 18, 10).sum(axis=2)
    standardized_shape = (
        binned_shape - binned_shape.mean(axis=0)
    ) / (binned_shape.std(axis=0) + 1e-9)
    u_matrix, singular_values, _ = np.linalg.svd(standardized_shape, full_matrices=False)
    shape_pcs = u_matrix[:, :6] * singular_values[:6]
    for component_id in range(shape_pcs.shape[1]):
        features[f"shape_pc_{component_id + 1}"] = shape_pcs[:, component_id]

    return features


def build_base_features(signals: np.ndarray) -> pd.DataFrame:
    """Peak-relative features for the PCA/PSD base model."""
    baseline = np.median(signals[:, :120], axis=1)
    centered = baseline[:, None] - signals
    positive = np.clip(centered, 0, None)

    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, np.ones(5) / 5, mode="same"),
        1,
        centered,
    )
    peak_idx = np.argmax(smoothed[:, 120:240], axis=1) + 120
    peak = smoothed[np.arange(len(signals)), peak_idx]

    idx = np.arange(signals.shape[1])[None, :]
    rel = idx - peak_idx[:, None]
    full = (rel >= -10) & (rel <= 180)
    tail = (rel >= 25) & (rel <= 180)
    charge = (positive * full).sum(axis=1)
    psd = (positive * tail).sum(axis=1) / np.maximum(charge, 1)

    return pd.DataFrame(
        {
            "log_charge": np.log1p(charge),
            "log_peak": np.log1p(np.maximum(peak, 1)),
            "psd_rel": psd,
        }
    )


def pca_boundary_score(features: pd.DataFrame) -> np.ndarray:
    values = features[["log_charge", "log_peak"]].to_numpy()
    values = (values - values.mean(axis=0)) / values.std(axis=0)
    _, _, vt = np.linalg.svd(values, full_matrices=False)
    normal = np.array([-vt[0, 1], vt[0, 0]])
    score = values @ normal
    psd = features["psd_rel"].to_numpy()
    if psd[score > 0].mean() < psd[score < 0].mean():
        score = -score
    return score


def rolling_psd_residual(features: pd.DataFrame, window: int = 1201) -> np.ndarray:
    log_charge = features["log_charge"].to_numpy()
    psd = features["psd_rel"].to_numpy()
    order = np.argsort(log_charge)
    sorted_psd = psd[order]
    trend = np.empty(len(features))
    half_window = window // 2
    for rank, row_id in enumerate(order):
        left = max(0, rank - half_window)
        right = min(len(order), rank + half_window + 1)
        trend[row_id] = np.median(sorted_psd[left:right])
    residual = psd - trend
    if psd[residual > 0].mean() < psd[residual < 0].mean():
        residual = -residual
    return residual


def labels_from_ensemble(
    pca_score: np.ndarray,
    psd_residual: np.ndarray,
    pca_quantile: float = 0.33,
    psd_quantile: float = 0.31,
) -> np.ndarray:
    pca_threshold = np.quantile(np.abs(pca_score), pca_quantile)
    psd_threshold = np.quantile(np.abs(psd_residual), psd_quantile)
    pca_vote = np.where(
        pca_score > pca_threshold,
        1,
        np.where(pca_score < -pca_threshold, 0, 2),
    )
    psd_vote = np.where(
        psd_residual > psd_threshold,
        1,
        np.where(psd_residual < -psd_threshold, 0, 2),
    )
    return np.where((pca_vote == psd_vote) & (pca_vote != 2), pca_vote, 2)


def apply_perm_120(raw_labels: np.ndarray) -> np.ndarray:
    mapping = np.array([1, 2, 0])
    return mapping[raw_labels]


def robust_scale(values: np.ndarray) -> np.ndarray:
    center = np.median(values, axis=0)
    scale = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    scale = np.where(scale == 0, values.std(axis=0) + 1e-9, scale)
    return (values - center) / scale


def kmeans(values: np.ndarray, n_clusters: int = 2, n_init: int = 30) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)
    n_rows = len(values)
    best_labels = None
    best_inertia = np.inf

    for _ in range(n_init):
        centers = values[rng.choice(n_rows, n_clusters, replace=False)].copy()
        labels = np.zeros(n_rows, dtype=int)
        for _ in range(100):
            distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                if mask.any():
                    centers[cluster_id] = values[mask].mean(axis=0)
        inertia = ((values - centers[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = values.max(axis=axis, keepdims=True)
    return (
        max_values + np.log(np.exp(values - max_values).sum(axis=axis, keepdims=True))
    ).squeeze(axis)


def log_gaussian(values: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    n_features = values.shape[1]
    sign, log_det = np.linalg.slogdet(covariance)
    if sign <= 0:
        covariance = covariance + np.eye(n_features) * 1e-3
        _, log_det = np.linalg.slogdet(covariance)
    inverse = np.linalg.inv(covariance)
    centered = values - mean
    quadratic = np.sum((centered @ inverse) * centered, axis=1)
    return -0.5 * (n_features * np.log(2 * np.pi) + log_det + quadratic)


def fit_gmm2(values: np.ndarray, n_iter: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    n_rows, n_features = values.shape
    labels = kmeans(values, n_clusters=2)
    weights = np.array([(labels == cluster).mean() for cluster in range(2)])
    means = np.array([values[labels == cluster].mean(axis=0) for cluster in range(2)])
    covariances = np.array(
        [
            np.cov(values[labels == cluster].T) + np.eye(n_features) * 1e-4
            for cluster in range(2)
        ]
    )

    for _ in range(n_iter):
        log_probs = np.column_stack(
            [
                np.log(weights[cluster] + 1e-12)
                + log_gaussian(values, means[cluster], covariances[cluster])
                for cluster in range(2)
            ]
        )
        responsibilities = np.exp(log_probs - logsumexp(log_probs, axis=1)[:, None])
        nk = responsibilities.sum(axis=0) + 1e-12
        weights = nk / n_rows
        means = (responsibilities.T @ values) / nk[:, None]
        for cluster in range(2):
            centered = values - means[cluster]
            covariances[cluster] = (
                centered.T @ (centered * responsibilities[:, [cluster]]) / nk[cluster]
                + np.eye(n_features) * 1e-4
            )

    return weights, means, covariances


def labels_from_gmm2(
    features: pd.DataFrame,
    columns: list[str],
    uncertain_fraction: float,
) -> np.ndarray:
    values = robust_scale(features[columns].to_numpy())
    weights, means, covariances = fit_gmm2(values)
    log_probs = np.column_stack(
        [
            np.log(weights[cluster] + 1e-12)
            + log_gaussian(values, means[cluster], covariances[cluster])
            for cluster in range(2)
        ]
    )
    posterior = np.exp(log_probs - logsumexp(log_probs, axis=1)[:, None])
    component = posterior.argmax(axis=1)
    confidence = np.abs(posterior[:, 1] - posterior[:, 0])

    psd_means = [
        features.loc[component == cluster, "psd_rel"].mean() for cluster in range(2)
    ]
    high_component = int(np.argmax(psd_means))
    raw_labels = np.where(component == high_component, 1, 0)
    raw_labels[confidence <= np.quantile(confidence, uncertain_fraction)] = 2
    return apply_perm_120(raw_labels)


def consensus_label(models: list[np.ndarray], min_votes: int) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack(models)
    labels = np.zeros(stacked.shape[1], dtype=int)
    ok = np.zeros(stacked.shape[1], dtype=bool)
    for label in [0, 1, 2]:
        votes = (stacked == label).sum(axis=0)
        mask = votes >= min_votes
        labels[mask] = label
        ok |= mask
    return labels, ok


def make_final_submission() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata, signals = load_raw()
    features = build_features(signals, metadata)

    base_features = build_base_features(signals)
    pca_score = pca_boundary_score(base_features)
    psd_residual = rolling_psd_residual(base_features)
    raw_base = labels_from_ensemble(pca_score, psd_residual, 0.33, 0.31)
    base = apply_perm_120(raw_base)

    shape_phys_columns = [
        "shape_pc_1",
        "shape_pc_2",
        "shape_pc_3",
        "log_charge",
        "log_peak",
        "psd_rel",
        "centroid",
    ]
    shape4_columns = [
        "shape_pc_1",
        "shape_pc_2",
        "shape_pc_3",
        "shape_pc_4",
        "log_charge",
        "psd_rel",
    ]
    phys6_columns = [
        "log_charge",
        "log_peak",
        "psd_rel",
        "psd_fixed",
        "centroid",
        "expected_frac",
    ]

    gmm_u44 = labels_from_gmm2(features, shape_phys_columns, uncertain_fraction=0.44)
    gmm_u46 = labels_from_gmm2(features, shape_phys_columns, uncertain_fraction=0.46)
    gmm_u48 = labels_from_gmm2(features, shape_phys_columns, uncertain_fraction=0.48)
    gmm_shape4 = labels_from_gmm2(features, shape4_columns, uncertain_fraction=0.46)
    gmm_phys6 = labels_from_gmm2(features, phys6_columns, uncertain_fraction=0.46)

    psd = features["psd_rel"].to_numpy()
    labels = base.copy()
    low_tail = psd <= np.quantile(psd, 0.12)
    labels[low_tail] = gmm_u44[low_tail]

    ring = (psd > np.quantile(psd, 0.12)) & (psd <= np.quantile(psd, 0.16))
    agreed_label, agreed = consensus_label(
        [gmm_u46, gmm_shape4, gmm_phys6, gmm_u44, gmm_u48],
        min_votes=5,
    )
    mask = ring & agreed & (agreed_label != labels)
    labels[mask] = agreed_label[mask]

    submission = pd.DataFrame({"index": np.arange(len(labels)), "cluster": labels})
    diagnostic = features.assign(cluster=labels)
    return submission, diagnostic



if __name__ == "__main__":
    final_submission, final_diagnostic = make_final_submission()
    final_submission.to_csv("submission.csv", index=False)
    final_diagnostic.to_csv("final_features_with_clusters.csv", index=False)
    print(final_submission["cluster"].value_counts().sort_index())
