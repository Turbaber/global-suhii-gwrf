"""Core functions for geographically weighted random forest."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def adaptive_bisquare_weights(
    coordinates: np.ndarray,
    center: np.ndarray,
    number_of_neighbors: int,
) -> np.ndarray:
    """Return adaptive bi-square weights for one local model."""
    distances = np.linalg.norm(coordinates - center, axis=1)
    neighbor_count = min(max(int(number_of_neighbors), 1), len(distances))
    bandwidth = np.partition(distances, neighbor_count - 1)[neighbor_count - 1]

    if bandwidth <= np.finfo(float).eps:
        return (distances <= np.finfo(float).eps).astype(float)

    scaled_distances = distances / bandwidth
    weights = np.zeros_like(distances, dtype=float)
    inside_bandwidth = scaled_distances <= 1
    weights[inside_bandwidth] = (1 - scaled_distances[inside_bandwidth] ** 2) ** 2
    return weights


def fit_local_random_forest(
    features: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
    parameters: dict,
) -> RandomForestRegressor:
    """Fit one weighted local random-forest model."""
    model = RandomForestRegressor(
        **parameters,
        n_jobs=1,
        random_state=42,
    )
    model.fit(features, target, sample_weight=sample_weights)
    return model

