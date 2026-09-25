"""Grouping embeddings into clusters, one summary per cluster.

The RAPTOR paper uses Gaussian mixtures over UMAP-reduced vectors with BIC to
pick the cluster count. On a few dozen chunks per paper that machinery is not
obviously needed, so this starts simpler and gets measured (rag.md 7.9.1):
Ward-linkage agglomerative clustering on unit vectors - deterministic, and it
tends toward balanced clusters, which keeps each summarisation prompt small.
"""

import math

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cluster(vectors: list[list[float]], cluster_size: int) -> list[list[int]]:
    """Indices of `vectors` grouped into clusters of about `cluster_size`.

    A cluster larger than twice the target is clustered again, so no single
    summary has to read an unbounded amount of text.
    """
    if not vectors:
        return []
    matrix = np.asarray(vectors, dtype=float)
    groups = _cluster(matrix, list(range(len(vectors))), cluster_size)
    return sorted(groups, key=lambda group: group[0])


def _cluster(matrix: np.ndarray, indices: list[int], cluster_size: int) -> list[list[int]]:
    count = math.ceil(len(indices) / cluster_size)
    if count <= 1:
        return [indices]

    labels = AgglomerativeClustering(n_clusters=count, linkage="ward").fit_predict(
        _unit(matrix[indices])
    )
    groups: dict[int, list[int]] = {}
    for index, label in zip(indices, labels, strict=True):
        groups.setdefault(int(label), []).append(index)

    result: list[list[int]] = []
    for group in groups.values():
        # count >= 2 means every group is smaller than `indices`, so this ends.
        if len(group) > 2 * cluster_size:
            result.extend(_cluster(matrix, group, cluster_size))
        else:
            result.append(group)
    return result


def _unit(matrix: np.ndarray) -> np.ndarray:
    """Ward uses Euclidean distance; on unit vectors that orders pairs like cosine."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)
