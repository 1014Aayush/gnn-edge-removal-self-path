"""Controlled edge removal on undirected graphs (paper §3.1), in plain NumPy.

PyG stores each undirected edge as two directed arcs. We collapse to unique pairs
(u, v) with u < v, choose pairs to remove, and mirror the kept pairs back into both
directions. Sampling arcs independently would mostly delete one direction while
information keeps flowing through the other.
"""
import numpy as np

STRATEGIES = ("random", "hetero_first", "homo_first")


def unique_pairs(edge_index: np.ndarray) -> np.ndarray:
    """(2, E) directed arcs -> (P, 2) unique undirected pairs u < v (self-loops dropped)."""
    u, v = edge_index
    keep = u != v
    a, b = np.minimum(u[keep], v[keep]), np.maximum(u[keep], v[keep])
    return np.unique(np.stack([a, b], 1), axis=0)


def edge_homophily(pairs: np.ndarray, y: np.ndarray) -> float:
    return float((y[pairs[:, 0]] == y[pairs[:, 1]]).mean())


def remove_edges(pairs: np.ndarray, y: np.ndarray, sparsity: float, strategy: str,
                 rng: np.random.Generator) -> np.ndarray:
    """Remove round(sparsity * P) unique pairs; return the kept pairs as (2, 2K) arcs.

    random       : uniformly random pairs
    hetero_first : cross-label pairs (y_u != y_v) first, random order within each group
    homo_first   : same-label pairs first
    """
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    P = len(pairs)
    k = int(round(sparsity * P))
    key = rng.random(P)
    cross = y[pairs[:, 0]] != y[pairs[:, 1]]
    if strategy == "hetero_first":
        key = key + (~cross)          # cross pairs sort first
    elif strategy == "homo_first":
        key = key + cross             # same-label pairs sort first
    kept = pairs[np.argsort(key, kind="stable")[k:]]
    return np.concatenate([kept.T, kept.T[::-1]], axis=1)
