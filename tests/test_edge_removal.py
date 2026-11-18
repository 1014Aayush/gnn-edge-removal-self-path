import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge_removal import edge_homophily, remove_edges, unique_pairs  # noqa: E402


def toy(seed=0, n=60, p=0.15):
    rng = np.random.default_rng(seed)
    A = np.triu(rng.random((n, n)) < p, 1)
    u, v = np.nonzero(A)
    ei = np.concatenate([np.stack([u, v]), np.stack([v, u])], 1)   # both directions, like PyG
    y = rng.integers(0, 3, n)
    return ei, y


def test_unique_pairs_collapses_both_directions():
    ei, _ = toy()
    pairs = unique_pairs(ei)
    assert len(pairs) == ei.shape[1] // 2
    assert (pairs[:, 0] < pairs[:, 1]).all()


@pytest.mark.parametrize("strategy", ["random", "hetero_first", "homo_first"])
@pytest.mark.parametrize("s", [0.0, 0.05, 0.5, 0.95])
def test_count_and_symmetry(strategy, s):
    ei, y = toy()
    pairs = unique_pairs(ei)
    out = remove_edges(pairs, y, s, strategy, np.random.default_rng(1))
    kept = unique_pairs(out)
    assert len(kept) == len(pairs) - int(round(s * len(pairs)))
    arcs = set(map(tuple, out.T))
    assert all((b, a) in arcs for a, b in arcs)          # every kept edge present in both directions
    assert out.shape[1] == 2 * len(kept)


def test_hetero_first_removes_only_cross_until_exhausted():
    ei, y = toy()
    pairs = unique_pairs(ei)
    n_cross = int((y[pairs[:, 0]] != y[pairs[:, 1]]).sum())
    s = (n_cross - 1) / len(pairs)
    kept = unique_pairs(remove_edges(pairs, y, s, "hetero_first", np.random.default_rng(0)))
    same = y[pairs[:, 0]] == y[pairs[:, 1]]
    assert set(map(tuple, pairs[same])) <= set(map(tuple, kept))   # no same-label pair removed


def test_full_hetero_removal_gives_homophily_one():
    ei, y = toy()
    pairs = unique_pairs(ei)
    s = 1 - edge_homophily(pairs, y)
    kept = unique_pairs(remove_edges(pairs, y, s, "hetero_first", np.random.default_rng(0)))
    assert edge_homophily(kept, y) == 1.0


def test_same_seed_same_removal_across_calls():
    ei, y = toy()
    pairs = unique_pairs(ei)
    a = remove_edges(pairs, y, 0.4, "random", np.random.default_rng(7))
    b = remove_edges(pairs, y, 0.4, "random", np.random.default_rng(7))
    assert np.array_equal(a, b)
