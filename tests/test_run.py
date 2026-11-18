"""v5 checks that need torch (run on Kaggle in notebook cell 3): ROC-AUC, the new models, one tiny training run."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import MODELS, build  # noqa: E402
from run import cell_key, roc_auc, train_eval  # noqa: E402


def brute_auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    d = pos[:, None] - neg[None, :]
    return ((d > 0).float() + 0.5 * (d == 0).float()).mean().item()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_roc_auc_matches_bruteforce_with_ties(seed):
    g = torch.Generator().manual_seed(seed)
    score = torch.randint(0, 20, (300,), generator=g).float()       # many ties
    y = torch.randint(0, 2, (300,), generator=g)
    assert abs(roc_auc(score, y) - brute_auc(score, y)) < 1e-6


def test_mlp_ignores_edges_and_gcnres_uses_them():
    x = torch.randn(6, 5)
    ei = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    for name, uses_graph in (("mlp", False), ("gcnres", True)):
        torch.manual_seed(0)
        m = build(name, 5, 3).eval()
        same = torch.allclose(m(x, ei), m(x, torch.empty(2, 0, dtype=torch.long)))
        assert same != uses_graph, name


def test_v4_rows_count_as_split0():
    r = {"dataset": "roman", "model": "gcn", "strategy": "random", "sparsity": "0.95", "seed": "3"}
    assert cell_key(r) == ("roman", 0, "gcn", "random", 0.95, 3)


@pytest.mark.parametrize("n_cls", [2, 4])
def test_train_eval_all_models(n_cls):
    class D:
        pass
    d = D()
    d.x, d.y, d.num_features = torch.randn(40, 8), torch.arange(40) % n_cls, 8
    ei = torch.randint(0, 40, (2, 120))
    masks = (torch.arange(40) < 20, (torch.arange(40) >= 20) & (torch.arange(40) < 30), torch.arange(40) >= 30)
    for m in MODELS:
        tf, tb, vb, eb = train_eval(m, d, ei, masks, 0, torch.device("cpu"))
        assert 0 <= tf <= 1 and 0 <= tb <= 1 and 0 <= eb < 200
