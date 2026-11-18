"""Two-layer GNNs (paper §3.2) plus the two attention models added in this extension.

Design (2 x 2):            no separate self-path      separate self-path
    mean aggregation        GCN (self-loops)           GraphSAGE (root weight)
    attention               GAT (self-loops)           GAT-sep (no self-loops + root weight)

GAT-sep follows Platonov et al. (ICLR 2023): attention over neighbours only, plus a
separate linear map of the node's own features, summed.

v5 adds two controls:
    GCN-res  GCN plus a linear residual (skip) map of the node's own features in each layer,
             summed. This is the self-path a "GCN with residual connections" gets. The 2 x 2
             predicts it behaves like GraphSAGE on Roman-empire (README, P5).
    MLP      the same two-layer network with no graph at all: the graph-free baseline, trained
             with the identical protocol (replaces the published baselines in Figure 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

HIDDEN, HEADS, HEAD_DIM, DROPOUT = 32, 8, 8, 0.5
# Whether dropout is also applied to the input features (Kipf & Welling style). The paper
# only says "dropout 0.5"; v1 used input dropout and under-reproduced Roman-empire by ~2-3
# points, so both settings are validated against the paper's Table 1 (README, P1).
INPUT_DROPOUT = False  # validated: reproduces the paper (README)


class GCN(nn.Module):
    def __init__(self, d_in, n_cls):
        super().__init__()
        self.c1, self.c2 = GCNConv(d_in, HIDDEN), GCNConv(HIDDEN, n_cls)

    def forward(self, x, ei):
        x = F.dropout(x, DROPOUT, self.training and INPUT_DROPOUT)
        x = F.dropout(F.relu(self.c1(x, ei)), DROPOUT, self.training)
        return self.c2(x, ei)


class SAGE(nn.Module):
    def __init__(self, d_in, n_cls):
        super().__init__()
        self.c1, self.c2 = SAGEConv(d_in, HIDDEN, aggr="mean"), SAGEConv(HIDDEN, n_cls, aggr="mean")

    def forward(self, x, ei):
        x = F.dropout(x, DROPOUT, self.training and INPUT_DROPOUT)
        x = F.dropout(F.relu(self.c1(x, ei)), DROPOUT, self.training)
        return self.c2(x, ei)


class GCNRes(nn.Module):
    """GCN with a linear residual connection per layer: h = GCNConv(x) + W_r x."""

    def __init__(self, d_in, n_cls):
        super().__init__()
        self.c1, self.c2 = GCNConv(d_in, HIDDEN), GCNConv(HIDDEN, n_cls)
        self.r1, self.r2 = nn.Linear(d_in, HIDDEN), nn.Linear(HIDDEN, n_cls)

    def forward(self, x, ei):
        x = F.dropout(x, DROPOUT, self.training and INPUT_DROPOUT)
        h = F.dropout(F.relu(self.c1(x, ei) + self.r1(x)), DROPOUT, self.training)
        return self.c2(h, ei) + self.r2(h)


class MLP(nn.Module):
    """Graph-free control: two linear layers, same width, dropout and training as GCN. Ignores edges."""

    def __init__(self, d_in, n_cls):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d_in, HIDDEN), nn.Linear(HIDDEN, n_cls)

    def forward(self, x, ei=None):
        x = F.dropout(x, DROPOUT, self.training and INPUT_DROPOUT)
        return self.l2(F.dropout(F.relu(self.l1(x)), DROPOUT, self.training))


class GAT(nn.Module):
    """Standard 8-head GAT (Velickovic et al. 2018): 8 heads x 8 dims, single-head output layer."""

    def __init__(self, d_in, n_cls, self_path=False):
        super().__init__()
        loops = not self_path
        self.c1 = GATConv(d_in, HEAD_DIM, heads=HEADS, dropout=DROPOUT, add_self_loops=loops)
        self.c2 = GATConv(HEADS * HEAD_DIM, n_cls, heads=1, dropout=DROPOUT, add_self_loops=loops)
        self.self_path = self_path
        if self_path:
            self.r1 = nn.Linear(d_in, HEADS * HEAD_DIM)
            self.r2 = nn.Linear(HEADS * HEAD_DIM, n_cls)

    def forward(self, x, ei):
        x = F.dropout(x, DROPOUT, self.training and INPUT_DROPOUT)
        h = self.c1(x, ei) + (self.r1(x) if self.self_path else 0)
        h = F.dropout(F.relu(h), DROPOUT, self.training)
        return self.c2(h, ei) + (self.r2(h) if self.self_path else 0)


MODELS = ("gcn", "sage", "gat", "gatsep", "gcnres", "mlp")
GRAPH_FREE = {"mlp"}   # trained once per (dataset, split, seed), not per removal level


def build(name, d_in, n_cls):
    return {"gcn": lambda: GCN(d_in, n_cls), "sage": lambda: SAGE(d_in, n_cls),
            "gat": lambda: GAT(d_in, n_cls, self_path=False),
            "gatsep": lambda: GAT(d_in, n_cls, self_path=True),
            "gcnres": lambda: GCNRes(d_in, n_cls), "mlp": lambda: MLP(d_in, n_cls)}[name]()


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for n in MODELS:
        m = build(n, 1433, 7)
        print(n, n_params(m))
        out = m(torch.randn(5, 1433), torch.tensor([[0, 1, 2], [1, 2, 3]]))
        assert out.shape == (5, 7)
