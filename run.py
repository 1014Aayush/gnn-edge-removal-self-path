"""Edge-removal sweep (paper §3): datasets x splits x models x strategies x sparsity x seeds.

Every model sees the SAME removed graph for a given (dataset, split, strategy, sparsity, seed),
and model initialisation is seed-matched, so comparisons are paired across strategies
and across architectures. Results are appended to a CSV; finished cells are skipped (resume).

    python run.py --datasets cora roman --models gcn sage gat gatsep --out results/part0.csv
    python run.py --datasets roman amazon tolokers --splits 0 1 2 3 4 5 6 7 8 9 --paired ...
    python run.py --mode validate ...   # quick reproduction check of the paper's GCN/SAGE numbers

--paired runs one seed per split (seed = split index), as in Platonov et al. (ICLR 2023).
--done-from FILE... skips cells that already exist in earlier result files (v4 rows have no
split column and count as split 0).
Graph-free models (MLP) are trained once per (dataset, split, seed) with strategy "none".
Binary datasets (Tolokers) are scored by ROC-AUC, the others by accuracy (column "metric").
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import HeterophilousGraphDataset, Planetoid
from torch_geometric.transforms import NormalizeFeatures

from edge_removal import STRATEGIES, edge_homophily, remove_edges, unique_pairs
import models
from models import GRAPH_FREE, build

EPOCHS, LR, WD = 200, 0.01, 5e-4
FIELDS = ["dataset", "split", "model", "strategy", "sparsity", "seed", "pairs_kept", "homophily_kept", "metric",
          "test_final", "test_bestval", "val_best", "epoch_best", "seconds"]
PLATONOV = {"roman": "Roman-empire", "amazon": "Amazon-ratings", "tolokers": "Tolokers"}


CORA_NORMALIZE = False  # raw Cora features: reproduces the paper (README, v3 validation)


def load(name, root, split=0):
    if name == "cora":
        if split != 0:
            raise ValueError("Cora has one (public) split")
        d = Planetoid(root, "Cora", transform=NormalizeFeatures() if CORA_NORMALIZE else None)[0]
        tr, va, te = d.train_mask, d.val_mask, d.test_mask
    elif name in PLATONOV:
        d = HeterophilousGraphDataset(root, PLATONOV[name])[0]
        tr, va, te = d.train_mask[:, split], d.val_mask[:, split], d.test_mask[:, split]   # provided splits 0-9
    else:
        raise ValueError(name)
    return d, tr, va, te


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def roc_auc(score, y):
    """Binary ROC-AUC (Mann-Whitney, ties averaged) on torch tensors."""
    r = torch.empty_like(score)
    order = score.argsort()
    r[order] = torch.arange(1, len(score) + 1, dtype=score.dtype, device=score.device)
    uniq, inv = torch.unique(score, return_inverse=True)      # average ranks of tied scores
    r = torch.zeros(len(uniq), dtype=score.dtype, device=score.device).index_add_(0, inv, r)[inv] / \
        torch.bincount(inv, minlength=len(uniq)).to(score.dtype)[inv]
    pos = y == 1
    n1, n0 = pos.sum().item(), (~pos).sum().item()
    return ((r[pos].sum().item() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_eval(model_name, data, ei, masks, seed, dev):
    tr, va, te = masks
    seed_all(seed)                                   # seed-matched initialisation
    n_cls = int(data.y.max()) + 1
    model = build(model_name, data.num_features, n_cls).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    x, y = data.x, data.y
    if n_cls == 2:
        score = lambda out, m: roc_auc(out[m].softmax(1)[:, 1], y[m])  # noqa: E731
    else:
        score = lambda out, m: (out[m].argmax(1) == y[m]).float().mean().item()  # noqa: E731
    best_val, best_test, best_ep = -1.0, 0.0, 0
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(x, ei)[tr], y[tr]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x, ei)
        v, t = score(out, va), score(out, te)
        if v > best_val:
            best_val, best_test, best_ep = v, t, ep
    return t, best_test, best_val, best_ep


def cell_key(r):
    return (r["dataset"], int(r.get("split") or 0), r["model"], r["strategy"], float(r["sparsity"]), int(r["seed"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "roman"])
    ap.add_argument("--models", nargs="+", default=["gcn", "sage", "gat", "gatsep"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--splits", nargs="+", type=int, default=[0])
    ap.add_argument("--paired", action="store_true", help="one seed per split: seed = split index (ignores --seeds)")
    ap.add_argument("--mode", default="full", choices=["full", "validate"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--done-from", nargs="*", default=[], help="earlier result CSVs whose cells are skipped")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--input-dropout", type=int, default=0, help="1 = dropout on input features too, 0 = between layers only")
    ap.add_argument("--cora-normalize", type=int, default=0, help="1 = row-normalise Cora features, 0 = raw binary features")
    args = ap.parse_args()
    global CORA_NORMALIZE
    CORA_NORMALIZE = bool(args.cora_normalize)
    print(f"cora feature normalisation: {CORA_NORMALIZE}", flush=True)
    models.INPUT_DROPOUT = bool(args.input_dropout)
    print(f"input dropout: {models.INPUT_DROPOUT}", flush=True)

    levels = [round(0.05 * i, 2) for i in range(20)]           # 0.00 ... 0.95
    strategies = list(STRATEGIES)
    if args.mode == "validate":                                  # the paper's Table 1 operating points, all 5 seeds
        levels = [0.0, 0.2, 0.95]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    for f in [args.out, *args.done_from]:
        if os.path.exists(f):
            with open(f) as fh:
                done.update(cell_key(r) for r in csv.DictReader(fh))
    print(f"cells already done: {len(done)}", flush=True)
    new_file = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    w = csv.DictWriter(fh, FIELDS)
    if new_file:
        w.writeheader()

    dev = torch.device(args.device)
    graph_models = [m for m in args.models if m not in GRAPH_FREE]
    free_models = [m for m in args.models if m in GRAPH_FREE]
    for ds in args.datasets:
        splits = [0] if ds == "cora" else args.splits
        for split in splits:
            data, tr, va, te = load(ds, args.root, split)
            pairs = unique_pairs(data.edge_index.numpy())
            y_np = data.y.numpy()
            h_full = edge_homophily(pairs, y_np)
            metric = "auc" if int(data.y.max()) == 1 else "acc"
            print(f"[{ds} split {split}] nodes={data.num_nodes} unique edges={len(pairs)} edge homophily={h_full:.3f} "
                  f"metric={metric}", flush=True)
            data = data.to(dev)
            masks = (tr.to(dev), va.to(dev), te.to(dev))
            seeds = [split] if (args.paired and ds != "cora") else args.seeds

            def record(m, strat, s, seed, ei, kept, h):
                t0 = time.time()
                tf, tb, vb, eb = train_eval(m, data, ei, masks, seed, dev)
                w.writerow(dict(dataset=ds, split=split, model=m, strategy=strat, sparsity=s, seed=seed, pairs_kept=kept,
                                homophily_kept=round(h, 4), metric=metric, test_final=round(tf, 4),
                                test_bestval=round(tb, 4), val_best=round(vb, 4), epoch_best=eb,
                                seconds=round(time.time() - t0, 2)))
                fh.flush()

            for seed in seeds:
                for m in free_models:                                  # graph-free: once per (split, seed)
                    if (ds, split, m, "none", 0.0, seed) not in done:
                        record(m, "none", 0.0, seed, torch.empty(2, 0, dtype=torch.long, device=dev), len(pairs), h_full)
                for strat in strategies:
                    for s in levels:
                        todo = [m for m in graph_models if (ds, split, m, strat, s, seed) not in done]
                        if not todo:
                            continue
                        # split 0 keeps the v4 removal seeds, so v4 and v5 rows share removed graphs
                        rng = np.random.default_rng(split * 100_000 + seed * 1000 + int(round(s * 100)))
                        ei_np = remove_edges(pairs, y_np, s, strat, rng)
                        kept = ei_np.shape[1] // 2
                        h = edge_homophily(ei_np[:, : kept].T, y_np) if kept else float("nan")
                        ei = torch.as_tensor(ei_np, dtype=torch.long, device=dev)
                        for m in todo:
                            record(m, strat, s, seed, ei, kept, h)
                        print(f"[{ds} split {split}] seed {seed} {strat:12s} s={s:.2f} done ({', '.join(todo)})", flush=True)
    fh.close()
    print("RUN DONE", flush=True)


if __name__ == "__main__":
    main()
