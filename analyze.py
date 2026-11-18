"""Reproduction check + extended Table 1 + extended Figure 1 + prediction checks.

    python analyze.py results/*.csv --metric test_final
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# The paper's Table 1 (mean over 5 seeds): full, oracle peak (s), random@.95, oracle@.95, inverse@.95
PAPER = {
    ("cora", "gcn"): (80.6, 88.0, 0.20, 58.9, 61.8, 54.2),
    ("cora", "sage"): (80.4, 86.6, 0.20, 55.8, 56.7, 55.4),
    ("roman", "gcn"): (44.9, 67.0, 0.95, 61.4, 67.0, 61.1),
    ("roman", "sage"): (74.7, 74.8, 0.00, 65.0, 69.7, 64.8),
}
NAMES = {"gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT", "gatsep": "GAT-sep"}
DSN = {"cora": "Cora", "roman": "Roman-empire"}
COL = {"hetero_first": "#2a78d6", "random": "#eb6834", "homo_first": "#1baf7a"}
LAB = {"hetero_first": "heterophilic first (oracle)", "random": "random", "homo_first": "homophilic first (inverse)"}


def agg(df, metric):
    g = df.groupby(["dataset", "model", "strategy", "sparsity"])[metric]
    return (g.mean() * 100).rename("mean").to_frame().join((g.std() * 100).rename("std"))


def operating_points(a, ds, m):
    full = a.loc[(ds, m, "random", 0.0)]
    het = a.loc[(ds, m, "hetero_first")]
    peak_s = float(het["mean"].idxmax())
    pick = lambda st, s: a.loc[(ds, m, st, s)]  # noqa: E731
    return dict(full=full["mean"], full_sd=full["std"], peak=het["mean"].max(), peak_s=peak_s,
                rand95=pick("random", 0.95)["mean"], orac95=pick("hetero_first", 0.95)["mean"],
                inv95=pick("homo_first", 0.95)["mean"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--metric", default="test_final", choices=["test_final", "test_bestval"])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    df = pd.concat([pd.read_csv(f) for f in args.files]).drop_duplicates(
        ["dataset", "model", "strategy", "sparsity", "seed"])
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- 1. reproduction check (both metrics) ----------
    print("1. REPRODUCTION CHECK vs the paper's Table 1 (GCN/SAGE). Pass = every operating point within 2.0 points.\n")
    for metric in ("test_final", "test_bestval"):
        a = agg(df, metric)
        worst = 0.0
        lines = []
        for (ds, m), ref in PAPER.items():
            try:
                op = operating_points(a, ds, m)
            except KeyError:
                continue
            ours = (op["full"], op["peak"], op["peak_s"], op["rand95"], op["orac95"], op["inv95"])
            diffs = [o - r for i, (o, r) in enumerate(zip(ours, ref)) if i != 2 and not np.isnan(o)]
            worst = max(worst, max(abs(d) for d in diffs))
            lines.append(f"   {DSN[ds]:12s} {NAMES[m]:9s} full {ours[0]:5.1f} ({ref[0]})  peak {ours[1]:5.1f}@{ours[2]:.2f} "
                         f"({ref[1]}@{ref[2]:.2f})  rand.95 {ours[3]:5.1f} ({ref[3]})  orac.95 {ours[4]:5.1f} ({ref[4]})  "
                         f"inv.95 {ours[5]:5.1f} ({ref[5]})")
        print(f"  metric = {metric}: largest deviation {worst:.1f} points -> {'PASS' if worst <= 2.0 else 'CHECK'}")
        print("\n".join(lines) + "\n")

    a = agg(df, args.metric)
    print(f"(primary metric for everything below: {args.metric})\n")

    # ---------- 2. extended Table 1 ----------
    rows = []
    for ds in ("cora", "roman"):
        for m in ("gcn", "sage", "gat", "gatsep"):
            try:
                op = operating_points(a, ds, m)
            except KeyError:
                continue
            rows.append({"Dataset": DSN[ds], "Model": NAMES[m], "Full graph": f"{op['full']:.1f} ± {op['full_sd']:.1f}",
                         "Oracle peak (s)": f"{op['peak']:.1f} ({op['peak_s']:.2f})", "Random @0.95": f"{op['rand95']:.1f}",
                         "Oracle @0.95": f"{op['orac95']:.1f}", "Inverse @0.95": f"{op['inv95']:.1f}"})
    t1 = pd.DataFrame(rows)
    t1.to_csv(f"{args.outdir}/table1_extended.csv", index=False)
    print("2. EXTENDED TABLE 1 (test accuracy %, mean ± std over seeds)\n")
    print(t1.to_string(index=False) + "\n")

    # ---------- 3. prediction checks (written before the run; see README) ----------
    print("3. PREDICTION CHECKS")
    d = df.copy()
    d["acc"] = d[args.metric] * 100
    for ds in ("cora", "roman"):
        for m in ("gcn", "sage", "gat", "gatsep"):
            sub = d[(d.dataset == ds) & (d.model == m)]
            if sub.empty:
                continue
            full = sub[(sub.strategy == "random") & (sub.sparsity == 0.0)].set_index("seed").acc
            r95 = sub[(sub.strategy == "random") & (sub.sparsity == 0.95)].set_index("seed").acc
            delta = (r95 - full).dropna()
            print(f"   {DSN[ds]:12s} {NAMES[m]:9s} random removal of 95% of edges: {delta.mean():+6.1f} points "
                  f"(seeds positive {int((delta > 0).sum())}/{len(delta)})  -> edges are "
                  f"{'NOISE (removal helps)' if delta.mean() > 0 and (delta > 0).all() else 'SIGNAL (removal hurts)' if delta.mean() < 0 and (delta < 0).all() else 'mixed'}")
    print("   Prediction P2: on Roman-empire, GAT behaves like GCN (noise) and GAT-sep like GraphSAGE (signal).")
    print("   Prediction P3: on Cora, all four models have heterophilic-first > random > homophilic-first and a peak above the full graph.")
    hh = d.groupby(["dataset"]).apply(lambda g: 1 - g[(g.strategy == "random") & (g.sparsity == 0.0)].homophily_kept.mean())
    print(f"   1 - h: Cora {hh.get('cora', np.nan):.3f}, Roman-empire {hh.get('roman', np.nan):.3f} (P4: oracle peak at s ~ 1 - h)\n")

    # ---------- 4. extended Figure 1 ----------
    models = [m for m in ("gcn", "sage", "gat", "gatsep") if m in set(df.model)]
    fig, axes = plt.subplots(len(models), 2, figsize=(9, 2.6 * len(models)), squeeze=False, sharex=True)
    for i, m in enumerate(models):
        for j, ds in enumerate(("cora", "roman")):
            ax = axes[i, j]
            for st in ("hetero_first", "random", "homo_first"):
                try:
                    s = a.loc[(ds, m, st)].reset_index()
                except KeyError:
                    continue
                ax.plot(s.sparsity, s["mean"], color=COL[st], lw=2, label=LAB[st])
                ax.fill_between(s.sparsity, s["mean"] - s["std"], s["mean"] + s["std"], color=COL[st], alpha=0.15, lw=0)
            ax.set_title(f"{NAMES[m]} · {DSN[ds]}", fontsize=9, loc="left")
            ax.grid(color="#e4e3df", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8, colors="#52514e")
    for ax in axes[-1]:
        ax.set_xlabel("fraction of edges removed", fontsize=9, color="#52514e")
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy (%)", fontsize=9, color="#52514e")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{args.outdir}/figure1_extended.png", dpi=170)
    print(f"saved {args.outdir}/table1_extended.csv and {args.outdir}/figure1_extended.png")


if __name__ == "__main__":
    main()
