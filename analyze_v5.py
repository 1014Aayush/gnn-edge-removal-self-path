"""v5 analysis: checks predictions P5-P9 (README) and the best-validation metric (E5).

    python analyze_v5.py prior/part0.csv prior/part1.csv results/v5_gpu0.csv results/v5_gpu1.csv --outdir results/v5

"Effect" = random removal of 95% of edges minus the full graph (random, s = 0), in points.
Split 0 has 5 seeds (seeds 0-4); the 10-split analysis uses seed = split (Platonov et al.).
Works on partial data: sections whose rows are missing are skipped.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

NAMES = {"gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT", "gatsep": "GAT-sep", "gcnres": "GCN-res", "mlp": "MLP"}
SELF_PATH = {"gcn": False, "gat": False, "sage": True, "gatsep": True, "gcnres": True}
DSN = {"cora": "Cora", "roman": "Roman-empire", "amazon": "Amazon-ratings", "tolokers": "Tolokers"}
COL = {"hetero_first": "#2a78d6", "random": "#eb6834", "homo_first": "#1baf7a"}
LAB = {"hetero_first": "heterophilic first (oracle)", "random": "random", "homo_first": "homophilic first (inverse)"}
GRAPH = ["gcn", "sage", "gat", "gatsep", "gcnres"]
verdicts = []


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.mean(), np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def verdict(name, ok, detail):
    verdicts.append((name, "PASS" if ok else "FAIL", detail))
    print(f"   {name}: {'PASS' if ok else 'FAIL'} - {detail}")


def load(files):
    df = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)], ignore_index=True)
    df["split"] = df.get("split", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["metric"] = df.get("metric", pd.Series("acc", index=df.index)).fillna("acc")
    df = df.drop_duplicates(["dataset", "split", "model", "strategy", "sparsity", "seed"])
    for c in ("test_final", "test_bestval"):
        df[c] = df[c] * 100
    return df


def per_seed(df, ds, m, metric, rows):
    """Full-graph score and random-95% effect for each (split, seed) row group."""
    sub = df[(df.dataset == ds) & (df.model == m)]
    sub = sub[rows(sub)]
    idx = ["split", "seed"]
    if m == "mlp":
        full = sub[sub.strategy == "none"].set_index(idx)[metric]
        return pd.DataFrame({"full": full, "effect": np.nan})
    full = sub[(sub.strategy == "random") & (sub.sparsity == 0.0)].set_index(idx)[metric]
    r95 = sub[(sub.strategy == "random") & (sub.sparsity == 0.95)].set_index(idx)[metric]
    return pd.DataFrame({"full": full, "effect": r95 - full}).dropna(subset=["full"])


split0 = lambda s: s.split == 0  # noqa: E731
paired = lambda s: s.seed == s.split  # noqa: E731


def effect_table(df, metric, rows, datasets):
    out = []
    for ds in datasets:
        for m in NAMES:
            t = per_seed(df, ds, m, metric, rows)
            if t.empty:
                continue
            e = t.effect.dropna()
            fm, flo, fhi = ci(t.full)
            row = {"Dataset": DSN[ds], "Model": NAMES[m], "n": len(t), "Full": round(fm, 1),
                   "Full 95% CI": f"[{flo:.1f}, {fhi:.1f}]"}
            if len(e):
                em, lo, hi = ci(e)
                row.update({"Effect": round(em, 1), "Effect 95% CI": f"[{lo:+.1f}, {hi:+.1f}]",
                            "n > 0": f"{int((e > 0).sum())}/{len(e)}"})
            out.append(row)
    return pd.DataFrame(out)


def curves(df, ds, m, metric, rows):
    sub = df[(df.dataset == ds) & (df.model == m) & (df.strategy != "none")]
    sub = sub[rows(sub)]
    g = sub.groupby(["strategy", "sparsity"])[metric]
    return pd.DataFrame({"mean": g.mean(), "std": g.std()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--outdir", default="results/v5")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = load(args.files)
    have = lambda ds, m=None: ((df.dataset == ds) & ((df.model == m) if m else True)).any()  # noqa: E731

    print("0. ROWS PER DATASET x MODEL x SPLIT-COUNT")
    print(df.groupby(["dataset", "model"]).agg(rows=("seed", "size"), splits=("split", "nunique")).to_string() + "\n")

    # ---------- E5: best-validation metric ----------
    print("E5. BEST-VALIDATION CHECK (split 0, 5 seeds): effect under both metrics")
    a = effect_table(df, "test_final", split0, ["cora", "roman"])
    b = effect_table(df, "test_bestval", split0, ["cora", "roman"])
    if not a.empty and "Effect" in a:
        e5 = a[["Dataset", "Model", "Effect", "n > 0"]].merge(
            b[["Dataset", "Model", "Effect", "Effect 95% CI", "n > 0"]], on=["Dataset", "Model"],
            suffixes=(" (final epoch)", " (best val)")).dropna()
        e5.to_csv(f"{args.outdir}/e5_bestval_check.csv", index=False)
        print(e5.to_string(index=False))
        same = (np.sign(e5["Effect (final epoch)"]) == np.sign(e5["Effect (best val)"])).all()
        verdict("E5", same, "every effect keeps its sign under the best-validation metric" if same
                else "some effect changes sign under the best-validation metric")
    print()

    # ---------- P5: GCN-res ----------
    t = effect_table(df, "test_final", split0, ["cora", "roman"])
    t.to_csv(f"{args.outdir}/table_split0_all_models.csv", index=False)
    print("SPLIT-0 TABLE (5 seeds, final-epoch metric)\n" + t.to_string(index=False) + "\n")
    if have("roman", "gcnres"):
        print("P5. GCN-res behaves like GraphSAGE")
        g = lambda ds, m: t[(t.Dataset == DSN[ds]) & (t.Model == NAMES[m])].iloc[0]  # noqa: E731
        r, s = g("roman", "gcnres"), g("roman", "sage")
        e = per_seed(df, "roman", "gcnres", "test_final", split0).effect
        ok = abs(r.Full - s.Full) <= 3 and (e < 0).all()
        verdict("P5 Roman", ok, f"GCN-res full {r.Full} vs SAGE {s.Full}; effect {r.Effect} ({int((e < 0).sum())}/{len(e)} seeds < 0)")
        if have("cora", "gcnres"):
            c, c0 = g("cora", "gcnres"), g("cora", "gcn")
            verdict("P5 Cora", abs(c.Full - c0.Full) <= 3, f"GCN-res full {c.Full} vs GCN {c0.Full}")
        print()

    # ---------- P6: MLP ----------
    if have("roman", "mlp"):
        print("P6. Own MLP baseline")
        mlp = t[(t.Dataset == "Roman-empire") & (t.Model == "MLP")].Full.iloc[0]
        floor = curves(df, "roman", "sage", "test_final", split0).loc[("random", 0.95), "mean"]
        floor2 = curves(df, "roman", "gatsep", "test_final", split0).loc[("random", 0.95), "mean"]
        gcn = t[(t.Dataset == "Roman-empire") & (t.Model == "GCN")].Full.iloc[0]
        verdict("P6 Roman", abs(mlp - floor) <= 3 and abs(mlp - floor2) <= 3 and mlp > gcn + 10,
                f"MLP {mlp:.1f}; SAGE/GAT-sep random@.95 {floor:.1f}/{floor2:.1f}; GCN full {gcn:.1f}")
        if have("cora", "mlp"):
            c = t[t.Dataset == "Cora"].set_index("Model").Full
            verdict("P6 Cora", c["MLP"] < c.drop("MLP").min(), f"MLP {c['MLP']:.1f}; graph models {c.drop('MLP').min():.1f}+")
        print()

    # ---------- P7: 10 Roman splits ----------
    multi = [ds for ds in ("roman", "amazon", "tolokers") if df[(df.dataset == ds)].split.nunique() > 1]
    tm = effect_table(df, "test_final", paired, multi)
    tmb = effect_table(df, "test_bestval", paired, multi)
    if not tm.empty:
        tm.to_csv(f"{args.outdir}/table_10splits.csv", index=False)
        tmb.to_csv(f"{args.outdir}/table_10splits_bestval.csv", index=False)
        print("10-SPLIT TABLE (seed = split; final-epoch metric; Tolokers = ROC-AUC)\n" + tm.to_string(index=False))
        print("\n   same, best-validation metric\n" + tmb.to_string(index=False) + "\n")
    if "roman" in multi:
        print("P7. Roman-empire: effect sign stable across splits")
        for m in GRAPH:
            e = per_seed(df, "roman", m, "test_final", paired).effect.dropna()
            if e.empty:
                continue
            want_pos = not SELF_PATH[m]
            k = int((e > 0).sum()) if want_pos else int((e < 0).sum())
            verdict(f"P7 {NAMES[m]}", k >= len(e) - 1, f"{'positive' if want_pos else 'negative'} in {k}/{len(e)} splits, mean {e.mean():+.1f}")
        print()

    # ---------- P8 / P9: medium homophily ----------
    peaks = []
    for ds in ("amazon", "tolokers"):
        if ds not in multi:
            continue
        sub = df[(df.dataset == ds) & (df.strategy == "random") & (df.sparsity == 0.0)]
        one_minus_h = 1 - sub.homophily_kept.mean()
        print(f"P8/P9. {DSN[ds]}: 1 - h = {one_minus_h:.3f}")
        eff = {m: per_seed(df, ds, m, "test_final", paired).effect.dropna().mean() for m in GRAPH}
        for a_, b_ in (("gcn", "gcnres"), ("gcn", "sage"), ("gat", "gatsep")):
            verdict(f"P8 {DSN[ds]} {NAMES[a_]} > {NAMES[b_]}", eff[a_] > eff[b_], f"{eff[a_]:+.1f} vs {eff[b_]:+.1f}")
        for m in GRAPH:
            c = curves(df, ds, m, "test_final", paired)
            if ("hetero_first", 0.0) not in c.index:
                continue
            het = c.loc["hetero_first", "mean"]
            s_peak = float(het.idxmax())
            peaks.append({"Dataset": DSN[ds], "Model": NAMES[m], "1 - h": round(one_minus_h, 3), "Oracle peak s": s_peak,
                          "Peak": round(het.max(), 1), "Full": round(c.loc[("random", 0.0), "mean"], 1)})
            if not SELF_PATH[m]:
                verdict(f"P9 {DSN[ds]} {NAMES[m]}", abs(s_peak - one_minus_h) <= 0.10, f"peak at s = {s_peak:.2f}")
        print()
    if peaks:
        pd.DataFrame(peaks).to_csv(f"{args.outdir}/p9_oracle_peaks.csv", index=False)
        print(pd.DataFrame(peaks).to_string(index=False) + "\n")

    # ---------- figures ----------
    def grid(datasets, rows, fname, title):
        models = [m for m in GRAPH if any(have(ds, m) for ds in datasets)]
        if not models:
            return
        fig, axes = plt.subplots(len(models), len(datasets), figsize=(4.5 * len(datasets), 2.4 * len(models)),
                                 squeeze=False, sharex=True)
        for i, m in enumerate(models):
            for j, ds in enumerate(datasets):
                ax = axes[i, j]
                c = curves(df, ds, m, "test_final", rows)
                for st in ("hetero_first", "random", "homo_first"):
                    if st in c.index.get_level_values(0):
                        s = c.loc[st].reset_index()
                        ax.plot(s.sparsity, s["mean"], color=COL[st], lw=2, label=LAB[st])
                        ax.fill_between(s.sparsity, s["mean"] - s["std"], s["mean"] + s["std"], color=COL[st], alpha=0.15, lw=0)
                mlp = per_seed(df, ds, "mlp", "test_final", rows).full
                if len(mlp):
                    ax.axhline(mlp.mean(), color="#8a8985", ls=":", lw=1.5, label="MLP (no graph)")
                h0 = df[(df.dataset == ds) & (df.strategy == "random") & (df.sparsity == 0)].homophily_kept.mean()
                ax.axvline(1 - h0, color="#c9c8c4", lw=1)
                ylab = "ROC-AUC" if (df[df.dataset == ds].metric == "auc").any() else "accuracy"
                ax.set_title(f"{NAMES[m]} · {DSN[ds]} ({ylab})", fontsize=9, loc="left")
                ax.grid(color="#e4e3df", lw=0.6)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(labelsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("fraction of edges removed (grey line: 1 − h)", fontsize=9)
        hl = axes[0, 0].get_legend_handles_labels()
        fig.legend(*hl, loc="upper center", ncol=4, frameon=False, fontsize=8)
        fig.suptitle(title, y=0.995, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(f"{args.outdir}/{fname}", dpi=160)
        plt.close(fig)
        print(f"saved {args.outdir}/{fname}")

    grid([ds for ds in ("cora", "roman") if have(ds)], split0, "fig_split0_6models.png", "split 0, 5 seeds")
    grid(multi, paired, "fig_10splits.png", "10 splits, seed = split")

    if verdicts:
        v = pd.DataFrame(verdicts, columns=["Check", "Result", "Detail"])
        v.to_csv(f"{args.outdir}/prediction_verdicts.csv", index=False)
        print("\nVERDICTS\n" + v.to_string(index=False))


if __name__ == "__main__":
    main()
