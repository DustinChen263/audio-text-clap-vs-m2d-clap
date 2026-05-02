"""Run the full CLAP vs M2D-CLAP comparison and dump every figure / table to
`results/figures/`. This mirrors notebooks/03_compare.ipynb cell-for-cell so the
notebook can be re-executed by anyone, but it can also be run headless from the
command line:

    python src/run_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="notebook", style="whitegrid")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from analyze import (  # noqa: E402
    class_prompt_retrieval,
    class_prompt_retrieval_text2audio,
    cosine_sim_matrix,
    l2_normalize,
    load_bundle,
    metrics_table,
    print_metrics,
    zero_shot_accuracy,
)

RESULTS = PROJECT_DIR / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    clap = load_bundle("CLAP", RESULTS / "clap")
    m2d = load_bundle("M2D-CLAP", RESULTS / "m2d_clap")

    for b in [clap, m2d]:
        print(
            f"{b['name']:<10}  audio {b['audio_emb'].shape}  text {b['text_emb'].shape}"
        )
    assert clap["audio_emb"].shape[0] == m2d["audio_emb"].shape[0]
    assert clap["captions"] == m2d["captions"], "captions must align"
    N = clap["audio_emb"].shape[0]
    print("aligned samples:", N)

    # ----- Retrieval against 50 unique class prompts (the apples-to-apples
    # version on ESC-50 — see README for why we don't use 2000-vs-2000 here).
    def labels_to_targets(b):
        rd = RESULTS / ("clap" if b["name"] == "CLAP" else "m2d_clap")
        class_names = (rd / "class_names.txt").read_text().splitlines()
        cemb = np.load(rd / "class_text_embeddings.npy")
        name2idx = {n: i for i, n in enumerate(class_names)}
        targets = np.array([name2idx[c] for c in b["labels"]])
        return targets, cemb, class_names

    rows = []
    for b in [clap, m2d]:
        targets, cemb, _ = labels_to_targets(b)
        a2t = class_prompt_retrieval(b["audio_emb"], cemb, targets)
        t2a = class_prompt_retrieval_text2audio(b["audio_emb"], cemb, targets)
        for direction, m in [("a2t", a2t), ("t2a", t2a)]:
            rows.append({"model": b["name"], "direction": direction, **m,
                         "paired_sim_mean": float(np.mean(b["paired"])),
                         "paired_sim_std":  float(np.std(b["paired"]))})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(FIGS / "retrieval_metrics.csv", index=False)
    print("\n=== Retrieval metrics (audio<->50 class prompts) ===")
    print(tbl[["model","direction","R@1","R@5","R@10","MedR","MeanR"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ----- Paired similarity, with negatives restricted to *cross-class* pairs.
    # If we used the full off-diagonal we'd also count same-class identical
    # captions as "negatives", which under-estimates the gap between models.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, b in zip(axes, [clap, m2d]):
        sim = b["sim"]
        n = sim.shape[0]
        # build a same-class mask
        labels_arr = np.array(b["labels"])
        same_class = labels_arr[:, None] == labels_arr[None, :]
        diag = np.eye(n, dtype=bool)
        cross_class_neg = sim[~same_class]                # negatives = different class
        in_class_pos    = sim[same_class & ~diag]         # in-class but off-diagonal
        pair_pos        = b["paired"]                     # the literal diagonal
        ax.hist(cross_class_neg, bins=80, alpha=0.55, density=True,
                color="steelblue", label="cross-class negatives")
        ax.hist(in_class_pos, bins=40, alpha=0.55, density=True,
                color="seagreen",  label="in-class positives (off-diag)")
        ax.hist(pair_pos, bins=40, alpha=0.85, density=True,
                color="tomato", label="paired positives (diag)")
        gap = pair_pos.mean() - cross_class_neg.mean()
        ax.axvline(pair_pos.mean(),       color="tomato",    linestyle="--", linewidth=1)
        ax.axvline(cross_class_neg.mean(),color="steelblue", linestyle="--", linewidth=1)
        ax.set_title(
            f"{b['name']}   pos μ={pair_pos.mean():.3f},  neg μ={cross_class_neg.mean():.3f},  gap={gap:.3f}"
        )
        ax.set_xlabel("cosine similarity")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    plt.tight_layout()
    plt.savefig(FIGS / "paired_similarity_distribution.png", dpi=150)
    plt.close(fig)

    # ----- Retrieval bar plot (50-prompt version).
    long = tbl.melt(
        id_vars=["model", "direction"],
        value_vars=["R@1", "R@5", "R@10"],
        var_name="metric",
        value_name="score",
    )
    g = sns.catplot(
        data=long, kind="bar",
        x="metric", y="score", hue="model", col="direction",
        height=4, aspect=1.1, palette=["#4C72B0", "#DD8452"],
    )
    g.set_axis_labels("", "Recall@K")
    g.set(ylim=(0, 1))
    for ax in g.axes.flat:
        for p in ax.patches:
            h = p.get_height()
            if h <= 0:
                continue
            ax.annotate(f"{h:.3f}",
                        (p.get_x() + p.get_width() / 2, h + 0.01),
                        ha="center", fontsize=9)
    g.figure.suptitle(
        "Retrieval against 50 ESC-50 class prompts (a2t = audio→prompt, t2a = prompt→audio)",
        y=1.04,
    )
    g.figure.savefig(FIGS / "retrieval_bars.png", dpi=150, bbox_inches="tight")
    plt.close(g.figure)

    # 5. heatmap
    K = 50
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    vmin = min(clap["sim"][:K, :K].min(), m2d["sim"][:K, :K].min())
    vmax = max(clap["sim"][:K, :K].max(), m2d["sim"][:K, :K].max())
    for ax, b in zip(axes, [clap, m2d]):
        im = ax.imshow(
            b["sim"][:K, :K], cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal"
        )
        ax.set_title(f"{b['name']} similarity matrix (first {K})")
        ax.set_xlabel("text idx")
        ax.set_ylabel("audio idx")
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle("Diagonal prominence ↔ better audio-text alignment", y=1.02)
    plt.savefig(FIGS / "similarity_matrix_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6. zero-shot accuracy
    def zs_for(b):
        rd = RESULTS / ("clap" if b["name"] == "CLAP" else "m2d_clap")
        cemb = np.load(rd / "class_text_embeddings.npy")
        class_names = (rd / "class_names.txt").read_text().splitlines()
        name2idx = {n: i for i, n in enumerate(class_names)}
        targets = np.array([name2idx[c] for c in b["labels"]])
        return zero_shot_accuracy(b["audio_emb"], cemb, targets), class_names, targets

    (zs_clap, class_names_clap, targets_clap) = zs_for(clap)
    (zs_m2d, class_names_m2d, targets_m2d) = zs_for(m2d)

    zs_table = pd.DataFrame(
        {
            "model": ["CLAP", "M2D-CLAP"],
            "top1_acc": [zs_clap["top1"], zs_m2d["top1"]],
            "top5_acc": [zs_clap["top5"], zs_m2d["top5"]],
        }
    )
    zs_table.to_csv(FIGS / "zero_shot_accuracy.csv", index=False)
    print("\n=== Zero-shot ESC-50 accuracy ===")
    print(zs_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(2)
    ax.bar(x - 0.18, zs_table["top1_acc"], width=0.35, label="top-1", color="#4C72B0")
    ax.bar(x + 0.18, zs_table["top5_acc"], width=0.35, label="top-5", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(zs_table["model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("ESC-50 zero-shot classification")
    for i, (t1, t5) in enumerate(zip(zs_table["top1_acc"], zs_table["top5_acc"])):
        ax.annotate(f"{t1:.3f}", (i - 0.18, t1 + 0.01), ha="center", fontsize=9)
        ax.annotate(f"{t5:.3f}", (i + 0.18, t5 + 0.01), ha="center", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "zero_shot_bars.png", dpi=150)
    plt.close(fig)

    # 7. per-class top-1
    def per_class_top1(b):
        rd = RESULTS / ("clap" if b["name"] == "CLAP" else "m2d_clap")
        cemb = np.load(rd / "class_text_embeddings.npy")
        class_names = (rd / "class_names.txt").read_text().splitlines()
        name2idx = {n: i for i, n in enumerate(class_names)}
        targets = np.array([name2idx[c] for c in b["labels"]])
        logits = l2_normalize(b["audio_emb"]) @ l2_normalize(cemb).T
        pred = logits.argmax(axis=1)
        acc = pd.DataFrame(
            {
                "class": [class_names[t] for t in targets],
                "correct": (pred == targets).astype(int),
            }
        )
        return acc.groupby("class")["correct"].mean()

    pc_clap = per_class_top1(clap).rename("CLAP")
    pc_m2d = per_class_top1(m2d).rename("M2D-CLAP")
    pc = pd.concat([pc_clap, pc_m2d], axis=1).fillna(0)
    pc["delta"] = pc["M2D-CLAP"] - pc["CLAP"]
    pc_sorted = pc.sort_values("delta", ascending=False)
    pc_sorted.to_csv(FIGS / "per_class_top1.csv")

    fig, ax = plt.subplots(figsize=(10, 12))
    pc_sorted[["CLAP", "M2D-CLAP"]].plot.barh(
        ax=ax, color=["#4C72B0", "#DD8452"]
    )
    ax.set_title("Per-class zero-shot top-1 accuracy (sorted by M2D-CLAP gain)")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("top-1 accuracy")
    plt.tight_layout()
    plt.savefig(FIGS / "per_class_top1.png", dpi=150)
    plt.close(fig)

    summary_lines = [f"Dataset: ESC-50, N = {N}, classes = 50"]
    for name in ["CLAP", "M2D-CLAP"]:
        sub = tbl[tbl["model"] == name]
        a2t = sub[sub["direction"] == "a2t"].iloc[0]
        t2a = sub[sub["direction"] == "t2a"].iloc[0]
        summary_lines.append(
            f"{name:<9} | "
            f"A→prompts  R@1={a2t['R@1']:.3f}  R@5={a2t['R@5']:.3f}  MedR={a2t['MedR']:.0f}  | "
            f"prompts→A  R@1={t2a['R@1']:.3f}  R@5={t2a['R@5']:.3f}  MedR={t2a['MedR']:.0f}"
        )
    summary_lines.append(
        f"Zero-shot ESC-50  | "
        f"CLAP     top-1={zs_clap['top1']:.4f}, top-5={zs_clap['top5']:.4f}  | "
        f"M2D-CLAP top-1={zs_m2d['top1']:.4f}, top-5={zs_m2d['top5']:.4f}"
    )
    for b in [clap, m2d]:
        labels_arr = np.array(b["labels"])
        n = b["sim"].shape[0]
        same = labels_arr[:, None] == labels_arr[None, :]
        cross_neg_mean = b["sim"][~same].mean()
        summary_lines.append(
            f"{b['name']:<9} paired sim μ={b['paired'].mean():.3f}, cross-class neg μ={cross_neg_mean:.3f}, "
            f"gap={b['paired'].mean()-cross_neg_mean:.3f}"
        )
    summary = "\n".join(summary_lines)
    (FIGS / "summary.txt").write_text(summary)
    print("\n=== Summary ===")
    print(summary)
    print(f"\nFigures + tables saved to {FIGS}")


if __name__ == "__main__":
    main()
