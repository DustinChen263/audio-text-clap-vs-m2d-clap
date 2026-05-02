"""Shared analysis helpers: cosine similarity matrix, retrieval metrics, and
plotting utilities used by both the per-model notebooks and the comparison
notebook.

A "result bundle" is a Python dict:
    {
        "name": str,                        e.g. "CLAP" or "M2D-CLAP"
        "audio_emb": np.ndarray (N, D),
        "text_emb":  np.ndarray (N, D),
        "captions":  list[str] (length N),
        "labels":    Optional[list[str]],   per-sample class label, used by zero-shot
        "sim":       np.ndarray (N, N),     cosine similarity matrix
        "paired":    np.ndarray (N,),       diagonal of `sim`
        "metrics":   dict,                  retrieval metrics (a2t, t2a)
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_sim_matrix(audio_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """Returns S where S[i, j] = cos(audio_i, text_j)."""
    a = l2_normalize(audio_emb)
    t = l2_normalize(text_emb)
    return a @ t.T


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def _ranks_along_rows(sim: np.ndarray) -> np.ndarray:
    """Rank of the diagonal entry within each row (0-indexed: 0 = top-1)."""
    diag = np.diag(sim)[:, None]
    return (sim > diag).sum(axis=1)  # # of strictly greater entries = rank


def retrieval_metrics(sim: np.ndarray) -> dict[str, float]:
    """Audio -> Text retrieval metrics for a (N x N) similarity matrix where
    row i is the query and column i is its ground-truth match."""
    n = sim.shape[0]
    ranks = _ranks_along_rows(sim)
    return {
        "R@1": float(np.mean(ranks < 1)),
        "R@5": float(np.mean(ranks < 5)),
        "R@10": float(np.mean(ranks < 10)),
        "MedR": float(np.median(ranks) + 1),  # 1-indexed median rank
        "MeanR": float(np.mean(ranks) + 1),
        "N": int(n),
    }


def both_directions(sim: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        "a2t": retrieval_metrics(sim),
        "t2a": retrieval_metrics(sim.T),
    }


def class_prompt_retrieval(
    audio_emb: np.ndarray,
    class_text_emb: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Retrieval against C unique class prompts.

    audio_emb       (N, D)
    class_text_emb  (C, D)   one prompt per unique class label
    targets         (N,)     int target class id per audio

    This avoids the ties problem you get when running A->T retrieval against
    a gallery of N captions where many captions are identical (e.g. ESC-50
    has 40 audios sharing one prompt).
    """
    a = l2_normalize(audio_emb)
    c = l2_normalize(class_text_emb)
    sim = a @ c.T                       # (N, C)
    diag = sim[np.arange(len(targets)), targets][:, None]
    ranks = (sim > diag).sum(axis=1)    # 0-indexed rank
    return {
        "R@1": float(np.mean(ranks < 1)),
        "R@5": float(np.mean(ranks < 5)),
        "R@10": float(np.mean(ranks < 10)),
        "MedR": float(np.median(ranks) + 1),
        "MeanR": float(np.mean(ranks) + 1),
        "N_query": int(len(targets)),
        "N_gallery": int(class_text_emb.shape[0]),
    }


def class_prompt_retrieval_text2audio(
    audio_emb: np.ndarray,
    class_text_emb: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Retrieval in the opposite direction: each class prompt is a query, and
    we ask whether any of the K=40 in-class audios appears in the top-K
    retrievals from the audio gallery (multi-positive R@K)."""
    a = l2_normalize(audio_emb)
    c = l2_normalize(class_text_emb)
    sim = c @ a.T                       # (C, N)
    n_audio = audio_emb.shape[0]
    n_class = class_text_emb.shape[0]
    metrics = {f"R@{k}": [] for k in (1, 5, 10)}
    medians = []
    for cls in range(n_class):
        scores = sim[cls]
        order = np.argsort(-scores)
        positive_mask = (targets == cls)
        ranks_of_positives = np.where(positive_mask[order])[0]
        if len(ranks_of_positives) == 0:
            continue
        # rank of the first positive (best in-class audio) — easiest case
        first_rank = ranks_of_positives.min()
        for k in (1, 5, 10):
            metrics[f"R@{k}"].append(int(first_rank < k))
        medians.append(int(first_rank))
    return {
        "R@1": float(np.mean(metrics["R@1"])),
        "R@5": float(np.mean(metrics["R@5"])),
        "R@10": float(np.mean(metrics["R@10"])),
        "MedR": float(np.median(medians) + 1),
        "MeanR": float(np.mean(medians) + 1),
        "N_query": int(n_class),
        "N_gallery": int(n_audio),
    }


# ---------------------------------------------------------------------------
# Bundle assembly
# ---------------------------------------------------------------------------


def build_bundle(
    name: str,
    audio_emb: np.ndarray,
    text_emb: np.ndarray,
    captions: list[str],
    labels: Optional[list[str]] = None,
) -> dict:
    sim = cosine_sim_matrix(audio_emb, text_emb)
    paired = np.diag(sim).copy()
    return {
        "name": name,
        "audio_emb": audio_emb,
        "text_emb": text_emb,
        "captions": captions,
        "labels": labels,
        "sim": sim,
        "paired": paired,
        "metrics": both_directions(sim),
    }


def load_bundle(
    name: str,
    results_dir: Path | str,
) -> dict:
    """Load embeddings + metadata produced by extract_clap.py / extract_m2d_clap.py."""
    rd = Path(results_dir)
    audio = np.load(rd / "audio_embeddings.npy")
    text = np.load(rd / "text_embeddings.npy")
    meta = pd.read_csv(rd / "metadata.csv")
    captions = meta["caption"].tolist()
    labels = meta["class_name"].tolist() if "class_name" in meta.columns else None
    return build_bundle(name, audio, text, captions, labels)


# ---------------------------------------------------------------------------
# Zero-shot classification (for label-bearing datasets like ESC-50)
# ---------------------------------------------------------------------------


def zero_shot_accuracy(
    audio_emb: np.ndarray,
    class_text_emb: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Argmax cosine similarity over class prompts.

    audio_emb        (N, D)
    class_text_emb   (C, D)  e.g. C=50 for ESC-50
    targets          (N,)    integer class labels
    """
    a = l2_normalize(audio_emb)
    c = l2_normalize(class_text_emb)
    logits = a @ c.T          # (N, C)
    top1 = logits.argmax(axis=1)
    top5 = np.argsort(-logits, axis=1)[:, :5]
    return {
        "top1": float(np.mean(top1 == targets)),
        "top5": float(np.mean([t in row for t, row in zip(targets, top5)])),
    }


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def metrics_table(bundles: list[dict]) -> pd.DataFrame:
    """Build a tidy metrics dataframe across multiple result bundles."""
    rows = []
    for b in bundles:
        for direction, m in b["metrics"].items():
            rows.append(
                {
                    "model": b["name"],
                    "direction": direction,
                    **m,
                    "paired_sim_mean": float(np.mean(b["paired"])),
                    "paired_sim_std": float(np.std(b["paired"])),
                }
            )
    return pd.DataFrame(rows)


def print_metrics(bundles: list[dict]) -> None:
    df = metrics_table(bundles)
    cols = ["model", "direction", "R@1", "R@5", "R@10", "MedR", "MeanR", "N"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
