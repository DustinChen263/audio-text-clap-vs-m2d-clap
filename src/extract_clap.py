"""Extract LAION-CLAP audio + text embeddings for a (audio_path, caption) CSV.

Outputs:
    results/clap/audio_embeddings.npy   (N, 512) float32
    results/clap/text_embeddings.npy    (N, 512) float32
    results/clap/metadata.csv           a copy of the (filtered) input rows

Usage:
    # Default: ESC-50 metadata
    python src/extract_clap.py

    # Or point at any compatible CSV
    python src/extract_clap.py --csv data/esc50_metadata.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_DIR / "data" / "esc50_metadata.csv"
RESULTS_DIR = PROJECT_DIR / "results" / "clap"


def filter_corrupt_files(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose audio is missing / unreadable / empty."""
    keep = []
    for path in df["audio_path"]:
        try:
            data, _ = sf.read(path)
            keep.append(len(data) > 0)
        except Exception:
            keep.append(False)
    n_drop = len(df) - sum(keep)
    if n_drop:
        print(f"[..] Dropping {n_drop} corrupt/missing audio rows")
    return df[keep].reset_index(drop=True)


def encode(csv_path: Path, results_dir: Path, batch_size: int = 32) -> None:
    import laion_clap

    df = pd.read_csv(csv_path)
    print(f"[..] Loaded CSV: {csv_path}  rows={len(df)}")
    df = filter_corrupt_files(df)

    audio_paths = df["audio_path"].tolist()
    captions = df["caption"].tolist()
    print(f"[..] Effective sample count: {len(audio_paths)}")

    print("[..] Loading LAION-CLAP (HTSAT-tiny + RoBERTa, enable_fusion=False)...")
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()  # auto-download default checkpoint

    print(f"[..] Extracting audio embeddings (batch_size={batch_size}) ...")
    audio_chunks: list[np.ndarray] = []
    for s in tqdm(range(0, len(audio_paths), batch_size), desc="audio"):
        chunk_paths = audio_paths[s : s + batch_size]
        emb = model.get_audio_embedding_from_filelist(
            x=chunk_paths, use_tensor=False
        )
        audio_chunks.append(np.asarray(emb))
    audio_emb = np.concatenate(audio_chunks, axis=0)

    print(f"[..] Extracting text embeddings (batch_size={batch_size}) ...")
    text_chunks: list[np.ndarray] = []
    for s in tqdm(range(0, len(captions), batch_size), desc="text"):
        chunk = captions[s : s + batch_size]
        emb = model.get_text_embedding(chunk, use_tensor=False)
        text_chunks.append(np.asarray(emb))
    text_emb = np.concatenate(text_chunks, axis=0)

    print(f"[..] audio_emb.shape = {audio_emb.shape}, text_emb.shape = {text_emb.shape}")

    # If the metadata exposes a class label column (e.g. ESC-50), also encode
    # one prompt per class so the comparison notebook can do zero-shot
    # classification.
    if "class_name" in df.columns:
        class_names = sorted(df["class_name"].unique().tolist())
        class_prompts = [
            f'This is a sound of {n.replace("_", " ")}.' for n in class_names
        ]
        print(f"[..] Encoding {len(class_prompts)} class prompts...")
        class_text_emb = np.asarray(
            model.get_text_embedding(class_prompts, use_tensor=False)
        )
    else:
        class_names, class_text_emb = [], None

    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "audio_embeddings.npy", audio_emb)
    np.save(results_dir / "text_embeddings.npy", text_emb)
    if class_text_emb is not None:
        np.save(results_dir / "class_text_embeddings.npy", class_text_emb)
        (results_dir / "class_names.txt").write_text("\n".join(class_names))
    df.to_csv(results_dir / "metadata.csv", index=False)
    print(f"[OK] Saved CLAP outputs to {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    encode(args.csv, args.out, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
