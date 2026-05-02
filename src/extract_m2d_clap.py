"""Extract M2D-CLAP_2025 audio + text embeddings for a (audio_path, caption) CSV.

Outputs:
    results/m2d_clap/audio_embeddings.npy   (N, D) float32
    results/m2d_clap/text_embeddings.npy    (N, D) float32  (D matches audio)
    results/m2d_clap/metadata.csv           a copy of the (filtered) input rows

Reference: Niizumi et al., "M2D-CLAP: Exploring General-purpose Audio-Language
Representations Beyond CLAP", IEEE Access, 2025.

Usage:
    # 1. Make sure the weights are downloaded first:
    python src/download_m2d_weights.py
    # 2. Then run extraction (defaults to ESC-50 metadata):
    python src/extract_m2d_clap.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

# portable_m2d is vendored alongside this file
from portable_m2d import PortableM2D  # noqa: E402  (after sys.path tweak)

DEFAULT_CSV = PROJECT_DIR / "data" / "esc50_metadata.csv"
RESULTS_DIR = PROJECT_DIR / "results" / "m2d_clap"
DEFAULT_WEIGHT = (
    PROJECT_DIR
    / "checkpoints"
    / "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
    / "checkpoint-30.pth"
)

SR = 16_000
CLIP_SECONDS = 10  # M2D-CLAP_2025 was trained with 10 s @ 16 kHz inputs


def filter_corrupt_files(df: pd.DataFrame) -> pd.DataFrame:
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


def load_audio_16k_10s(path: str) -> np.ndarray:
    """Load mono audio at 16 kHz and right-pad / truncate to exactly 10 seconds."""
    import librosa

    y, _ = librosa.load(path, sr=SR, mono=True)
    target = SR * CLIP_SECONDS
    if y.shape[-1] >= target:
        y = y[..., :target]
    else:
        y = np.pad(y, (0, target - y.shape[-1]))
    return y.astype(np.float32)


@torch.no_grad()
def encode_audio_in_batches(model: PortableM2D, paths: list[str], batch_size: int) -> np.ndarray:
    embs: list[np.ndarray] = []
    device = next(model.parameters()).device
    for start in tqdm(range(0, len(paths), batch_size), desc="audio"):
        chunk = paths[start : start + batch_size]
        wavs = [load_audio_16k_10s(p) for p in chunk]
        wavs_t = torch.tensor(np.stack(wavs)).to(device)
        emb = model.encode_clap_audio(wavs_t)
        embs.append(emb.detach().cpu().float().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def encode_text_in_batches(model: PortableM2D, texts: list[str], batch_size: int) -> np.ndarray:
    embs: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="text"):
        chunk = texts[start : start + batch_size]
        emb = model.encode_clap_text(chunk, truncate=True)
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().float().numpy()
        else:
            emb = np.asarray(emb, dtype=np.float32)
        embs.append(emb)
    return np.concatenate(embs, axis=0)


def encode(
    csv_path: Path,
    results_dir: Path,
    weight_file: Path,
    audio_batch: int = 16,
    text_batch: int = 32,
) -> None:
    if not weight_file.exists():
        raise FileNotFoundError(
            f"M2D-CLAP weight file not found: {weight_file}\n"
            f"Run `python src/download_m2d_weights.py` first."
        )

    df = pd.read_csv(csv_path)
    print(f"[..] Loaded CSV: {csv_path}  rows={len(df)}")
    df = filter_corrupt_files(df)
    audio_paths = df["audio_path"].tolist()
    captions = df["caption"].tolist()
    print(f"[..] Effective sample count: {len(audio_paths)}")

    print(f"[..] Loading PortableM2D from {weight_file}")
    model = PortableM2D(weight_file=str(weight_file), flat_features=True)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[..] Using device: {device}")
    model = model.to(device).eval()

    audio_emb = encode_audio_in_batches(model, audio_paths, batch_size=audio_batch)
    text_emb = encode_text_in_batches(model, captions, batch_size=text_batch)
    print(f"[..] audio_emb.shape = {audio_emb.shape}, text_emb.shape = {text_emb.shape}")
    assert audio_emb.shape[0] == text_emb.shape[0] == len(df)
    assert audio_emb.shape[1] == text_emb.shape[1], (
        "Audio and text embedding dims must match for cosine similarity. "
        f"Got audio={audio_emb.shape}, text={text_emb.shape}"
    )

    if "class_name" in df.columns:
        class_names = sorted(df["class_name"].unique().tolist())
        class_prompts = [
            f'This is a sound of {n.replace("_", " ")}.' for n in class_names
        ]
        print(f"[..] Encoding {len(class_prompts)} class prompts...")
        with torch.no_grad():
            cemb = model.encode_clap_text(class_prompts, truncate=True)
            if isinstance(cemb, torch.Tensor):
                cemb = cemb.detach().cpu().float().numpy()
            else:
                cemb = np.asarray(cemb, dtype=np.float32)
    else:
        class_names, cemb = [], None

    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "audio_embeddings.npy", audio_emb)
    np.save(results_dir / "text_embeddings.npy", text_emb)
    if cemb is not None:
        np.save(results_dir / "class_text_embeddings.npy", cemb)
        (results_dir / "class_names.txt").write_text("\n".join(class_names))
    df.to_csv(results_dir / "metadata.csv", index=False)
    print(f"[OK] Saved M2D-CLAP outputs to {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--audio_batch", type=int, default=16)
    parser.add_argument("--text_batch", type=int, default=32)
    args = parser.parse_args()
    encode(
        args.csv,
        args.out,
        args.weights,
        audio_batch=args.audio_batch,
        text_batch=args.text_batch,
    )


if __name__ == "__main__":
    main()
