"""Download and unpack the M2D-CLAP_2025 checkpoint into ./checkpoints/.

Usage:
    python src/download_m2d_weights.py

This is the journal-paper checkpoint:
    M2D-CLAP: Exploring General-purpose Audio-Language Representations Beyond CLAP
    (IEEE Access, 2025)

Source release: https://github.com/nttcslab/m2d/releases/tag/v0.5.0
"""

from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = (
    "https://github.com/nttcslab/m2d/releases/download/v0.5.0/"
    "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip"
)
WEIGHT_DIR_NAME = "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
CHECKPOINT_FILE = "checkpoint-30.pth"


def _resolve_paths() -> tuple[Path, Path, Path]:
    project_dir = Path(__file__).resolve().parent.parent
    ckpt_root = project_dir / "checkpoints"
    weight_dir = ckpt_root / WEIGHT_DIR_NAME
    weight_file = weight_dir / CHECKPOINT_FILE
    return ckpt_root, weight_dir, weight_file


def _report_progress(blocknum: int, blocksize: int, totalsize: int) -> None:
    if totalsize <= 0:
        return
    downloaded = blocknum * blocksize
    pct = min(100.0, downloaded * 100.0 / totalsize)
    bar = "#" * int(pct // 2) + "-" * (50 - int(pct // 2))
    sys.stdout.write(
        f"\r[{bar}] {pct:6.2f}%  ({downloaded / 1e6:7.1f} / {totalsize / 1e6:7.1f} MB)"
    )
    sys.stdout.flush()


def download_and_extract() -> Path:
    ckpt_root, weight_dir, weight_file = _resolve_paths()
    ckpt_root.mkdir(parents=True, exist_ok=True)

    if weight_file.exists():
        print(f"[OK] M2D-CLAP weights already present: {weight_file}")
        return weight_file

    zip_path = ckpt_root / f"{WEIGHT_DIR_NAME}.zip"
    if not zip_path.exists():
        print(f"[..] Downloading {ZIP_URL}")
        print(f"     -> {zip_path}")
        urllib.request.urlretrieve(ZIP_URL, zip_path, _report_progress)
        print()
    else:
        print(f"[OK] Zip already downloaded: {zip_path}")

    print(f"[..] Extracting into {ckpt_root}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(ckpt_root)

    if not weight_file.exists():
        raise RuntimeError(
            f"Extraction completed but expected file is missing: {weight_file}\n"
            f"Inspect contents of {ckpt_root} manually."
        )

    print(f"[OK] Weights ready: {weight_file}")
    print(f"[..] You may delete the zip if disk is tight: {zip_path}")
    return weight_file


if __name__ == "__main__":
    download_and_extract()
