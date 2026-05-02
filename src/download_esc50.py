"""Download ESC-50 (Environmental Sound Classification, 50 classes) and build
a (audio_path, caption) CSV ready for the audio-text retrieval pipeline.

ESC-50: 2,000 environmental audio clips, 5 s each, 44.1 kHz, organized in 50
balanced classes. https://github.com/karolpiczak/ESC-50

We use the class label as a natural-language caption with the LAION-CLAP-style
prompt:
    "This is a sound of {class_name}."

This matches the zero-shot prompting strategy used in both the LAION CLAP paper
and the M2D-CLAP paper, so paired-similarity numbers are directly comparable.

Output (relative to project root):
    data/ESC-50-master/                       full repo unzipped (audio + meta)
    data/esc50_metadata.csv                   columns: audio_path, caption, fold, class_name, target

Usage:
    python src/download_esc50.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ZIP_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
ZIP_NAME = "ESC-50-master.zip"
DIR_NAME = "ESC-50-master"


def _resolve_paths() -> tuple[Path, Path, Path, Path, Path]:
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"
    zip_path = data_dir / ZIP_NAME
    extract_dir = data_dir / DIR_NAME
    audio_dir = extract_dir / "audio"
    meta_csv = extract_dir / "meta" / "esc50.csv"
    return data_dir, zip_path, extract_dir, audio_dir, meta_csv


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


def download_and_extract() -> tuple[Path, Path]:
    data_dir, zip_path, extract_dir, audio_dir, meta_csv = _resolve_paths()
    data_dir.mkdir(parents=True, exist_ok=True)

    if not extract_dir.exists():
        if not zip_path.exists():
            print(f"[..] Downloading ESC-50 from {ZIP_URL}")
            print(f"     -> {zip_path}")
            urllib.request.urlretrieve(ZIP_URL, zip_path, _report_progress)
            print()
        print(f"[..] Extracting into {data_dir}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
    else:
        print(f"[OK] ESC-50 already extracted at {extract_dir}")

    if not meta_csv.exists():
        raise RuntimeError(f"Expected ESC-50 metadata CSV not found: {meta_csv}")
    if not audio_dir.exists():
        raise RuntimeError(f"Expected ESC-50 audio dir not found: {audio_dir}")

    return audio_dir, meta_csv


def caption_for(class_name: str) -> str:
    """ESC-50 raw class names use underscores (e.g., 'crackling_fire').
    Convert to a natural prompt that both CLAP and M2D-CLAP handle well."""
    name = class_name.replace("_", " ")
    return f"This is a sound of {name}."


def build_metadata_csv(audio_dir: Path, meta_csv: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    # ESC-50's CSV columns: filename, fold, target, category, esc10, src_file, take
    df = df.rename(columns={"filename": "audio_filename", "category": "class_name"})
    df["audio_path"] = df["audio_filename"].apply(
        lambda fn: str((audio_dir / fn).resolve())
    )
    df["caption"] = df["class_name"].apply(caption_for)
    out_df = df[
        ["audio_filename", "audio_path", "caption", "class_name", "target", "fold"]
    ]
    out_df.to_csv(out_csv, index=False)
    return out_df


def main() -> None:
    audio_dir, meta_csv = download_and_extract()
    project_dir = Path(__file__).resolve().parent.parent
    out_csv = project_dir / "data" / "esc50_metadata.csv"

    df = build_metadata_csv(audio_dir, meta_csv, out_csv)

    print()
    print(f"[OK] Built metadata CSV: {out_csv}")
    print(f"     rows         = {len(df)}")
    print(f"     classes      = {df['class_name'].nunique()}")
    print(f"     example caption: {df['caption'].iloc[0]!r}")
    print(f"     example audio  : {df['audio_path'].iloc[0]}")


if __name__ == "__main__":
    main()
