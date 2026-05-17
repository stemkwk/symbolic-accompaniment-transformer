#!/usr/bin/env python3
"""Download the Slakh2100 Dataset – Lightweight Symbolic-Only version (YourMT3).

Source  : https://huggingface.co/datasets/mimbres/YourMT3-dataset
Paper   : "YourMT3: a toolkit for training multi-task and multi-track music transcription model"
           Sungkyun Chang et al. (2022)

This script uses the Hugging Face Hub API to download ONLY the MIDI and yaml files 
of the entire Slakh2100 dataset (approx 96.9 MB total), completely skipping 
the 82+ GB of heavy WAV audio files.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _count_midis(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.mid")) + sum(1 for _ in root.rglob("*.midi"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the full Slakh2100 (YourMT3-16k) MIDI & metadata files from Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/raw/slakh2100",
        help="Where to extract the MIDI files (default: data/raw/slakh2100).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download and overwrite even if files are already present.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    slakh_dir = out_dir / "slakh2100_yourmt3_16k"
    indexes_dir = out_dir / "yourmt3_indexes"

    if not args.force and slakh_dir.exists() and indexes_dir.exists():
        n = _count_midis(slakh_dir)
        if n > 0:
            print(
                f"Slakh MIDI files already present at {slakh_dir} ({n:,} files). "
                f"Pass --force to re-download."
            )
            return

    # Ensure huggingface_hub is installed
    try:
        import huggingface_hub
    except ImportError:
        print("huggingface_hub is required. Installing via pip...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
            import huggingface_hub
        except Exception as exc:
            raise SystemExit(
                f"Failed to install huggingface_hub dynamically. Please run: pip install huggingface_hub"
            ) from exc

    from huggingface_hub import snapshot_download

    print("\n=======================================================")
    print("Downloading lightweight Slakh2100 dataset from Hugging Face...")
    print("Repository: mimbres/YourMT3-dataset")
    print("Filter: ONLY .mid, .yaml, and yourmt3_indexes/*.json files (~96.9 MB)")
    print("=======================================================\n")

    try:
        # Download only MIDI and yaml files of the entire slakh2100 dataset
        local_dir = snapshot_download(
            repo_id="mimbres/YourMT3-dataset",
            repo_type="dataset",
            allow_patterns=[
                "slakh2100_yourmt3_16k/**/*.mid",
                "slakh2100_yourmt3_16k/**/*.yaml",
                "yourmt3_indexes/*.json"
            ],
            tqdm_class=None, # Use default print or hf_transfer if installed
        )
    except Exception as exc:
        raise SystemExit(f"Hugging Face download failed: {exc}") from exc

    # Copy files from HF cache to the target directory
    print(f"\nHF cache download complete. Copying files to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    src_slakh = Path(local_dir) / "slakh2100_yourmt3_16k"
    src_indexes = Path(local_dir) / "yourmt3_indexes"

    if src_slakh.exists():
        print(f"  Copying slakh2100_yourmt3_16k...")
        if slakh_dir.exists():
            shutil.rmtree(slakh_dir)
        shutil.copytree(src_slakh, slakh_dir)

    if src_indexes.exists():
        print(f"  Copying yourmt3_indexes...")
        if indexes_dir.exists():
            shutil.rmtree(indexes_dir)
        shutil.copytree(src_indexes, indexes_dir)

    n_mid = _count_midis(slakh_dir)
    print(f"\n✔ Success! {n_mid:,} MIDI files successfully prepared under {slakh_dir}")
    print(f"Index files copied to {indexes_dir}")
    print(f"\nNext step:")
    print(f"  python scripts/prepare_data.py --slakh_dir {slakh_dir} --out_dir data/processed")


if __name__ == "__main__":
    main()
