#!/usr/bin/env python3
"""Download the Slakh2100 dataset - MIDI/metadata only (no audio).

Two sources are available:

  default (no flag)
    Hugging Face `mimbres/YourMT3-dataset`. Pulls ONLY the loose .mid/.yaml
    files of the YourMT3-16k variant (~97 MB). This is a *reduced* mirror:
    884 tracks (train 463 / val 270 / test 151).

  --redux
    The full, deduplicated Slakh2100-redux (1710 tracks: 1289/270/151) from
    Zenodo record 4599666. No official MIDI-only archive exists, so we STREAM
    the 104 GB FLAC tarball and extract only the MIDI + metadata.yaml on the
    fly - audio bytes are read-through but never written (~150 MB lands on
    disk). Best run on a fast-network machine (e.g. the training box).

Examples
--------
  # Reduced MIDI-only mirror (fast, 884 tracks)
  python scripts/tools/download_slakh.py --out_dir data/raw/slakh2100

  # Full deduplicated redux (1710 tracks, streams 104 GB)
  python scripts/tools/download_slakh.py --redux --out_dir data/raw

Paper : "YourMT3" (Chang et al., 2022) / Slakh2100 (Manilow et al., 2019)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

# Slakh2100-redux (deduplicated, 1710 tracks). Single gzip stream, ~104 GB.
REDUX_URL = (
    "https://zenodo.org/records/4599666/files/"
    "slakh2100_flac_redux.tar.gz?download=1"
)


def _count_midis(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.mid")) + sum(1 for _ in root.rglob("*.midi"))


def _count_tracks(root: Path) -> int:
    return sum(1 for _ in root.rglob("all_src.mid"))


# --------------------------------------------------------------------------- #
# Source 1 - YourMT3-16k loose MIDI mirror on Hugging Face (~97 MB)            #
# --------------------------------------------------------------------------- #
def download_yourmt3(out_dir: Path, force: bool = False) -> None:
    slakh_dir = out_dir / "slakh2100_yourmt3_16k"
    indexes_dir = out_dir / "yourmt3_indexes"

    if not force and slakh_dir.exists() and indexes_dir.exists():
        n = _count_midis(slakh_dir)
        if n > 0:
            print(
                f"Slakh MIDI files already present at {slakh_dir} ({n:,} files). "
                f"Pass --force to re-download."
            )
            return

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub is required. Installing via pip...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "huggingface_hub"],
                check=True,
            )
        except Exception as exc:
            raise SystemExit(
                "Failed to install huggingface_hub. Run: pip install huggingface_hub"
            ) from exc

    from huggingface_hub import snapshot_download

    print("\n=======================================================")
    print("Downloading lightweight Slakh2100 (YourMT3-16k) from Hugging Face...")
    print("Repository: mimbres/YourMT3-dataset")
    print("Filter: ONLY .mid, .yaml, and yourmt3_indexes/*.json files (~97 MB)")
    print("=======================================================\n")

    try:
        local_dir = snapshot_download(
            repo_id="mimbres/YourMT3-dataset",
            repo_type="dataset",
            allow_patterns=[
                "slakh2100_yourmt3_16k/**/*.mid",
                "slakh2100_yourmt3_16k/**/*.yaml",
                "yourmt3_indexes/*.json",
            ],
            tqdm_class=None,
        )
    except Exception as exc:
        raise SystemExit(f"Hugging Face download failed: {exc}") from exc

    print(f"\nHF cache download complete. Copying files to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    src_slakh = Path(local_dir) / "slakh2100_yourmt3_16k"
    src_indexes = Path(local_dir) / "yourmt3_indexes"

    if src_slakh.exists():
        print("  Copying slakh2100_yourmt3_16k...")
        if slakh_dir.exists():
            shutil.rmtree(slakh_dir)
        shutil.copytree(src_slakh, slakh_dir)

    if src_indexes.exists():
        print("  Copying yourmt3_indexes...")
        if indexes_dir.exists():
            shutil.rmtree(indexes_dir)
        shutil.copytree(src_indexes, indexes_dir)

    n_mid = _count_midis(slakh_dir)
    print(f"\n[OK] Success! {n_mid:,} MIDI files prepared under {slakh_dir}")
    print(f"\nNext step:")
    print(f"  python scripts/prepare_data.py --slakh_dir {slakh_dir} --out_dir data/processed")


# --------------------------------------------------------------------------- #
# Source 2 - full redux, stream the 104 GB tarball, keep MIDI/yaml only        #
# --------------------------------------------------------------------------- #
class _CountingReader:
    """Wrap a file object to count bytes streamed (for progress reporting)."""

    def __init__(self, fileobj):
        self._f = fileobj
        self.n = 0

    def read(self, size: int = -1) -> bytes:
        b = self._f.read(size)
        self.n += len(b)
        return b


def _wanted(member: tarfile.TarInfo, include_omitted: bool) -> bool:
    if not member.isfile():
        return False
    name = member.name
    base = name.rsplit("/", 1)[-1]
    if not (name.endswith(".mid") or base == "metadata.yaml"):
        return False
    if not include_omitted and "/omitted/" in name:
        return False
    return True


def _safe_target(dest: Path, name: str) -> Path | None:
    """Resolve a tar member name under dest, refusing path traversal."""
    if name.startswith(("/", "\\")) or ".." in Path(name).parts:
        return None
    target = (dest / name).resolve()
    if dest.resolve() not in target.parents and target != dest.resolve():
        return None
    return target


def download_redux(out_dir: Path, force: bool = False,
                   include_omitted: bool = False) -> None:
    dest = out_dir / "slakh2100_redux"
    if not force and dest.exists() and _count_tracks(dest) > 0:
        print(
            f"Redux MIDI already present at {dest} "
            f"({_count_tracks(dest):,} tracks). Pass --force to re-download."
        )
        return
    dest.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print("Streaming Slakh2100-redux (1710 tracks) from Zenodo 4599666...")
    print("The 104 GB FLAC tarball is read-through; only .mid + metadata.yaml")
    print(f"are written to disk (~150 MB). {'INCLUDING' if include_omitted else 'Skipping'} omitted/ duplicates.")
    print("This transfers 104 GB - best on a fast-network machine.")
    print("=======================================================\n")

    req = urllib.request.Request(
        REDUX_URL, headers={"User-Agent": "jam-transformer/slakh-dl"}
    )
    extracted = 0
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            reader = _CountingReader(resp)
            # 'r|gz' = streaming, non-seekable; we extract each member in place.
            with tarfile.open(fileobj=reader, mode="r|gz") as tf:
                for member in tf:
                    if not _wanted(member, include_omitted):
                        continue
                    target = _safe_target(dest, member.name)
                    if target is None:
                        print(f"  ! skipping suspicious path: {member.name}")
                        continue
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                    extracted += 1
                    if extracted % 200 == 0:
                        gb = reader.n / 1e9
                        rate = gb / max(time.time() - t0, 1e-6)
                        print(
                            f"  extracted {extracted:,} files | "
                            f"streamed {gb:5.1f} GB | {rate:.2f} GB/s | "
                            f"{time.time() - t0:5.0f}s"
                        )
    except KeyboardInterrupt:
        raise SystemExit(
            f"\nInterrupted after extracting {extracted:,} files. "
            f"The gzip stream is not resumable - re-run with --force to restart."
        )
    except Exception as exc:
        raise SystemExit(
            f"\nRedux stream failed after {extracted:,} files / "
            f"{reader.n / 1e9:.1f} GB: {exc}\n"
            f"The stream is not resumable; re-run with --force to restart."
        )

    n_tracks = _count_tracks(dest)
    n_mid = _count_midis(dest)
    print(
        f"\n[OK] Success! {n_tracks:,} tracks ({n_mid:,} MIDI files) under {dest}"
    )
    print(f"  Streamed {reader.n / 1e9:.1f} GB total in {time.time() - t0:.0f}s.")
    print(f"\nNext step:")
    print(f"  python scripts/prepare_data.py --slakh_dir {dest} --out_dir data/processed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Slakh2100 MIDI/metadata (no audio).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/raw/slakh2100",
        help="Where to write files. Default data/raw/slakh2100 (yourmt3); "
             "for --redux a data/raw root is recommended.",
    )
    parser.add_argument(
        "--redux", action="store_true",
        help="Stream the full deduplicated redux (1710 tracks, 104 GB transfer) "
             "instead of the 884-track YourMT3 mirror.",
    )
    parser.add_argument(
        "--include_omitted", action="store_true",
        help="With --redux, also extract the omitted/ duplicate tracks "
             "(yields the full 2100 incl. duplicate MIDIs). Off by default.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download/overwrite even if files are already present.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.redux:
        download_redux(out_dir, force=args.force,
                       include_omitted=args.include_omitted)
    else:
        if args.include_omitted:
            print("Note: --include_omitted only applies with --redux; ignoring.")
        download_yourmt3(out_dir, force=args.force)


if __name__ == "__main__":
    main()
