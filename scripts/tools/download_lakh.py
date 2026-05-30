"""Download the Lakh MIDI Dataset – Clean subset (~1.7 GB, 17 k MIDIs).

Source  : https://colinraffel.com/projects/lmd/
Paper   : "Learning-Based Methods for Comparing Sequences, with Applications to
           Audio-to-MIDI Alignment and Music Structure Analysis", Colin Raffel (2016)
Tarball : http://hog.ee.columbia.edu/craffel/lmd/lmd_clean.tar.gz

This script is intentionally dependency-free (stdlib only) so it can run on a
cold server before `pip install -e .` finishes.  Idempotent: if the target
directory already contains MIDI files it skips the download.

Usage
-----

    python scripts/download_lakh.py --out_dir data/raw/lmd_clean

Layout produced (matches what `prepare_data.py --lakh_dir` expects):

    data/raw/lmd_clean/
      0/
        0a1b2c....mid       # MD5-hash-bucketed flat layout (unchanged from tarball)
      1/
        ...
      f/
        ...
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

# LMD-clean: 17,257 unique cleaned MIDI files.
# See https://colinraffel.com/projects/lmd/ for the other subsets.
DEFAULT_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_clean.tar.gz"
# Approximate archive size (bytes) – used only when Content-Length is absent.
_APPROX_BYTES = 1_700_000_000


def _count_midis(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.mid")) + sum(1 for _ in root.rglob("*.midi"))


def _download_with_progress(url: str, dest: Path) -> None:
    """Stream the tar.gz to disk with a live progress bar.

    Streams in 64-KB chunks to keep memory usage flat even for the 1.7 GB
    archive.  Writes straight to disk so no additional temp copy is needed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "jam-transformer/0.1"})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.URLError as exc:
        raise SystemExit(f"Download failed: {exc}") from exc

    total = int(resp.headers.get("Content-Length") or _APPROX_BYTES)
    chunk_size = 1 << 16  # 64 KB
    done = 0

    with dest.open("wb") as fh, resp:
        while True:
            buf = resp.read(chunk_size)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            pct = done * 100.0 / total
            bar_w = 30
            filled = int(bar_w * done / total)
            bar = "#" * filled + "-" * (bar_w - filled)
            sys.stdout.write(
                f"\r  [{bar}] {done / 1e6:7.1f} / {total / 1e6:7.1f} MB  ({pct:5.1f}%)"
            )
            sys.stdout.flush()
    sys.stdout.write("\n")


def _extract_tar(tar_path: Path, out_dir: Path) -> int:
    """Extract the tar.gz into `out_dir`, stripping the top-level directory
    name so the content lands directly under `out_dir`.

    The tarball typically has a single top-level folder (``lmd_clean/``).  We
    strip it so ``out_dir/0/xxx.mid`` rather than ``out_dir/lmd_clean/0/xxx.mid``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        total = len(members)
        for i, member in enumerate(members):
            if i % 500 == 0:
                pct = i * 100.0 / max(total, 1)
                sys.stdout.write(f"\r  extracting… {i}/{total}  ({pct:.0f}%)")
                sys.stdout.flush()

            # Strip the top-level directory component (e.g. "lmd_clean/").
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue  # top-level dir entry itself — skip
            rel = Path(*parts[1:])

            if member.isdir():
                (out_dir / rel).mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                dest = out_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as src, dest.open("wb") as dst:  # type: ignore[arg-type]
                    dst.write(src.read())
                n += 1
    sys.stdout.write(f"\r  extracting… done ({n} files)                  \n")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Lakh MIDI Dataset – Clean subset (~1.7 GB).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out_dir", type=str, default="data/raw/lmd_clean",
        help="Where to extract the MIDI files (default: data/raw/lmd_clean).",
    )
    parser.add_argument(
        "--url", type=str, default=DEFAULT_URL,
        help="Override the download URL.",
    )
    parser.add_argument(
        "--keep_tar", action="store_true",
        help="Keep the downloaded tar.gz after extraction (for re-use).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if MIDI files are already present.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if not args.force and out_dir.exists():
        n = _count_midis(out_dir)
        if n > 0:
            print(
                f"Lakh MIDI already present at {out_dir}  ({n:,} files).  "
                f"Pass --force to re-download."
            )
            return

    tar_path = out_dir.parent / "_lmd_clean.tar.gz"
    print(f"Fetching: {args.url}")
    print(f"Saving to: {tar_path}  (~1.7 GB, this may take a few minutes)")
    _download_with_progress(args.url, tar_path)
    print(f"Download complete: {tar_path.stat().st_size / 1e6:.0f} MB")

    print(f"Extracting into {out_dir} …")
    n = _extract_tar(tar_path, out_dir)
    if n == 0:
        raise SystemExit(
            "No files extracted — the archive may be empty or the layout has changed. "
            "Inspect the tar.gz manually: `tar -tzf _lmd_clean.tar.gz | head -20`"
        )
    print(f"Extracted {n:,} MIDI files -> {out_dir}")

    if not args.keep_tar:
        tar_path.unlink(missing_ok=True)
        print(f"Removed {tar_path}")

    n_mid = _count_midis(out_dir)
    print(f"\nReady: {n_mid:,} MIDI files under {out_dir}")
    print(f"Next step:")
    print(f"  python scripts/prepare_data.py --lakh_dir {out_dir} --out_dir data/processed")


if __name__ == "__main__":
    main()
