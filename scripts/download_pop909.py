"""Download the POP909 dataset (909 pop-song MIDIs with melody + accompaniment).

Source: https://github.com/music-x-lab/POP909-Dataset

This script is intentionally dependency-free (stdlib only) so it can run on a
cold server before `pip install -e .` finishes. Idempotent: if the target
directory already contains POP909 MIDIs it skips the download.

Usage
-----

    python scripts/download_pop909.py --out_dir data/raw/POP909

Layout produced (matches what `prepare_data.py --pop909_dir` expects):

    data/raw/POP909/
      001/001.mid
      002/002.mid
      ...
      909/909.mid
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = (
    "https://codeload.github.com/music-x-lab/POP909-Dataset/zip/refs/heads/master"
)
ARCHIVE_ROOT_PREFIX = "POP909-Dataset-master/POP909/"   # path inside the zip


def _count_midis(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.mid")) + sum(1 for _ in root.rglob("*.midi"))


def _download_with_progress(url: str, dest: Path) -> None:
    """Stream the zip to disk with a simple progress bar. Avoids loading the
    full ~80 MB into memory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "jam-transformer/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        chunk = 1 << 15            # 32 KB
        done = 0
        with dest.open("wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = done * 100.0 / total
                    sys.stdout.write(
                        f"\r  downloading… {done/1e6:6.1f} / {total/1e6:6.1f} MB "
                        f"({pct:5.1f}%)"
                    )
                else:
                    sys.stdout.write(f"\r  downloading… {done/1e6:6.1f} MB")
                sys.stdout.flush()
    sys.stdout.write("\n")


def _extract_midis(zip_path: Path, out_dir: Path) -> int:
    """Extract only the per-song MIDI files from the archive, flattening the
    `POP909-Dataset-master/POP909/<NNN>/<NNN>.mid` structure into
    `<out_dir>/<NNN>/<NNN>.mid`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name.startswith(ARCHIVE_ROOT_PREFIX):
                continue
            if not (name.endswith(".mid") or name.endswith(".midi")):
                continue
            rel = name[len(ARCHIVE_ROOT_PREFIX):]      # e.g. "001/001.mid"
            if not rel or rel.endswith("/"):
                continue
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            n_extracted += 1
    return n_extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Download POP909 MIDIs.")
    parser.add_argument("--out_dir", type=str, default="data/raw/POP909",
                        help="Where to place per-song MIDI folders.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL,
                        help="Archive URL (defaults to GitHub master tarball).")
    parser.add_argument("--keep_zip", action="store_true",
                        help="Do not delete the downloaded zip after extraction.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if MIDIs are already present.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if not args.force and out_dir.exists() and _count_midis(out_dir) > 0:
        n = _count_midis(out_dir)
        print(f"POP909 already present at {out_dir} ({n} MIDI files). "
              f"Pass --force to re-download.")
        return

    zip_path = out_dir.parent / "_pop909_master.zip"
    print(f"→ fetching {args.url}")
    print(f"→ saving to {zip_path}")
    try:
        _download_with_progress(args.url, zip_path)
    except urllib.error.URLError as e:
        raise SystemExit(f"Download failed: {e}")

    print(f"→ extracting MIDIs into {out_dir}")
    n = _extract_midis(zip_path, out_dir)
    if n == 0:
        raise SystemExit(
            "No MIDI files found inside the archive — the upstream layout may "
            "have changed. Inspect the zip manually or open an issue."
        )
    print(f"extracted {n} MIDI files")

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print(f"removed {zip_path}")


if __name__ == "__main__":
    main()
