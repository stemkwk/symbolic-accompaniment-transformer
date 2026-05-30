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
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = (
    "https://codeload.github.com/music-x-lab/POP909-Dataset/zip/refs/heads/master"
)
ARCHIVE_ROOT_PREFIX = "POP909-Dataset-master/POP909/"   # path inside the zip

# ---------------------------------------------------------------------------
# POP909-CL constants — pinned to a specific commit for reproducibility.
# Source: AndyWeasley2004/POP909-CL-Dataset (MIT licence)
# Paper:  "BACHI: Boundary-Aware Symbolic Chord Recognition Through Masked
#          Iterative Decoding on Pop and Classical Music" (ICASSP 2026)
# ---------------------------------------------------------------------------
_CL_COMMIT = "dfed572711e302c47278623128cc9e8b1608c230"
_CL_ZIP_URL = (
    "https://github.com/AndyWeasley2004/POP909-CL-Dataset/raw/"
    f"{_CL_COMMIT}/POP909_processed.zip"
)
_CL_SCRIPT_URL = (
    "https://raw.githubusercontent.com/AndyWeasley2004/POP909-CL-Dataset/"
    f"{_CL_COMMIT}/process_pop909.py"
)


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


def _fetch_bytes(url: str, desc: str) -> bytes:
    """Download *url* into memory with a progress indicator."""
    req = urllib.request.Request(url, headers={"User-Agent": "jam-transformer/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        chunks, done = [], 0
        while True:
            buf = resp.read(1 << 15)
            if not buf:
                break
            chunks.append(buf)
            done += len(buf)
            if total:
                sys.stdout.write(
                    f"\r  {desc} {done/1e6:6.1f} / {total/1e6:6.1f} MB"
                    f" ({done*100/total:5.1f}%)"
                )
            else:
                sys.stdout.write(f"\r  {desc} {done/1e6:6.1f} MB")
            sys.stdout.flush()
    sys.stdout.write("\n")
    return b"".join(chunks)


def _find_midi_root(base: Path) -> Path:
    """Return the directory inside *base* that directly contains .mid files."""
    if list(base.glob("*.mid")):
        return base
    for child in sorted(base.iterdir()):
        if child.is_dir() and list(child.glob("*.mid")):
            return child
    raise SystemExit(
        f"No .mid files found inside the extracted POP909-CL archive under {base}.\n"
        "The upstream zip layout may have changed — inspect it manually."
    )


def _install_cl(pop909_dir: Path, *, force: bool = False) -> None:
    """Download POP909-CL, run their process_pop909.py, and drop
    ``chord_symbol.csv`` **and** ``<NNN>_cl.mid`` into each
    ``data/raw/POP909/<NNN>/`` folder.

    The ``_cl.mid`` files are needed by ``prepare_data.py`` to compute the
    per-song preroll offset so that ``chord_symbol.csv`` positions align with
    the original POP909 MIDI time grid.

    Idempotent: skips if both CSVs and _cl.mid files are present (unless
    *force* is True).  Requires ``miditoolkit`` and ``tqdm``.
    """
    # ── idempotency ──────────────────────────────────────────────────────────
    existing_csv  = sum(1 for _ in pop909_dir.rglob("chord_symbol.csv"))
    existing_midi = sum(1 for _ in pop909_dir.rglob("*_cl.mid"))
    if existing_csv > 0 and existing_midi > 0 and not force:
        print(
            f"POP909-CL data already present "
            f"({existing_csv} chord_symbol.csv + {existing_midi} _cl.mid files "
            f"under {pop909_dir}). "
            "Pass --force_cl to re-generate."
        )
        return

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)

        # ── 1. download POP909_processed.zip ─────────────────────────────────
        print(f"→ fetching POP909-CL MIDI archive  (commit {_CL_COMMIT[:8]}…)")
        print(f"  source: {_CL_ZIP_URL}")
        try:
            zip_bytes = _fetch_bytes(_CL_ZIP_URL, "downloading…")
        except urllib.error.URLError as exc:
            raise SystemExit(f"POP909-CL zip download failed: {exc}")

        zip_path = tmp / "POP909_processed.zip"
        zip_path.write_bytes(zip_bytes)

        # ── 2. extract zip ────────────────────────────────────────────────────
        print("→ extracting POP909_processed.zip …")
        extract_root = tmp / "cl_extracted"
        extract_root.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_root)
        midi_root = _find_midi_root(extract_root)
        n_midis = len(list(midi_root.glob("*.mid")))
        print(f"  found {n_midis} MIDI files in {midi_root.name}/")

        # Sanitise filenames: some POP909-CL MIDIs have trailing spaces in
        # their stem (e.g. "043 .mid"). Windows disallows trailing spaces in
        # directory names so process_pop909.py would fail when creating the
        # output directory.  Rename in-place before running the script.
        n_renamed = 0
        for midi_file in sorted(midi_root.glob("*.mid")):
            clean_stem = midi_file.stem.strip()
            if clean_stem != midi_file.stem:
                midi_file.rename(midi_file.parent / f"{clean_stem}.mid")
                n_renamed += 1
        if n_renamed:
            print(f"  sanitised {n_renamed} filename(s) with leading/trailing whitespace")

        # ── 3. download process_pop909.py from pinned commit ──────────────────
        print(f"→ fetching process_pop909.py  (commit {_CL_COMMIT[:8]}…)")
        try:
            script_bytes = _fetch_bytes(_CL_SCRIPT_URL, "downloading…")
        except urllib.error.URLError as exc:
            raise SystemExit(f"process_pop909.py download failed: {exc}")
        script_path = tmp / "process_pop909.py"
        script_path.write_bytes(script_bytes)

        # ── 4. run process_pop909.py → chord_symbol.csv per song ─────────────
        csv_out = tmp / "chord_csvs"
        csv_out.mkdir()
        print("→ running process_pop909.py  (requires miditoolkit + tqdm) …")
        try:
            subprocess.run(
                [sys.executable, str(script_path),
                 "--pop909-root", str(midi_root),
                 "--out",         str(csv_out)],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"process_pop909.py exited with code {exc.returncode}.\n"
                "Ensure project dependencies are installed:\n"
                "  pip install -e .\n"
                "Then re-run:  python scripts/download_pop909.py --also_cl"
            )
        except FileNotFoundError:
            raise SystemExit(
                f"Could not find Python executable: {sys.executable}\n"
                "Run from the project's virtual environment."
            )

        # ── 5. copy chord_symbol.csv + <NNN>_cl.mid → data/raw/POP909/<NNN>/ ──
        csv_files = sorted(csv_out.rglob("chord_symbol.csv"))
        if not csv_files:
            raise SystemExit(
                "process_pop909.py produced no chord_symbol.csv files.\n"
                "Check the script output above for errors."
            )

        n_copied = n_skipped = n_midi_copied = 0
        for csv_file in csv_files:
            song_id  = csv_file.parent.name          # "001", "002", …
            dest_dir = pop909_dir / song_id
            if dest_dir.is_dir():
                shutil.copy2(csv_file, dest_dir / "chord_symbol.csv")
                n_copied += 1
                # Also copy the CL MIDI — required by _parse_cl_chord_csv() to
                # compute the per-song preroll that aligns CSV offsets with the
                # original POP909 MIDI time grid.
                cl_midi_src = midi_root / f"{song_id}.mid"
                if cl_midi_src.exists():
                    shutil.copy2(cl_midi_src, dest_dir / f"{song_id}_cl.mid")
                    n_midi_copied += 1
            else:
                n_skipped += 1

        print(
            f"✓ installed {n_copied} chord_symbol.csv + "
            f"{n_midi_copied} _cl.mid files into {pop909_dir}/"
        )
        if n_skipped:
            print(
                f"  ({n_skipped} songs skipped — no matching folder in {pop909_dir})"
            )

    # ── 6. usage hint ─────────────────────────────────────────────────────────
    print()
    print("Next step — re-tokenise POP909 shards to use the corrected annotations:")
    print(f"  # Windows")
    print(f"  del data\\processed\\pop909_*.pt")
    print(f"  python scripts/prepare_data.py --pop909_dir {pop909_dir} --out_dir data/processed")


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
    parser.add_argument("--also_cl", action="store_true",
                        help=(
                            "Also download POP909-CL (BACHI/ICASSP-2026 corrected "
                            "annotations) and install chord_symbol.csv into each "
                            "song folder. Requires miditoolkit + tqdm to be installed."
                        ))
    parser.add_argument("--force_cl", action="store_true",
                        help="Re-generate CL chord_symbol.csv files even if present.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if not args.force and out_dir.exists() and _count_midis(out_dir) > 0:
        n = _count_midis(out_dir)
        print(f"POP909 already present at {out_dir} ({n} MIDI files). "
              f"Pass --force to re-download.")
        if args.also_cl:
            print()
            _install_cl(out_dir, force=args.force_cl)
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

    if args.also_cl:
        print()
        _install_cl(out_dir, force=args.force_cl)


if __name__ == "__main__":
    main()
