"""Bundle the locally-prepared assets into a tarball for server upload.

Default contents:
    project_transformer/    (source, configs, scripts, tests)
    data/processed/         (tokenized .pt + meta + chunk index)

data/raw/ is excluded — the server never needs raw MIDIs once tokens are built.

Compression options (--compress):
    zst  [default]  zstd level 3, 멀티스레드.  gz 대비 압축 속도 3-5배 빠름,
                    압축률 동일. Ubuntu 22.04(RunPod/Vast.ai) zstd 내장 포함.
                    로컬: pip install zstandard  (한 번만)
                    서버 추출: zstd -d jam_tx_bundle.tar.zst | tar x
    gz              gzip level 6. 별도 패키지 불필요. zst보다 느리지만
                    압축률 유사. 추출: tar xzf jam_tx_bundle.tgz
    none            압축 없음. 소스만 재전송할 때 등 번들 자체가 필요 없는 경우.
                    코드만 바꿨다면 RESYNC=1 ./server/upload_bundle.sh 가 훨씬 빠름.

After upload:
    # zst (default)
    zstd -d jam_tx_bundle.tar.zst | tar x && cd project_transformer

    # gz
    tar xzf jam_tx_bundle.tgz && cd project_transformer
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import tarfile
import time
from pathlib import Path

# Directories excluded from the source tree walk.
_SOURCE_EXCLUDES = {
    # build / cache artefacts
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", ".git",
    # virtual environments (any name pattern)
    ".mm_env", "env", "ENV", "active_env",
    # training outputs (generated on the server, not needed there)
    "checkpoints", "logs", "output", "lightning_logs", "sweep_results",
    # artefacts pulled *from* the server — never re-upload back
    "pulled_checkpoints", "pulled_logs",
    # data: raw MIDIs excluded; processed tokens added separately below
    "data",
    # local-only: inspection results, pre-built bundles, soundfonts
    "inspection", "bundles", "soundfonts",
    # misc local-only directories
    "tools", "notebooks",
}

# Individual files always excluded by name.
_SOURCE_EXCLUDE_FILES = {
    ".DS_Store",
    ".env",       # secrets — set manually on the server
}

# File extensions that are packaging artefacts or too large / useless on server.
# This prevents bundling the bundle itself or previously fetched checkpoints.
_SOURCE_EXCLUDE_SUFFIXES = {
    # packaging artefacts (jam_tx_bundle.tgz, jam_*.zip etc.)
    ".tgz", ".tar", ".zst", ".bz2", ".xz", ".zip",
    # checksum sidecars
    ".sha256",
    # compiled bytecode (belt-and-suspenders — pycache dir is already excluded)
    ".pyc", ".pyo",
    # audio / MIDI outputs that don't belong on a training server
    ".wav", ".mp3", ".sf2",
}


def _should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & _SOURCE_EXCLUDES:
        return True
    if path.name in _SOURCE_EXCLUDE_FILES:
        return True
    # Reject archive/checksum files regardless of name
    # (handles jam_tx_bundle.tgz, jam_tx_bundle.tar.zst, *.sha256, etc.)
    for suffix in _SOURCE_EXCLUDE_SUFFIXES:
        if path.name.endswith(suffix):
            return True
    return False


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Return hex SHA-256 of *path* (streaming, constant memory)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _fill_tar(tar: tarfile.TarFile, root: Path,
              data_dir: Path | None) -> tuple[int, int]:
    """Add source tree and (optionally) data/processed into *tar*.
    Returns (n_files, total_raw_bytes)."""
    n_files = 0
    total_bytes = 0

    # --- source tree ---
    for dirpath, dirnames, filenames in os.walk(root):
        dpath = Path(dirpath)
        rel = dpath.relative_to(root)
        dirnames[:] = [d for d in dirnames if not _should_skip(rel / d)]
        for fn in filenames:
            rp = rel / fn
            if _should_skip(rp):
                continue
            full = dpath / fn
            arc = Path("project_transformer") / rp
            tar.add(str(full), arcname=str(arc).replace("\\", "/"), recursive=False)
            n_files += 1
            total_bytes += full.stat().st_size

    # --- tokenized data ---
    if data_dir is not None and data_dir.exists():
        for p in sorted(data_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(data_dir)
            arc = Path("project_transformer") / "data" / "processed" / rel
            tar.add(str(p), arcname=str(arc).replace("\\", "/"), recursive=False)
            n_files += 1
            total_bytes += p.stat().st_size
    elif data_dir is not None:
        print(f"  ! data dir not found: {data_dir} (skipped)")

    return n_files, total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package source + tokens for server upload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=str, default="",
                        help="Output archive path. Default: jam_tx_bundle.tgz "
                             "(or .tar.zst / .tar for other --compress modes).")
    parser.add_argument("--compress", choices=["gz", "zst", "none"], default="zst",
                        help="Compression algorithm (default: zst). zst는 gz 대비 3-5배 빠름.")
    parser.add_argument("--include_data", type=str, default="data/processed",
                        help="Path to tokenized data dir (set to '' to exclude).")
    parser.add_argument("--source_root", type=str, default=".",
                        help="Project root to bundle from.")
    parser.add_argument("--no-checksum", action="store_true",
                        help="Skip writing the .sha256 file.")
    args = parser.parse_args()

    root = Path(args.source_root).resolve()

    # Resolve output path and extension
    ext_map = {"gz": ".tgz", "zst": ".tar.zst", "none": ".tar"}
    default_stem = "jam_tx_bundle"
    if args.out:
        out = Path(args.out).resolve()
    else:
        bundles_dir = root / "bundles"
        bundles_dir.mkdir(exist_ok=True)
        out = bundles_dir / (default_stem + ext_map[args.compress])

    # Resolve data dir
    data_dir: Path | None = None
    if args.include_data:
        data_dir = Path(args.include_data)
        if not data_dir.is_absolute():
            data_dir = (root / data_dir).resolve()

    print(f"→ packaging from  {root}")
    print(f"→ compress        {args.compress}")
    print(f"→ writing to      {out}")
    t0 = time.perf_counter()

    if args.compress == "gz":
        with tarfile.open(out, "w:gz", compresslevel=6) as tar:
            n_files, total_bytes = _fill_tar(tar, root, data_dir)

    elif args.compress == "none":
        with tarfile.open(out, "w:") as tar:
            n_files, total_bytes = _fill_tar(tar, root, data_dir)

    elif args.compress == "zst":
        try:
            import zstandard  # pip install zstandard
        except ImportError:
            sys.exit(
                "zstd compression requires: pip install zstandard\n"
                "Or use --compress gz (default) instead."
            )
        cctx = zstandard.ZstdCompressor(level=3, threads=-1)  # threads=-1 = auto
        with open(out, "wb") as fout:
            with cctx.stream_writer(fout, closefd=False) as zst_out:
                # 'w|' = streaming (no seeking) — required when fileobj is not seekable
                with tarfile.open(fileobj=zst_out, mode="w|") as tar:  # type: ignore[arg-type]
                    n_files, total_bytes = _fill_tar(tar, root, data_dir)

    elapsed = time.perf_counter() - t0
    size = out.stat().st_size
    ratio = (1.0 - size / max(1, total_bytes)) * 100.0
    print(f"\nOK  {n_files} files  |  {total_bytes/1e6:.1f} MB raw  →  "
          f"{size/1e6:.1f} MB {args.compress}  ({ratio:.1f}% reduction)  "
          f"in {elapsed:.1f}s")

    # --- SHA-256 checksum ---
    sha_path: Path | None = None
    if not args.no_checksum:
        print("→ computing SHA-256 ...", end=" ", flush=True)
        digest = _sha256(out)
        # Standard sha256sum format: "HASH  filename"
        sha_path = out.with_suffix("").with_suffix(".sha256")
        sha_path.write_text(f"{digest}  {out.name}\n", encoding="utf-8")
        print(f"{digest[:16]}…  →  {sha_path.name}")

    # --- Upload instructions ---
    print()
    print("─" * 60)
    print("Transfer (from local machine):")
    print()
    print("  # First-time or full re-upload:")
    print(f"  SSH_HOST=user@server ./server/upload_bundle.sh")
    print()
    if sha_path:
        print("  # The script verifies SHA-256 on the remote after transfer.")
        print()
    print("  # Or manually with rsync (fastest SSH cipher, no double-compress):")
    print(f"  rsync -av --rsh='ssh -o Compression=no -c aes128-gcm@openssh.com' \\")
    print(f"    --whole-file --progress {out.name} user@server:~/")
    print()
    print("On the server after extraction:")
    if args.compress == "gz":
        print(f"  tar xzf {out.name} && cd project_transformer")
    elif args.compress == "zst":
        print(f"  zstd -d {out.name} | tar x && cd project_transformer")
    elif args.compress == "none":
        print(f"  tar xf {out.name} && cd project_transformer")
    print("  bash server/00_bringup.sh")
    print("  bash server/10_dry_run.sh")
    print(f"  bash server/20_train.sh")
    print("─" * 60)


if __name__ == "__main__":
    main()
