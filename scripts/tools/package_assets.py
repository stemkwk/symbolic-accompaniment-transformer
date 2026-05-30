#!/usr/bin/env python3
"""Package heavy binary assets into separate zip files for GitHub Releases.

Three release bundles — upload each as a separate Release asset:

  python scripts/package_assets.py --preset checkpoints   → jam_checkpoints.zip
  python scripts/package_assets.py --preset soundfonts    → jam_soundfonts.zip
  python scripts/package_assets.py --preset data          → jam_data_processed.zip

Recipients only need to download what they need:
  - Inference only      : jam_checkpoints.zip + jam_soundfonts.zip
  - Resume training     : jam_checkpoints.zip + jam_data_processed.zip
  - Full setup          : all three

Custom bundles (advanced):
  python scripts/package_assets.py --no-soundfonts --no-data --no-raw --output my_bundle.zip
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jam_transformer.utils.logger import logger

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "checkpoints": {
        "output": "jam_checkpoints.zip",
        "include": {"checkpoints"},
        "description": "Model weights only — update with each training milestone",
    },
    "soundfonts": {
        "output": "jam_soundfonts.zip",
        "include": {"soundfonts"},
        "description": "FluidSynth piano samples (.sf2) — upload once, rarely changes",
    },
    "data": {
        "output": "jam_data_processed.zip",
        "include": {"data/processed"},
        "description": "Preprocessed training tensors (.pt) — upload once after full preprocessing",
    },
}

# All possible asset targets (name, relative_dir, description)
ALL_TARGETS = [
    ("checkpoints",   "checkpoints",    "Model weights & training history"),
    ("soundfonts",    "soundfonts",     "FluidSynth instruments (.sf2)"),
    ("data/raw",      "data/raw",       "Raw dataset files (.mid)"),
    ("data/processed","data/processed", "Preprocessed training tokens (.pt)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package project assets into zip files for GitHub Releases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
presets:
  checkpoints   jam_checkpoints.zip    — model weights
  soundfonts    jam_soundfonts.zip     — SF2 piano samples
  data          jam_data_processed.zip — preprocessed training tensors

examples:
  # Recommended: create the three standard release bundles
  python scripts/package_assets.py --preset checkpoints
  python scripts/package_assets.py --preset soundfonts
  python scripts/package_assets.py --preset data

  # Custom: checkpoints + soundfonts in one file
  python scripts/package_assets.py --no-data --no-raw --output jam_light_assets.zip
""",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        help="Create a standard release bundle (checkpoints / soundfonts / data).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output zip filename. Defaults to the preset name or 'jam_assets_bundle.zip'.",
    )
    parser.add_argument("--no-data",        action="store_true", help="Exclude data/processed/.")
    parser.add_argument("--no-raw",         action="store_true", help="Exclude data/raw/.")
    parser.add_argument("--no-checkpoints", action="store_true", help="Exclude checkpoints/.")
    parser.add_argument("--no-soundfonts",  action="store_true", help="Exclude soundfonts/.")
    return parser.parse_args()


def add_dir_to_zip(zip_file: zipfile.ZipFile, dir_path: Path, root_path: Path) -> int:
    """Recursively add a directory to a ZipFile. Returns number of files added."""
    count = 0
    for path in dir_path.rglob("*"):
        if path.is_file() and path.name != ".gitkeep":
            arcname = path.relative_to(root_path)
            logger.debug(f"Archiving: {arcname}")
            zip_file.write(path, arcname)
            count += 1
    return count


def main() -> None:
    args = parse_args()
    root_path = Path(__file__).resolve().parents[2]

    # ------------------------------------------------------------------
    # Resolve which targets to include and output filename
    # ------------------------------------------------------------------
    if args.preset:
        preset = PRESETS[args.preset]
        include_names = preset["include"]
        output_name = args.output or preset["output"]
        logger.info(f"Preset: {args.preset} — {preset['description']}")
    else:
        # Manual flag mode
        exclude = set()
        if args.no_checkpoints: exclude.add("checkpoints")
        if args.no_soundfonts:  exclude.add("soundfonts")
        if args.no_raw:         exclude.add("data/raw")
        if args.no_data:        exclude.add("data/processed")
        include_names = {name for name, _, _ in ALL_TARGETS} - exclude
        output_name = args.output or "jam_assets_bundle.zip"

    output_zip = (root_path / output_name).resolve()
    logger.info(f"Output: {output_zip}")

    # ------------------------------------------------------------------
    # Collect valid targets
    # ------------------------------------------------------------------
    targets = [(n, root_path / d, desc) for n, d, desc in ALL_TARGETS if n in include_names]
    valid_targets = []
    for name, path, desc in targets:
        if path.exists() and any(path.iterdir()):
            valid_targets.append((name, path, desc))
            logger.info(f"Found: [{name}] — {desc}")
        else:
            logger.warning(f"Skipping [{name}]: directory missing or empty.")

    if not valid_targets:
        logger.error("No assets found to package.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build zip
    # ------------------------------------------------------------------
    logger.info(f"Creating {output_zip.name} ...")
    try:
        total_files = 0
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, path, desc in valid_targets:
                logger.info(f"Packaging {name} ...")
                count = add_dir_to_zip(zf, path, root_path)
                logger.info(f"  {count} files added from [{name}].")
                total_files += count

        size_mb = output_zip.stat().st_size / (1024 * 1024)
        logger.success("=" * 50)
        logger.success(f"Done: {output_zip.name}")
        logger.success(f"  Files  : {total_files}")
        logger.success(f"  Size   : {size_mb:.1f} MB")
        logger.success("=" * 50)
        logger.info("Upload this file to GitHub Releases as a release asset.")
        logger.info("Recipients extract it directly in the repository root folder.")

    except Exception as e:
        logger.error(f"Failed: {e}")
        if output_zip.exists():
            output_zip.unlink()
        sys.exit(1)


if __name__ == "__main__":
    main()
