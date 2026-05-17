#!/usr/bin/env python3
"""Script to package all ignored heavy assets (checkpoints, soundfonts, and preprocessed data)
into a single zip bundle for easy sharing via Google Drive / OneDrive.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# Add src/ to path so we can use project logger
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jam_transformer.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package ignored project assets (checkpoints, soundfonts, data) into a zip."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="jam_assets_bundle.zip",
        help="Path to the output zip file (default: jam_assets_bundle.zip)",
    )
    parser.add_argument(
        "--no-data",
        action="store_true",
        help="Skip packaging the data/processed/ directory to keep the bundle size smaller.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip packaging the data/raw/ directory.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Skip packaging the checkpoints/ directory.",
    )
    parser.add_argument(
        "--no-soundfonts",
        action="store_true",
        help="Skip packaging the soundfonts/ directory.",
    )
    return parser.parse_args()


def add_dir_to_zip(zip_file: zipfile.ZipFile, dir_path: Path, root_path: Path) -> int:
    """Recursively adds a directory's contents to a ZipFile, preserving relative structure.

    Returns the number of files added.
    """
    count = 0
    # Walk through directory
    for path in dir_path.rglob("*"):
        if path.is_file():
            # Skip placeholders like .gitkeep or dummy files
            if path.name == ".gitkeep":
                continue
            # Archive path is relative to the project root
            arcname = path.relative_to(root_path)
            logger.debug(f"Archiving: {arcname}")
            zip_file.write(path, arcname)
            count += 1
    return count


def main() -> None:
    args = parse_args()
    root_path = Path(__file__).resolve().parents[1]
    output_zip = Path(args.output).resolve()

    logger.info("Initializing Jam Transformer Asset Packager...")
    logger.info(f"Target Bundle Path: {output_zip}")

    # Define directories to package
    targets = []
    if not args.no_checkpoints:
        targets.append(("checkpoints", root_path / "checkpoints", "Model weights & training history"))
    if not args.no_soundfonts:
        targets.append(("soundfonts", root_path / "soundfonts", "FluidSynth instruments (.sf2)"))
    if not args.no_raw:
        targets.append(("data/raw", root_path / "data" / "raw", "Raw dataset files (.mid)"))
    if not args.no_data:
        targets.append(("data/processed", root_path / "data" / "processed", "Preprocessed training tokens (.pt)"))

    # Check existence
    valid_targets = []
    for name, path, desc in targets:
        if path.exists() and any(path.iterdir()):
            valid_targets.append((name, path, desc))
            logger.info(f"Found assets: [{name}] - {desc}")
        else:
            logger.warning(f"Skipping [{name}]: Directory does not exist or is empty.")

    if not valid_targets:
        logger.error("No valid assets found to package! Make sure your folders contain files.")
        sys.exit(1)

    logger.info(f"Creating zip file: {output_zip.name}...")
    try:
        total_files = 0
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for name, path, desc in valid_targets:
                logger.info(f"Packaging {name}...")
                count = add_dir_to_zip(zip_file, path, root_path)
                logger.info(f"Successfully packaged {count} files from [{name}].")
                total_files += count

        # Print final file size information
        size_mb = output_zip.stat().st_size / (1024 * 1024)
        logger.success("==================================================")
        logger.success("📦 Assets Packaging Completed Successfully!")
        logger.success(f"• Total Files Bundled: {total_files}")
        logger.success(f"• Output Archive: {output_zip.name}")
        logger.success(f"• Archive Size: {size_mb:.2f} MB")
        logger.success("==================================================")
        logger.info("✨ How to share this bundle:")
        logger.info("1. Upload this zip file to your Google Drive.")
        logger.info("2. Share the view/download link in your README.md.")
        logger.info("3. Other users just need to extract the zip file directly in the repository root folder!")

    except Exception as e:
        logger.error(f"Failed to create assets bundle: {e}")
        if output_zip.exists():
            output_zip.unlink()
        sys.exit(1)


if __name__ == "__main__":
    main()
