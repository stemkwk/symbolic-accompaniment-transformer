"""MIDI → tokenized .pt shards (relative harmonic encoding).

Modes:
    --pop909_dir <path>  POP909. GT chord/key from annotations/.
    --lakh_dir   <path>  Lakh MIDI Clean. Chord auto-extracted, core 9-quality set.
    --slakh_dir  <path>  Slakh2100. Chord auto-extracted, full 12-quality set.
    --midi_dir   <path>  Generic MIDI. No chord/key extraction.
    --synthetic          Random toy songs (CI smoke test).

Melody extraction
-----------------
  Lakh:  --melody_method {weight,miner}     (default: weight)
  Slakh: --slakh_melody {instrument,miner,weight}  (default: instrument)
         'instrument' uses metadata.yaml inst_class labels (Synth Lead / Pipe /
         Reed / Brass) — near-GT quality.  No fallback for missing melody-class
         stems (those 27% of songs are skipped).

Parallelism
-----------
  --num_workers N  enables multiprocessing (default: 1 = single-threaded).
  Each worker loads the tokenizer and midi-miner models once via an initializer,
  then processes files independently.  Recommended: os.cpu_count() // 2.

Chord quality vocabulary (12 qualities, sus2 merged into add9)
---------------------------------------------------------------
Core 9  (indices 0-8):  maj min dom7 maj7 min7 dim aug add9 sus4
Extended 3 (indices 9-11, Slakh-tier): dim7 hdim7 dom9
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from jam_transformer.config import load_config
from jam_transformer.tokenizer import build_tokenizer
from jam_transformer.utils.logger import logger
from jam_transformer.preprocessing.shards import write_meta, _save_shard
from jam_transformer.preprocessing.melody import (
    _GM_BASS_PROGRAMS,
    _lakh_track_events,
    _load_mm_models,
    _DEFAULT_MM_MODELS,
)
from jam_transformer.preprocessing.synthetic import _synthesize_song
from jam_transformer.preprocessing.pop909 import _encode_one, _find_pop909_midis
from jam_transformer.preprocessing.lakh import _encode_lakh_one, _find_lakh_midis
from jam_transformer.preprocessing.slakh import _encode_slakh_one, _find_slakh_dirs

# ---------------------------------------------------------------------------
# Re-export symbols imported by tests/
# (tests/test_basics.py, test_dynamics.py, test_integration.py)
# ---------------------------------------------------------------------------
from jam_transformer.preprocessing.shards import tokenizer_fingerprint  # noqa: F401
# _synthesize_song, _lakh_track_events, _GM_BASS_PROGRAMS,
# _encode_lakh_one, write_meta  already imported above


# ---------------------------------------------------------------------------
# Parallel worker state (module-level for Windows spawn compatibility)
# ---------------------------------------------------------------------------

_W_TOKENIZER = None   # set once per worker process via _parallel_init
_W_MM_MODELS = None
_W_STATE: dict = {}   # holds all per-run params except the per-file path


def _parallel_init(config_path: str, state: dict) -> None:
    """ProcessPoolExecutor initializer: runs once per worker process.

    Loads the tokenizer and (if needed) midi-miner models into module globals
    so they are reused across all files processed by this worker.
    """
    global _W_TOKENIZER, _W_MM_MODELS, _W_STATE
    cfg = load_config(config_path)
    _W_TOKENIZER = build_tokenizer(cfg.tokenizer)
    _W_STATE = state
    if state.get("melody_method") == "miner" or state.get("slakh_melody") == "miner":
        _W_MM_MODELS = _load_mm_models(Path(state["mm_models"]))


def _lakh_parallel_fn(midi_path_str: str) -> "str | None":
    try:
        return _encode_lakh_one(
            Path(midi_path_str), _W_TOKENIZER,
            _W_STATE["cond_tracks"], _W_STATE["target_tracks"],
            Path(_W_STATE["out_dir"]),
            name_prefix=_W_STATE["lakh_prefix"],
            min_melody_coverage=_W_STATE["min_mel_cov"],
            min_stem_notes=_W_STATE["min_stem_notes"],
            chord_match_threshold=_W_STATE["chord_match_threshold"],
            mm_models=_W_MM_MODELS,
            miner_fallback=_W_STATE["miner_fallback"],
        )
    except Exception:
        return None


def _slakh_parallel_fn(song_dir_str: str) -> "str | None":
    try:
        return _encode_slakh_one(
            Path(song_dir_str), _W_TOKENIZER,
            _W_STATE["cond_tracks"], _W_STATE["target_tracks"],
            Path(_W_STATE["out_dir"]),
            name_prefix=_W_STATE["slakh_prefix"],
            min_melody_coverage=_W_STATE["min_mel_cov"],
            min_stem_notes=_W_STATE["min_stem_notes"],
            chord_match_threshold=_W_STATE["chord_match_threshold"],
            mm_models=_W_MM_MODELS,
            miner_fallback=_W_STATE["miner_fallback"],
            slakh_melody=_W_STATE["slakh_melody"],
        )
    except Exception:
        return None


def _run_parallel(worker_fn, items, num_workers: int, config_path: str,
                  state: dict, desc: str) -> tuple[list[str], int]:
    """Run worker_fn over items with ProcessPoolExecutor. Returns (stems, n_skip)."""
    new_stems: list[str] = []
    n_skip = 0
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_parallel_init,
        initargs=(config_path, state),
    ) as pool:
        chunksize = max(8, len(items) // (num_workers * 8))
        for stem in tqdm(
            pool.map(worker_fn, [str(x) for x in items], chunksize=chunksize),
            total=len(items), desc=desc,
        ):
            if stem:
                new_stems.append(stem)
            else:
                n_skip += 1
    return new_stems, n_skip


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize MIDI songs into .pt shards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_data.py --pop909_dir data/raw/POP909 --out_dir data/processed
  python scripts/prepare_data.py --lakh_dir   data/raw/lmd_clean --out_dir data/processed
  python scripts/prepare_data.py --slakh_dir  data/raw/slakh2100 --out_dir data/processed
  python scripts/prepare_data.py --synthetic  --out_dir data/processed
  python scripts/prepare_data.py --lakh_dir   data/raw/lmd_clean --out_dir data/processed_miner \\
      --melody_method miner --miner_fallback weight --num_workers 8
""",
    )
    parser.add_argument("--config",        type=str, default="configs/config.yaml")
    parser.add_argument("--out_dir",       type=str, default="data/processed")
    parser.add_argument("--pop909_dir",    type=str, default=None)
    parser.add_argument("--lakh_dir",      type=str, default=None)
    parser.add_argument("--slakh_dir",     type=str, default=None)
    parser.add_argument("--midi_dir",      type=str, default=None)
    parser.add_argument("--synthetic",     action="store_true")
    parser.add_argument("--num_songs",     type=int, default=32)
    parser.add_argument("--cond_tracks",   type=str, default="melody")
    parser.add_argument("--target_tracks", type=str, default="accompaniment")
    parser.add_argument("--pop909_prefix", type=str, default="pop909_")
    parser.add_argument("--lakh_prefix",   type=str, default="lakh_")
    parser.add_argument("--slakh_prefix",  type=str, default="slakh_")
    # Lakh melody extraction
    parser.add_argument("--melody_method", type=str, default="weight",
                        choices=["weight", "miner"],
                        help="Melody track selector for Lakh (default: weight)")
    parser.add_argument("--miner_fallback", type=str, default="weight",
                        choices=["weight", "skip"],
                        help="When miner fails: 'weight' fallback or 'skip' (default: weight)")
    # Slakh melody extraction
    parser.add_argument("--slakh_melody", type=str, default="instrument",
                        choices=["instrument", "miner", "weight"],
                        help="Melody selector for Slakh (default: instrument)")
    # midi-miner model files (track_separate must be pip-installed separately)
    parser.add_argument("--mm_models",  type=str, default=str(_DEFAULT_MM_MODELS),
                        help="Dir containing melody_model / bass_model / etc. "
                             "(default: C:\\Users\\hojun\\mm_models)")
    # parallelism
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Worker processes for Lakh/Slakh (default: 1). "
                             "Recommended: os.cpu_count()//2. Each worker loads "
                             "the tokenizer and midi-miner models independently.")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    tokenizer = build_tokenizer(cfg.tokenizer)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_tracks   = [t.strip() for t in args.cond_tracks.split(",")   if t.strip()]
    target_tracks = [t.strip() for t in args.target_tracks.split(",") if t.strip()]
    for t in cond_tracks + target_tracks:
        if t not in cfg.tokenizer.tracks:
            raise ValueError(f"Track '{t}' not in tokenizer.tracks={cfg.tokenizer.tracks}")

    prep                  = cfg.preprocessing
    min_mel_cov           = float(prep.min_melody_coverage)
    chord_match_threshold = float(prep.chord_match_threshold)
    min_stem_notes        = int(prep.min_stem_notes)

    logger.info(f"Vocab size     : {tokenizer.vocab_size}")
    logger.info(f"Cond tracks    : {cond_tracks}")
    logger.info(f"Target         : {target_tracks}")
    logger.info(f"Output dir     : {out_dir}")
    logger.info(f"Min melody cov : {min_mel_cov:.0%} "
                f"({'enabled' if min_mel_cov > 0 else 'disabled'})")
    logger.info(f"Num workers    : {args.num_workers}")

    # ── synthetic (always single-threaded — fast enough) ─────────────────────
    if args.synthetic:
        logger.info(f"Generating {args.num_songs} synthetic songs …")
        new_stems: list[str] = []
        for i in tqdm(range(args.num_songs)):
            events, tempo, kr, km = _synthesize_song(seed=i)
            ids, mask = tokenizer.encode_song(
                events, cond_tracks, target_tracks,
                tempo_bpm=tempo, key_root=kr, key_mode=km,
            )
            stem = f"synth_{i:04d}"
            _save_shard(out_dir, stem, ids, mask, key_root=kr, key_mode=km)
            new_stems.append(stem)
        write_meta(out_dir, tokenizer, cond_tracks, target_tracks, new_stems)
        logger.info(f"Wrote {len(new_stems)} synthetic shards.")
        return

    # Shared state dict passed to each worker via _parallel_init.
    state = dict(
        cond_tracks=cond_tracks, target_tracks=target_tracks,
        out_dir=str(out_dir),
        lakh_prefix=args.lakh_prefix, slakh_prefix=args.slakh_prefix,
        min_mel_cov=min_mel_cov, min_stem_notes=min_stem_notes,
        chord_match_threshold=chord_match_threshold,
        melody_method=args.melody_method, miner_fallback=args.miner_fallback,
        slakh_melody=args.slakh_melody,
        mm_models=args.mm_models,
    )

    # ── Lakh ─────────────────────────────────────────────────────────────────
    if args.lakh_dir:
        midis = _find_lakh_midis(Path(args.lakh_dir))
        if not midis:
            raise SystemExit(f"No MIDI files found under {args.lakh_dir}.")
        logger.info(f"Lakh: {len(midis):,} files — core 9-quality chord set. "
                    f"melody_method={args.melody_method}")

        if args.num_workers > 1:
            new_stems, n_skip = _run_parallel(
                _lakh_parallel_fn, midis, args.num_workers,
                args.config, state, desc="Lakh",
            )
        else:
            # single-threaded: load miner models in main process
            mm_models = None
            if args.melody_method == "miner":
                logger.info("Loading midi-miner models …")
                mm_models = _load_mm_models(Path(args.mm_models))
                logger.info(f"  done. fallback={args.miner_fallback}")
            new_stems, n_skip = [], 0
            for p in tqdm(midis):
                stem = _encode_lakh_one(
                    p, tokenizer, cond_tracks, target_tracks, out_dir,
                    name_prefix=args.lakh_prefix,
                    min_melody_coverage=min_mel_cov,
                    min_stem_notes=min_stem_notes,
                    chord_match_threshold=chord_match_threshold,
                    mm_models=mm_models,
                    miner_fallback=args.miner_fallback,
                )
                if stem: new_stems.append(stem)
                else:    n_skip += 1

        write_meta(out_dir, tokenizer, cond_tracks, target_tracks, new_stems)
        logger.info(f"Lakh: saved {len(new_stems):,}  skipped {n_skip:,}")
        return

    # ── Slakh ─────────────────────────────────────────────────────────────────
    if args.slakh_dir:
        song_dirs = _find_slakh_dirs(Path(args.slakh_dir))
        if not song_dirs:
            raise SystemExit(f"No all_src.mid files under {args.slakh_dir}.")
        logger.info(f"Slakh: {len(song_dirs):,} songs — full 12-quality chord set. "
                    f"slakh_melody={args.slakh_melody}")

        if args.num_workers > 1:
            new_stems, n_skip = _run_parallel(
                _slakh_parallel_fn, song_dirs, args.num_workers,
                args.config, state, desc="Slakh",
            )
        else:
            mm_models = None
            if args.slakh_melody == "miner":
                logger.info("Loading midi-miner models …")
                mm_models = _load_mm_models(Path(args.mm_models))
                logger.info("  done.")
            new_stems, n_skip = [], 0
            for d in tqdm(song_dirs):
                stem = _encode_slakh_one(
                    d, tokenizer, cond_tracks, target_tracks, out_dir,
                    name_prefix=args.slakh_prefix,
                    min_melody_coverage=min_mel_cov,
                    min_stem_notes=min_stem_notes,
                    chord_match_threshold=chord_match_threshold,
                    mm_models=mm_models,
                    miner_fallback=args.miner_fallback,
                    slakh_melody=args.slakh_melody,
                )
                if stem: new_stems.append(stem)
                else:    n_skip += 1

        write_meta(out_dir, tokenizer, cond_tracks, target_tracks, new_stems)
        logger.info(f"Slakh: saved {len(new_stems):,}  skipped {n_skip:,}")
        return

    # ── POP909 / generic (single-threaded — fast enough) ─────────────────────
    if args.pop909_dir:
        midis       = _find_pop909_midis(Path(args.pop909_dir))
        name_prefix = args.pop909_prefix
        logger.info(f"POP909: {len(midis)} files — GT chord/key annotations.")
    elif args.midi_dir:
        midis       = (sorted(Path(args.midi_dir).rglob("*.mid")) +
                       sorted(Path(args.midi_dir).rglob("*.midi")))
        name_prefix = ""
        logger.info(f"Generic MIDI: {len(midis)} files — no chord extraction.")
    else:
        raise SystemExit(
            "Provide one of: --synthetic / --pop909_dir / --lakh_dir / --slakh_dir / --midi_dir"
        )

    new_stems, n_skip = [], 0
    for p in tqdm(midis):
        stem = _encode_one(p, tokenizer, cond_tracks, target_tracks, out_dir,
                           name_prefix=name_prefix,
                           min_melody_coverage=min_mel_cov)
        if stem: new_stems.append(stem)
        else:    n_skip += 1
    write_meta(out_dir, tokenizer, cond_tracks, target_tracks, new_stems)
    logger.info(f"Saved {len(new_stems)}/{len(midis)}  skipped {n_skip}")


if __name__ == "__main__":
    main()
