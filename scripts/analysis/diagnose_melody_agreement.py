"""Melody inter-method agreement diagnostic.

Compares weight-based melody selection (prepare_data.py heuristic) against
midi-miner RandomForest classifier on sampled files from each dataset source.

For POP909 (GT labels available), also computes accuracy vs ground truth.

Track identity is based on (note_count, median_pitch) fingerprint — NOT GM
program number — because POP909 tracks all share program=0 (piano), making
program comparison useless there.

Usage:
    python scripts/diagnose_melody_agreement.py [--n 200] [--seed 42] [--out results.txt]
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
MM_MODELS  = Path(r"C:\Users\hojun\mm_models")

sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import track_separate as ts
import logging as _lg
ts.logger = _lg.getLogger("mm")
ts.logger.addHandler(_lg.NullHandler())

from jam_transformer.config import load_config

cfg = load_config(str(REPO_ROOT / "configs" / "config.yaml"))

_GM_BASS_PROGRAMS: frozenset[int] = frozenset(range(32, 40))
_GM_MELODY_HINT_PROGRAMS: frozenset[int] = frozenset(
    list(range(40, 44)) + list(range(56, 64)) +
    list(range(64, 80)) + list(range(80, 88))
)

# Track identity: (note_count, median_pitch_int)
Fingerprint = tuple[int, int]


def _make_fp(notes) -> Fingerprint:
    """Build a (note_count, median_pitch) fingerprint from a list of note objects."""
    pitches = sorted(n.pitch for n in notes)
    return (len(pitches), pitches[len(pitches) // 2])


# ── midi-miner model load ──────────────────────────────────────────────────

def _load_mm_models():
    names = ("melody_model", "bass_model", "chord_model", "drum_model")
    models = []
    for name in names:
        m = pickle.load(open(MM_MODELS / name, "rb"))
        for est in m.estimators_:
            if not hasattr(est, "monotonic_cst"):
                est.monotonic_cst = None
        models.append(m)
    return models   # [melody_model, bass_model, chord_model, drum_model]


# ── weight-based selection (prepare_data.py heuristic) ───────────────────

def _weight_select_fingerprint(midi_path: Path, min_notes: int = 4) -> Optional[Fingerprint]:
    """Return fingerprint of the weight-selected melody instrument, or None."""
    try:
        import miditoolkit
        midi = miditoolkit.MidiFile(str(midi_path))
    except Exception:
        return None

    tpb = midi.ticks_per_beat or 480

    non_drum = [i for i in midi.instruments if not i.is_drum and len(i.notes) >= min_notes]
    if len(non_drum) < 2:
        return None

    def _med_pitch(inst):
        p = sorted(n.pitch for n in inst.notes)
        return float(p[len(p) // 2])

    def _mono_rate(inst):
        notes = sorted(inst.notes, key=lambda n: n.start)
        if len(notes) < 2:
            return 1.0
        return sum(1 for a, b in zip(notes, notes[1:]) if a.end <= b.start) / (len(notes) - 1)

    def _density(inst):
        span = max(n.end for n in inst.notes) - min(n.start for n in inst.notes)
        beats = span / max(tpb, 1)
        return len(inst.notes) / max(beats, 1.0)

    bass    = [i for i in non_drum if i.program in _GM_BASS_PROGRAMS]
    melodic = [i for i in non_drum if i.program not in _GM_BASS_PROGRAMS]
    if len(melodic) < 2:
        melodic = non_drum

    mps = [_med_pitch(i) for i in melodic]
    mrs = [_mono_rate(i) for i in melodic]
    mds = [_density(i)   for i in melodic]
    lo_p, hi_p = min(mps), max(mps)
    lo_d, hi_d = min(mds), max(mds)
    pr = max(hi_p - lo_p, 1.0)
    dr = max(hi_d - lo_d, 1.0)

    def _score(idx):
        return (0.10 * (mps[idx] - lo_p) / pr
                + 0.80 * mrs[idx]
                + 0.05 * (1 - (mds[idx] - lo_d) / dr)
                + 0.05 * (1 if melodic[idx].program in _GM_MELODY_HINT_PROGRAMS else 0))

    best_idx = max(range(len(melodic)), key=_score)
    return _make_fp(melodic[best_idx].notes)


# ── midi-miner selection ───────────────────────────────────────────────────

def _miner_select_fingerprint(midi_path: Path, mm_models: list) -> Optional[Fingerprint]:
    """Return fingerprint of the midi-miner-selected melody instrument, or None.

    The fingerprint is derived from the PrettyMIDI instrument object at the
    row index predicted as melody.  Row index i in the DataFrame corresponds
    to pm.instruments[i] (cal_file_features preserves instrument order).
    """
    try:
        feats, pm = ts.cal_file_features(str(midi_path))
    except Exception:
        return None
    if feats is None or pm is None:
        return None
    try:
        df = ts.add_labels(feats)
        df = ts.predict_labels(df, *mm_models)
    except Exception:
        return None

    mel_rows = df[df["melody_predict"] == True]
    if len(mel_rows) == 0:
        return None
    row_idx = int(mel_rows.index[0])
    if row_idx >= len(pm.instruments):
        return None
    inst = pm.instruments[row_idx]
    if not inst.notes:
        return None
    return _make_fp(inst.notes)


# ── POP909 ground-truth selection ──────────────────────────────────────────

def _gt_fingerprint(midi_path: Path) -> Optional[Fingerprint]:
    """Return fingerprint of the GT melody track (POP909 only, track name 'MELODY')."""
    try:
        import miditoolkit
        midi = miditoolkit.MidiFile(str(midi_path))
    except Exception:
        return None
    for inst in midi.instruments:
        if inst.name.strip().upper() == "MELODY" and inst.notes:
            return _make_fp(inst.notes)
    return None


# ── per-source evaluation ─────────────────────────────────────────────────

def _evaluate_source(
    paths: list[Path],
    mm_models: list,
    has_gt: bool = False,
    label: str = "",
) -> dict:
    n_total    = len(paths)
    agree      = 0
    disagree   = 0
    w_ok_gt    = 0  # weight matches GT
    m_ok_gt    = 0  # miner  matches GT
    w_gt_valid = 0
    m_gt_valid = 0
    skipped    = 0

    for p in paths:
        w = _weight_select_fingerprint(p)
        m = _miner_select_fingerprint(p, mm_models)

        if w is None or m is None:
            skipped += 1
            continue

        if w == m:
            agree += 1
        else:
            disagree += 1

        if has_gt:
            gt = _gt_fingerprint(p)
            if gt is not None:
                w_gt_valid += 1
                if w == gt:
                    w_ok_gt += 1
                m_gt_valid += 1
                if m == gt:
                    m_ok_gt += 1

    valid = agree + disagree
    return {
        "source":    label,
        "n_sample":  n_total,
        "n_valid":   valid,
        "skipped":   skipped,
        "agree":     agree,
        "disagree":  disagree,
        "agree_pct": 100.0 * agree / valid if valid else float("nan"),
        "w_acc_pct": 100.0 * w_ok_gt / w_gt_valid if w_gt_valid else float("nan"),
        "m_acc_pct": 100.0 * m_ok_gt / m_gt_valid if m_gt_valid else float("nan"),
        "has_gt":    has_gt,
    }


# ── sampling helpers ───────────────────────────────────────────────────────

def _sample_lakh(n: int, seed: int) -> list[Path]:
    root = REPO_ROOT / "data" / "raw" / "lmd_clean"
    if not root.exists():
        print(f"[WARN] Lakh dir not found: {root}")
        return []
    all_mid = list(root.rglob("*.mid"))
    rng = random.Random(seed)
    return rng.sample(all_mid, min(n, len(all_mid)))


def _sample_slakh(n: int, seed: int) -> list[Path]:
    root = REPO_ROOT / "data" / "raw" / "slakh2100"
    if not root.exists():
        print(f"[WARN] Slakh dir not found: {root}")
        return []
    all_mid = list(root.rglob("all_src.mid"))
    rng = random.Random(seed)
    return rng.sample(all_mid, min(n, len(all_mid)))


def _sample_pop909(n: int, seed: int) -> list[Path]:
    root = REPO_ROOT / "data" / "raw" / "POP909"
    if not root.exists():
        print(f"[WARN] POP909 dir not found: {root}")
        return []
    # Only the main per-song MIDI (e.g. 001/001.mid), not _cl.mid or versions/.
    all_mid = [
        p for p in root.rglob("*.mid")
        if "versions" not in str(p).lower() and "_cl." not in p.name
    ]
    rng = random.Random(seed)
    return rng.sample(all_mid, min(n, len(all_mid)))


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Melody method agreement diagnostic.")
    ap.add_argument("--n",    type=int,  default=200, help="Samples per source (default: 200)")
    ap.add_argument("--seed", type=int,  default=42,  help="Random seed")
    ap.add_argument("--out",  type=str,  default=None, help="Optional output .txt file")
    args = ap.parse_args()

    print("Loading midi-miner models …")
    mm_models = _load_mm_models()
    print("  done.\n")

    results = []

    for label, paths, has_gt in [
        ("POP909", _sample_pop909(args.n, args.seed), True),
        ("Lakh",   _sample_lakh(  args.n, args.seed), False),
        ("Slakh",  _sample_slakh( args.n, args.seed), False),
    ]:
        if not paths:
            continue
        print(f"Evaluating {label} ({len(paths)} files) …")
        r = _evaluate_source(paths, mm_models, has_gt=has_gt, label=label)
        results.append(r)

    # ── print table ─────────────────────────────────────────────────────────
    lines = []
    lines.append("\n── Melody Selection Agreement: Weight vs midi-miner ──────────────────────")
    lines.append(f"{'Source':<10} {'n_sample':>8} {'n_valid':>8} {'skipped':>8} "
                 f"{'agree%':>8} {'w_acc%':>8} {'m_acc%':>8}")
    lines.append("-" * 70)
    for r in results:
        agree_s = f"{r['agree_pct']:7.1f}" if r["n_valid"] > 0 else "    N/A"
        w_acc_s = f"{r['w_acc_pct']:7.1f}" if r["has_gt"] and r["n_valid"] > 0 else "    N/A"
        m_acc_s = f"{r['m_acc_pct']:7.1f}" if r["has_gt"] and r["n_valid"] > 0 else "    N/A"
        lines.append(f"{r['source']:<10} {r['n_sample']:>8} {r['n_valid']:>8} {r['skipped']:>8} "
                     f"{agree_s}% {w_acc_s}% {m_acc_s}%")
    lines.append("-" * 70)
    lines.append("")
    lines.append("agree%  = fraction of valid files where both methods pick the same track")
    lines.append("          (identity = note_count + median_pitch fingerprint)")
    lines.append("w_acc%  = weight method accuracy vs POP909 GT  (only POP909 row)")
    lines.append("m_acc%  = miner  method accuracy vs POP909 GT  (only POP909 row)")
    lines.append("")
    lines.append("Code comment reference accuracy: weight=94.6%, miner=95.8% (POP909 GT)")

    output = "\n".join(lines)
    print(output)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
