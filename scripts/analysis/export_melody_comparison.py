"""Export melody method comparison MIDIs for listening test.

For each sampled POP909 file, extracts:
  {song}_full.mid    — original (all tracks)
  {song}_gt.mid      — GT MELODY track only
  {song}_weight.mid  — weight-heuristic selected track only
  {song}_miner.mid   — midi-miner selected track only

Output is organized into agree/ and disagree/ subdirs.
After running, convert to WAV with Docker (see printed instructions at end).

Usage:
    python scripts/export_melody_comparison.py [--n 200] [--seed 42] [--max_disagree 6] [--max_agree 3]
"""
from __future__ import annotations

import argparse
import copy
import pickle
import random
import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
MM_MODELS = Path(r"C:\Users\hojun\mm_models")

sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import track_separate as ts
import logging as _lg
ts.logger = _lg.getLogger("mm")
ts.logger.addHandler(_lg.NullHandler())

_GM_BASS_PROGRAMS: frozenset[int] = frozenset(range(32, 40))
_GM_MELODY_HINT_PROGRAMS: frozenset[int] = frozenset(
    list(range(40, 44)) + list(range(56, 64)) +
    list(range(64, 80)) + list(range(80, 88))
)

Fingerprint = tuple[int, int]  # (note_count, median_pitch)


def _fp(notes) -> Optional[Fingerprint]:
    pitches = sorted(n.pitch for n in notes)
    if not pitches:
        return None
    return (len(pitches), pitches[len(pitches) // 2])


def _load_mm_models():
    names = ("melody_model", "bass_model", "chord_model", "drum_model")
    models = []
    for name in names:
        m = pickle.load(open(MM_MODELS / name, "rb"))
        for est in m.estimators_:
            if not hasattr(est, "monotonic_cst"):
                est.monotonic_cst = None
        models.append(m)
    return models


def _weight_select_fp(midi, tpb, min_notes: int = 4) -> Optional[Fingerprint]:
    """Weight heuristic on an already-loaded miditoolkit MidiFile."""
    non_drum = [i for i in midi.instruments if not i.is_drum and len(i.notes) >= min_notes]
    if len(non_drum) < 2:
        return None

    def _med(inst): p = sorted(n.pitch for n in inst.notes); return float(p[len(p)//2])
    def _mono(inst):
        ns = sorted(inst.notes, key=lambda n: n.start)
        return 1.0 if len(ns)<2 else sum(1 for a,b in zip(ns,ns[1:]) if a.end<=b.start)/(len(ns)-1)
    def _dens(inst):
        span = max(n.end for n in inst.notes)-min(n.start for n in inst.notes)
        return len(inst.notes)/max(span/max(tpb,1), 1.0)

    melodic = [i for i in non_drum if i.program not in _GM_BASS_PROGRAMS]
    if len(melodic) < 2:
        melodic = non_drum

    mps = [_med(i) for i in melodic]
    mrs = [_mono(i) for i in melodic]
    mds = [_dens(i) for i in melodic]
    lo_p, hi_p = min(mps), max(mps)
    lo_d, hi_d = min(mds), max(mds)
    pr = max(hi_p-lo_p, 1.0); dr = max(hi_d-lo_d, 1.0)

    def _score(idx):
        return (0.10*(mps[idx]-lo_p)/pr + 0.80*mrs[idx]
                + 0.05*(1-(mds[idx]-lo_d)/dr)
                + 0.05*(1 if melodic[idx].program in _GM_MELODY_HINT_PROGRAMS else 0))

    best = max(range(len(melodic)), key=_score)
    return _fp(melodic[best].notes)


def _miner_select_fp(midi_path: Path, mm_models: list) -> Optional[Fingerprint]:
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
    mel = df[df["melody_predict"] == True]
    if len(mel) == 0:
        return None
    idx = int(mel.index[0])
    if idx >= len(pm.instruments):
        return None
    return _fp(pm.instruments[idx].notes)


def _evaluate(midi_path: Path, mm_models: list) -> dict:
    """Return dict with keys: gt_fp, w_fp, m_fp, agree, gt_ok_w, gt_ok_m."""
    import miditoolkit
    try:
        midi = miditoolkit.MidiFile(str(midi_path))
    except Exception:
        return {}
    tpb = midi.ticks_per_beat or 480

    gt_inst = next((i for i in midi.instruments
                    if i.name.strip().upper() == "MELODY" and i.notes), None)
    gt_fp = _fp(gt_inst.notes) if gt_inst else None

    w_fp = _weight_select_fp(midi, tpb)
    m_fp = _miner_select_fp(midi_path, mm_models)

    if w_fp is None or m_fp is None:
        return {}

    return {
        "gt_fp":   gt_fp,
        "w_fp":    w_fp,
        "m_fp":    m_fp,
        "agree":   w_fp == m_fp,
        "gt_ok_w": gt_fp is not None and w_fp == gt_fp,
        "gt_ok_m": gt_fp is not None and m_fp == gt_fp,
        "midi":    midi,
        "tpb":     tpb,
    }


def _export_one(midi_path: Path, ev: dict, out_dir: Path):
    """Write full/gt/weight/miner MIDI files to out_dir."""
    import miditoolkit
    out_dir.mkdir(parents=True, exist_ok=True)
    midi = ev["midi"]
    stem = midi_path.stem

    # Map fingerprint → miditoolkit instrument
    fp_to_inst: dict = {}
    for inst in midi.instruments:
        f = _fp(inst.notes)
        if f is not None and f not in fp_to_inst:
            fp_to_inst[f] = inst

    # Full
    midi.dump(str(out_dir / f"{stem}_full.mid"))

    for tag, fp_key in [("gt", "gt_fp"), ("weight", "w_fp"), ("miner", "m_fp")]:
        fp_val = ev.get(fp_key)
        if fp_val is None:
            continue
        inst = fp_to_inst.get(fp_val)
        if inst is None:
            continue
        new_mid = copy.deepcopy(midi)
        new_mid.instruments = [inst]
        new_mid.dump(str(out_dir / f"{stem}_{tag}.mid"))

    # Write a README for this song
    verdict = []
    if ev["gt_fp"] is not None:
        verdict.append(f"weight vs GT : {'CORRECT' if ev['gt_ok_w'] else 'WRONG'}")
        verdict.append(f"miner  vs GT : {'CORRECT' if ev['gt_ok_m'] else 'WRONG'}")
    verdict.append(f"weight vs miner : {'AGREE' if ev['agree'] else 'DISAGREE'}")
    (out_dir / f"{stem}_notes.txt").write_text("\n".join(verdict) + "\n", encoding="utf-8")


def _sample_paths(source: str, n: int, seed: int) -> list[Path]:
    if source == "pop909":
        root = REPO_ROOT / "data" / "raw" / "POP909"
        if not root.exists():
            print(f"[WARN] POP909 not found: {root}"); return []
        all_mid = [p for p in root.rglob("*.mid")
                   if "versions" not in str(p).lower() and "_cl." not in p.name]
    elif source == "lakh":
        root = REPO_ROOT / "data" / "raw" / "lmd_clean"
        if not root.exists():
            print(f"[WARN] Lakh not found: {root}"); return []
        all_mid = list(root.rglob("*.mid"))
    elif source == "slakh":
        root = REPO_ROOT / "data" / "raw" / "slakh2100"
        if not root.exists():
            print(f"[WARN] Slakh not found: {root}"); return []
        all_mid = list(root.rglob("all_src.mid"))
    else:
        raise ValueError(f"Unknown source: {source}")
    rng = random.Random(seed)
    return rng.sample(all_mid, min(n, len(all_mid)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",      type=str, default="pop909",
                    choices=["pop909", "lakh", "slakh"],
                    help="Dataset to sample from (default: pop909)")
    ap.add_argument("--n",           type=int, default=500,
                    help="Files to scan (default: 500; use more for lakh/slakh due to high skip rate)")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--max_disagree",type=int, default=6)
    ap.add_argument("--max_agree",   type=int, default=3)
    ap.add_argument("--out_dir",     type=str, default="data/melody_comparison")
    args = ap.parse_args()

    has_gt = (args.source == "pop909")
    out_base = REPO_ROOT / args.out_dir / args.source

    print("Loading midi-miner models …")
    mm_models = _load_mm_models()
    print("  done.\n")

    sample = _sample_paths(args.source, args.n, args.seed)
    if not sample:
        return

    agree_cases    = []
    disagree_cases = []
    scanned = 0

    print(f"Scanning {len(sample)} {args.source} files …")
    for p in sample:
        if len(agree_cases) >= args.max_agree and len(disagree_cases) >= args.max_disagree:
            break
        ev = _evaluate(p, mm_models)
        scanned += 1
        if not ev:
            continue
        if ev["agree"] and len(agree_cases) < args.max_agree:
            agree_cases.append((p, ev))
        elif not ev["agree"] and len(disagree_cases) < args.max_disagree:
            disagree_cases.append((p, ev))

    print(f"  scanned {scanned}, selected: {len(agree_cases)} agree + {len(disagree_cases)} disagree\n")

    def _song_id(p: Path) -> str:
        # all_src.mid (Slakh) → use parent dir name (Track00001, etc.)
        return p.parent.name if p.stem == "all_src" else p.stem

    # Export
    for p, ev in agree_cases:
        _export_one(p, ev, out_base / "agree" / _song_id(p))
    for p, ev in disagree_cases:
        _export_one(p, ev, out_base / "disagree" / _song_id(p))

    # Summary
    print("── Exported files ───────────────────────────────────────────")
    for category, cases in [("agree", agree_cases), ("disagree", disagree_cases)]:
        for p, ev in cases:
            sid = _song_id(p)
            if has_gt:
                w_tag = "W:OK" if ev["gt_ok_w"] else "W:NG"
                m_tag = "M:OK" if ev["gt_ok_m"] else "M:NG"
                print(f"  [{category:8}] {sid}  {w_tag} {m_tag}")
            else:
                print(f"  [{category:8}] {sid}  (no GT)")

    print(f"\nMIDIs saved to: {out_base}")


if __name__ == "__main__":
    main()
