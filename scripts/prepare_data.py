"""MIDI → tokenized .pt shards (relative harmonic encoding).

Modes:
    --pop909_dir <path>  POP909. GT chord/key from annotations/.
    --lakh_dir   <path>  Lakh MIDI Clean. Chord auto-extracted, core 9-quality set.
    --slakh_dir  <path>  Slakh2100. Chord auto-extracted, full 12-quality set.
    --midi_dir   <path>  Generic MIDI. No chord/key extraction.
    --synthetic          Random toy songs (CI smoke test).

Chord quality vocabulary (12 qualities, sus2 merged into add9)
---------------------------------------------------------------
Core 9  (indices 0-8):  maj min dom7 maj7 min7 dim aug add9 sus4
Extended 3 (indices 9-11, Slakh-tier): dim7 hdim7 dom9

chord_map format
----------------
  dict[(bar: int, pos_resolution_units: int), (chord_root_0_11, quality_idx) | None]
  None  → CHORD_N (unknown / no chord)
  (r,q) → chord root 0-11 and quality index into CHORD_QUALITIES

The tokenizer converts (root, quality_idx) → SCALE_DEGREE + QUALITY tokens
using the piece's key_root at encode time.

Chord extraction
----------------
Template matching: score = |pitch_classes ∩ template| / |template|.
Threshold: 0.75.  n_qualities controls the candidate pool:
  Lakh:  9 (core only — fewer false positives from passing tones)
  Slakh: 12 (full set — cleaner multi-track harmonic signal)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from jam_transformer.config import load_config
from jam_transformer.tokenizer import (
    BaseTokenizer,
    NoteEvent,
    REMITokenizer,
    build_tokenizer,
)
from jam_transformer.logger import logger
from jam_transformer.midi_io import midi_to_events


META_FILENAME = "_dataset_meta.json"

_GM_BASS_PROGRAMS: frozenset[int] = frozenset(range(32, 40))
_GM_MELODY_HINT_PROGRAMS: frozenset[int] = frozenset(
    list(range(40, 44)) + list(range(56, 64)) +
    list(range(64, 80)) + list(range(80, 88))
)

# ---------------------------------------------------------------------------
# Chord template matching
# ---------------------------------------------------------------------------

_CHORD_TEMPLATES: dict[int, frozenset[int]] = {
    0:  frozenset([0, 4, 7]),           # maj
    1:  frozenset([0, 3, 7]),           # min
    2:  frozenset([0, 4, 7, 10]),       # dom7
    3:  frozenset([0, 4, 7, 11]),       # maj7
    4:  frozenset([0, 3, 7, 10]),       # min7
    5:  frozenset([0, 3, 6]),           # dim
    6:  frozenset([0, 4, 8]),           # aug
    7:  frozenset([0, 2, 4, 7]),        # add9  (sus2 collapses here)
    8:  frozenset([0, 5, 7]),           # sus4
    9:  frozenset([0, 3, 6, 9]),        # dim7
    10: frozenset([0, 3, 6, 10]),       # hdim7
    11: frozenset([0, 4, 7, 10, 2]),    # dom9
}

_MATCH_THRESHOLD = 0.75


def _match_chord(
    pitch_classes: frozenset[int],
    n_qualities: int,
) -> "tuple[int, int] | None":
    """Return (root_0_11, quality_idx) for the best-matching chord above threshold.

    Returns None when no candidate reaches _MATCH_THRESHOLD.
    n_qualities: 9 for Lakh (core), 12 for Slakh (full).
    """
    if len(pitch_classes) < 2:
        return None

    best_score = _MATCH_THRESHOLD - 1e-9
    best: "tuple[int, int] | None" = None

    for root in range(12):
        relative = frozenset((p - root) % 12 for p in pitch_classes)
        for q_idx, template in _CHORD_TEMPLATES.items():
            if q_idx >= n_qualities:
                continue
            score = len(relative & template) / len(template)
            if score > best_score:
                best_score = score
                best = (root, q_idx)

    return best


def _extract_chords_from_midi(
    midi_path: Path,
    cfg,
    n_qualities: int,
) -> "dict[tuple[int, int], tuple[int, int] | None]":
    """Beat-level chord extraction → (bar, pos) chord map.

    Only positions with a confident match (score ≥ _MATCH_THRESHOLD) are added.
    """
    try:
        import miditoolkit
        midi = miditoolkit.MidiFile(str(midi_path))
    except Exception as exc:
        logger.debug(f"Chord extraction failed for {midi_path.name}: {exc}")
        return {}

    tpb = midi.ticks_per_beat
    all_notes: list[tuple[int, int, int]] = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            all_notes.append((n.start, n.end, n.pitch))

    if not all_notes:
        return {}

    max_beat      = int(max(n[1] for n in all_notes) / tpb) + 1
    beats_per_bar = 4
    # Guard: skip chord extraction for pathologically long/dense files.
    # 3000 beats ≈ 12 min at 120 BPM. Beyond this the O(beats×notes) scan
    # can run for hours on large Lakh files.
    _MAX_BEATS = 3000
    if max_beat > _MAX_BEATS or len(all_notes) > 50_000:
        logger.debug(
            f"{midi_path.name}: skipping chord extraction "
            f"(max_beat={max_beat}, notes={len(all_notes)})"
        )
        return {}

    # Sort notes once; scan with a two-pointer per beat for O(n log n) total.
    all_notes.sort()
    chord_map: dict[tuple[int, int], tuple[int, int] | None] = {}
    for beat in range(max_beat):
        beat_start = beat * tpb
        beat_end   = (beat + 1) * tpb
        active_pcs = frozenset(
            n[2] % 12 for n in all_notes
            if n[0] < beat_end and n[1] > beat_start
        )
        result = _match_chord(active_pcs, n_qualities)
        if result is not None:
            bar = beat // beats_per_bar
            pos = (beat % beats_per_bar) * cfg.resolution
            chord_map[(bar, pos)] = result

    return chord_map


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def tokenizer_fingerprint(cfg) -> str:
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True).encode()
    ).hexdigest()[:16]


def write_meta(
    out_dir: Path,
    tokenizer: BaseTokenizer,
    cond_tracks: list[str],
    target_tracks: list[str],
    new_shard_names: Optional[list[str]] = None,
) -> None:
    index_path = out_dir / "_chunk_index.json"
    if new_shard_names is not None:
        shard_lens: dict[str, int] = {}
        if index_path.exists():
            try:
                shard_lens = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                shard_lens = {}
        for stem in new_shard_names:
            p = out_dir / f"{stem}.pt"
            if p.exists():
                data = torch.load(p, map_location="cpu", weights_only=False)
                mask = data["mask"]
                nz   = mask.nonzero()
                sep  = int(nz[0].item()) if nz.numel() > 0 else int(mask.numel())
                shard_lens[f"{stem}.pt"] = {"n": int(data["ids"].numel()), "sep": sep}
    else:
        shard_lens = {}
        for p in sorted(out_dir.glob("*.pt")):
            if p.name.startswith("_"):
                continue
            data = torch.load(p, map_location="cpu", weights_only=False)
            mask = data["mask"]
            nz   = mask.nonzero()
            sep  = int(nz[0].item()) if nz.numel() > 0 else int(mask.numel())
            shard_lens[p.name] = {"n": int(data["ids"].numel()), "sep": sep}

    meta = {
        "vocab_size":             tokenizer.vocab_size,
        "tokenizer_config":       asdict(tokenizer.cfg),
        "tokenizer_fingerprint":  tokenizer_fingerprint(tokenizer.cfg),
        "n_shards":               len(shard_lens),
        "cond_tracks":            cond_tracks,
        "target_tracks":          target_tracks,
    }
    (out_dir / META_FILENAME).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    index_path.write_text(
        json.dumps(shard_lens, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Shard I/O  (now includes key_root / key_mode)
# ---------------------------------------------------------------------------

def _save_shard(
    out_dir: Path,
    name: str,
    ids: list[int],
    mask: list[bool],
    key_root: int = -1,
    key_mode: int = -1,
) -> None:
    torch.save(
        {
            "ids":      torch.tensor(ids,  dtype=torch.long),
            "mask":     torch.tensor(mask, dtype=torch.bool),
            "name":     name,
            "key_root": key_root,
            "key_mode": key_mode,
        },
        out_dir / f"{name}.pt",
    )


def _safe_name(p: Path) -> str:
    return p.stem.replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# POP909 chord / key annotation parsers
# ---------------------------------------------------------------------------

_HARTE_ROOT_TO_SEMITONE: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

# Full 12-quality Harte mapping. sus2 → add9 (index 7).
_HARTE_QUALITY_TO_IDX_FULL: dict[str, int] = {
    "maj": 0, "M": 0, "": 0,
    "min": 1, "m": 1,
    "7": 2, "dom7": 2,
    "maj7": 3, "M7": 3,
    "min7": 4, "m7": 4,
    "dim": 5, "o": 5,
    "aug": 6, "+": 6,
    "add9": 7, "2": 7, "sus2": 7,   # sus2 → add9
    "sus4": 8, "sus": 8,
    "dim7": 9, "o7": 9,
    "hdim7": 10, "m7b5": 10,
    "9": 11, "dom9": 11,
}

# Core 9-quality mapping for Lakh (indices 0-8 only).
_HARTE_QUALITY_TO_IDX_CORE: dict[str, int] = {
    "maj": 0, "M": 0, "": 0,
    "min": 1, "m": 1,
    "7": 2, "dom7": 2,
    "maj7": 3, "M7": 3,
    "min7": 4, "m7": 4,
    "dim": 5, "dim7": 5, "hdim7": 5, "o": 5, "o7": 5, "m7b5": 5,
    "aug": 6, "+": 6,
    "add9": 7, "2": 7, "sus2": 7,
    "sus4": 8, "sus": 8,
}


def _parse_pop909_chord_file(
    chord_file: Path,
    n_qualities: int,
    midi_path: "Path | None" = None,
    tempo_bpm: float = 120.0,
) -> "dict[tuple[int, int], tuple[int, int] | None]":
    """Parse POP909 chord_midi.txt → (bar, pos) chord map.

    POP909 format: start_sec  end_sec  chord_name  (tab-separated)
    e.g. "2.721993  4.055323  B:maj"

    When midi_path is provided, uses miditoolkit's tick_to_time mapping to
    accurately handle tempo changes (40% of POP909 songs have multiple tempos).
    Falls back to single-tempo approximation if midi_path is None or unreadable.

    Returns (root, quality_idx) for known chords, None for N/X chords.
    """
    import numpy as np

    chord_map: dict[tuple[int, int], tuple[int, int] | None] = {}
    try:
        lines = chord_file.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.debug(f"Cannot read {chord_file}: {exc}")
        return chord_map

    beats_per_bar = 4
    resolution    = 4   # sixteenth notes per beat

    # Build precise seconds→beat mapping from MIDI tempo changes.
    tick_to_time: "np.ndarray | None" = None
    tpb = 480  # default; overwritten below
    if midi_path is not None:
        try:
            import miditoolkit
            _mid = miditoolkit.MidiFile(str(midi_path))
            tick_to_time = _mid.get_tick_to_time_mapping()
            tpb = _mid.ticks_per_beat
        except Exception:
            tick_to_time = None

    def _sec_to_bar_pos(sec: float) -> "tuple[int, int]":
        if tick_to_time is not None:
            tick = int(np.searchsorted(tick_to_time, sec, side="left"))
            tick = min(tick, len(tick_to_time) - 1)
            beat_idx = tick // tpb
        else:
            sec_per_beat = 60.0 / max(tempo_bpm, 1.0)
            beat_idx = round(sec / sec_per_beat)
        bar = beat_idx // beats_per_bar
        pos = (beat_idx % beats_per_bar) * resolution
        return bar, pos

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start_sec = float(parts[0])
        except ValueError:
            continue
        chord_str = parts[2]

        bar, pos = _sec_to_bar_pos(start_sec)

        if chord_str in ("N", "X", ""):
            chord_map[(bar, pos)] = None
            continue

        root_str, quality_str = (chord_str.split(":", 1) if ":" in chord_str
                                 else (chord_str, ""))
        quality_str = quality_str.split("/")[0]

        root   = _HARTE_ROOT_TO_SEMITONE.get(root_str, -1)
        q_map  = _HARTE_QUALITY_TO_IDX_FULL if n_qualities >= 12 else _HARTE_QUALITY_TO_IDX_CORE
        q_idx  = q_map.get(quality_str, -1)

        if root < 0 or q_idx < 0 or q_idx >= n_qualities:
            chord_map[(bar, pos)] = None
        else:
            chord_map[(bar, pos)] = (root, q_idx)

    return chord_map


def _parse_pop909_key_file(
    key_file: Path,
) -> "tuple[int, int] | None":
    """Parse POP909 key_audio.txt → (root_0_11, mode_0_1) or None.

    POP909 format: start_sec  end_sec  root:mode  (tab-separated)
    e.g. "2.670294  191.982585  Gb:maj"
    Returns the first (dominant) key found.
    """
    _KEY_ROOT = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "Fb": 4, "F": 5, "F#": 6, "Gb": 6,
        "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
        "B": 11, "Cb": 11,
    }
    _MODE = {"major": 0, "maj": 0, "M": 0, "minor": 1, "min": 1, "m": 1}

    try:
        lines = key_file.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.debug(f"Cannot read {key_file}: {exc}")
        return None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # POP909 format: start_sec end_sec root:mode  (3 fields)
        # Fallback: root:mode  (1-2 fields, legacy)
        if len(parts) >= 3 and ":" in parts[2]:
            key_str = parts[2]
        elif len(parts) >= 2 and ":" in parts[1]:
            key_str = parts[1]
        elif len(parts) >= 1 and ":" in parts[0]:
            key_str = parts[0]
        else:
            continue
        root_str, mode_str = key_str.split(":", 1)
        root = _KEY_ROOT.get(root_str, -1)
        mode = _MODE.get(mode_str.lower(), -1)
        if root >= 0 and mode >= 0:
            return (root, mode)

    return None


# ---------------------------------------------------------------------------
# Key auto-extraction (Lakh / Slakh)
# ---------------------------------------------------------------------------

def _estimate_key_from_midi(midi_path: Path) -> "tuple[int, int] | None":
    """Estimate key via pitch-class profile (Krumhansl-Schmuckler heuristic).

    Returns (root_0_11, mode) or None on failure.
    """
    try:
        import miditoolkit
        midi = miditoolkit.MidiFile(str(midi_path))
    except Exception:
        return None

    pc_count = [0.0] * 12
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            pc_count[n.pitch % 12] += (n.end - n.start)

    total = sum(pc_count)
    if total <= 0:
        return None
    pc = [x / total for x in pc_count]

    # Krumhansl-Kessler profiles
    major = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    def _corr(profile: list[float], shift: int) -> float:
        rotated = profile[shift:] + profile[:shift]
        mean_p = sum(pc) / 12
        mean_r = sum(rotated) / 12
        num = sum((pc[i] - mean_p) * (rotated[i] - mean_r) for i in range(12))
        dp  = sum((pc[i] - mean_p) ** 2 for i in range(12)) ** 0.5
        dr  = sum((rotated[i] - mean_r) ** 2 for i in range(12)) ** 0.5
        if dp * dr < 1e-9:
            return 0.0
        return num / (dp * dr)

    best_corr  = -999.0
    best_root  = 0
    best_mode  = 0
    for root in range(12):
        for mode, profile in enumerate([major, minor]):
            c = _corr(profile, root)
            if c > best_corr:
                best_corr = c
                best_root = root
                best_mode = mode

    return (best_root, best_mode)


# ---------------------------------------------------------------------------
# Synthetic toy generator
# ---------------------------------------------------------------------------

def _synthesize_song(seed: int, n_bars: int = 16) -> tuple[list[NoteEvent], float, int, int]:
    rng = random.Random(seed)
    tempo = rng.choice([80, 100, 120, 140])
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    chord_roots = [60, 65, 67, 64]
    events: List[NoteEvent] = []
    for bar in range(n_bars):
        for q in range(4):
            events.append(NoteEvent(
                track="melody", bar=bar, position=q * 4,
                pitch=rng.choice(scale), duration=4, velocity=rng.randint(70, 100),
            ))
        root = chord_roots[bar % len(chord_roots)]
        for offs in (0, 4, 7):
            events.append(NoteEvent(
                track="piano", bar=bar, position=0,
                pitch=root + offs, duration=16, velocity=70,
            ))
    return events, tempo, 0, 0  # key_root=0 (C), key_mode=0 (major)


# ---------------------------------------------------------------------------
# POP909 encoder
# ---------------------------------------------------------------------------

def _encode_one(
    midi_path: Path,
    tokenizer: BaseTokenizer,
    cond_tracks: list[str],
    target_tracks: list[str],
    out_dir: Path,
    name_prefix: str = "",
) -> Optional[str]:
    """Encode one POP909 (or generic) MIDI with GT chord/key annotations."""
    stem = name_prefix + _safe_name(midi_path)
    if (out_dir / f"{stem}.pt").exists():
        return stem

    try:
        events, tempo = midi_to_events(midi_path, tokenizer.cfg)
    except Exception as e:
        logger.warning(f"Skipping {midi_path.name}: {e}")
        return None
    if not events:
        return None

    chord_map = None
    key_root  = None
    key_mode  = None
    n_q = tokenizer.cfg.chord_qualities

    # POP909 GT annotations live directly in the song folder (not annotations/).
    # Files: chord_midi.txt (start_sec end_sec chord) + key_audio.txt (start_sec end_sec key).
    song_dir = midi_path.parent
    if isinstance(tokenizer, REMITokenizer):
        chord_file = song_dir / "chord_midi.txt"
        key_file   = song_dir / "key_audio.txt"
        if chord_file.exists():
            cm = _parse_pop909_chord_file(chord_file, n_q, midi_path=midi_path)
            chord_map = cm or None
        if key_file.exists():
            kresult = _parse_pop909_key_file(key_file)
            if kresult is not None:
                key_root, key_mode = kresult

    ids, mask = tokenizer.encode_song(
        events,
        condition_tracks=cond_tracks,
        target_tracks=target_tracks,
        tempo_bpm=tempo,
        chord_map=chord_map,
        key_root=key_root,
        key_mode=key_mode,
    )
    if sum(mask) < 8:
        return None

    stem = name_prefix + _safe_name(midi_path)
    _save_shard(
        out_dir, stem, ids, mask,
        key_root=key_root if key_root is not None else -1,
        key_mode=key_mode if key_mode is not None else -1,
    )
    return stem


def _find_pop909_midis(root: Path) -> list[Path]:
    found = sorted(root.rglob("*.mid")) + sorted(root.rglob("*.midi"))
    return [p for p in found
            if "versions" not in {part.lower() for part in p.parts}]


# ---------------------------------------------------------------------------
# Lakh helpers
# ---------------------------------------------------------------------------

def _find_lakh_midis(root: Path) -> list[Path]:
    return sorted(root.rglob("*.mid")) + sorted(root.rglob("*.midi"))


def _lakh_track_events(
    midi_path: Path, cfg, min_notes: int = 4,
) -> Optional[tuple[list[NoteEvent], float]]:
    try:
        import miditoolkit
        midi = miditoolkit.MidiFile(str(midi_path))
    except Exception as exc:
        logger.debug(f"miditoolkit failed on {midi_path.name}: {exc}")
        return None

    tempo_bpm = midi.tempo_changes[0].tempo if midi.tempo_changes else 120.0
    tpb  = midi.ticks_per_beat
    res  = cfg.resolution
    ppb  = cfg.positions_per_bar
    plo, phi = cfg.pitch_min, cfg.pitch_max
    dur_hi   = cfg.duration_max

    non_drum = [i for i in midi.instruments if not i.is_drum and len(i.notes) >= min_notes]
    if len(non_drum) < 2:
        return None

    # Guard against 32-bit tick overflow in MIDI files (e.g. max_tick = 3×2^32).
    # Such files produce bar indices in the tens of millions, causing the
    # `while cur_bar < e.bar` loop in tokenizer._emit_track() to run millions
    # of times and emit billions of BAR tokens.
    # _extract_chords_from_midi already has a _MAX_BEATS = 3000 guard; mirror it here.
    _MAX_BARS = 2000  # ≈ 10 min at 120 BPM, 4/4
    _all_note_ends = [n.end for inst in non_drum for n in inst.notes]
    if _all_note_ends:
        _max_tick_est = max(_all_note_ends)
        _max_bar_est  = int(_max_tick_est * res / max(tpb, 1)) // ppb
        if _max_bar_est > _MAX_BARS:
            logger.debug(
                f"{midi_path.name}: skipping — max_bar_est={_max_bar_est:,} > {_MAX_BARS} "
                f"(likely 32-bit tick overflow)"
            )
            return None

    def _med_pitch(inst):
        p = sorted(n.pitch for n in inst.notes)
        return float(p[len(p) // 2])

    def _mono_rate(inst):
        notes = sorted(inst.notes, key=lambda n: n.start)
        if len(notes) < 2:
            return 1.0
        return sum(1 for a, b in zip(notes, notes[1:]) if a.end <= b.start) / (len(notes)-1)

    def _density(inst):
        span  = max(n.end for n in inst.notes) - min(n.start for n in inst.notes)
        beats = span / max(tpb, 1)
        return len(inst.notes) / max(beats, 1.0)

    bass    = [i for i in non_drum if i.program in _GM_BASS_PROGRAMS]
    melodic = [i for i in non_drum if i.program not in _GM_BASS_PROGRAMS]
    if len(melodic) < 2:
        melodic, bass = non_drum, []

    mps = [_med_pitch(i) for i in melodic]
    mrs = [_mono_rate(i)  for i in melodic]
    mds = [_density(i)    for i in melodic]
    lo_p, hi_p = min(mps), max(mps)
    lo_d, hi_d = min(mds), max(mds)
    pr = max(hi_p - lo_p, 1.0)
    dr = max(hi_d - lo_d, 1.0)

    def _score(idx):
        return (0.40 * (mps[idx]-lo_p)/pr + 0.40 * mrs[idx]
                + 0.15 * (1-(mds[idx]-lo_d)/dr)
                + 0.05 * (1 if melodic[idx].program in _GM_MELODY_HINT_PROGRAMS else 0))

    ranked   = sorted(range(len(melodic)), key=_score, reverse=True)
    melodic  = [melodic[i] for i in ranked]
    track_for: dict[int, str] = {}
    track_for[id(melodic[0])] = "melody"
    if len(melodic) >= 2:
        track_for[id(melodic[1])] = "bridge"
    for inst in melodic[2:]:
        track_for[id(inst)] = "piano"
    for inst in bass:
        track_for[id(inst)] = "piano"

    events: list[NoteEvent] = []
    for inst in non_drum:
        track = track_for.get(id(inst), cfg.tracks[-1])
        if track not in cfg.tracks:
            track = cfg.tracks[-1]
        for n in inst.notes:
            sp = int(round(n.start * res / tpb))
            ep = int(round(n.end   * res / tpb))
            events.append(NoteEvent(
                track=track,
                bar=sp // ppb, position=sp % ppb,
                pitch=max(plo, min(phi, n.pitch)),
                duration=max(1, min(dur_hi, ep - sp)),
                velocity=max(1, min(127, n.velocity)),
            ))

    return events, tempo_bpm


def _encode_lakh_one(
    midi_path: Path,
    tokenizer: BaseTokenizer,
    cond_tracks: list[str],
    target_tracks: list[str],
    out_dir: Path,
    name_prefix: str = "lakh_",
) -> Optional[str]:
    """Encode one Lakh MIDI. Chord: core 9-quality. Key: auto-estimate."""
    # Fast skip: check if shard already exists before any heavy processing.
    stem = _safe_name(midi_path)
    h    = hashlib.md5(str(midi_path).encode("utf-8", errors="replace")).hexdigest()[:8]
    if len(name_prefix) + len(stem) + 9 > 120:
        stem = stem[:max(0, 120 - len(name_prefix) - 9)]
    raw_name = f"{name_prefix}{stem}_{h}"
    if (out_dir / f"{raw_name}.pt").exists():
        return raw_name

    result = _lakh_track_events(midi_path, tokenizer.cfg)
    if result is None:
        return None
    events, tempo = result

    chord_map = None
    key_root  = None
    key_mode  = None

    if isinstance(tokenizer, REMITokenizer):
        extracted = _extract_chords_from_midi(midi_path, tokenizer.cfg, n_qualities=9)
        chord_map = extracted or None
        kresult   = _estimate_key_from_midi(midi_path)
        if kresult is not None:
            key_root, key_mode = kresult

    ids, mask = tokenizer.encode_song(
        events, cond_tracks, target_tracks,
        tempo_bpm=tempo, chord_map=chord_map,
        key_root=key_root, key_mode=key_mode,
    )
    if sum(mask) < 8:
        return None

    _save_shard(
        out_dir, raw_name, ids, mask,
        key_root=key_root if key_root is not None else -1,
        key_mode=key_mode if key_mode is not None else -1,
    )
    return raw_name


# ---------------------------------------------------------------------------
# Slakh2100 helpers
# ---------------------------------------------------------------------------

def _find_slakh_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("all_src.mid")})


def _encode_slakh_one(
    song_dir: Path,
    tokenizer: BaseTokenizer,
    cond_tracks: list[str],
    target_tracks: list[str],
    out_dir: Path,
    name_prefix: str = "slakh_",
) -> Optional[str]:
    """Encode one Slakh song. Chord: full 12-quality. Key: auto-estimate."""
    stem = _safe_name(song_dir)
    h    = hashlib.md5(str(song_dir).encode("utf-8", errors="replace")).hexdigest()[:8]
    if len(name_prefix) + len(stem) + 9 > 120:
        stem = stem[:max(0, 120 - len(name_prefix) - 9)]
    raw_name = f"{name_prefix}{stem}_{h}"
    if (out_dir / f"{raw_name}.pt").exists():
        return raw_name

    midi_path = song_dir / "all_src.mid"
    if not midi_path.exists():
        midis = sorted(song_dir.glob("*.mid"))
        if not midis:
            return None
        midi_path = midis[0]

    result = _lakh_track_events(midi_path, tokenizer.cfg)
    if result is None:
        return None
    events, tempo = result

    chord_map = None
    key_root  = None
    key_mode  = None

    if isinstance(tokenizer, REMITokenizer):
        extracted = _extract_chords_from_midi(midi_path, tokenizer.cfg, n_qualities=12)
        chord_map = extracted or None
        kresult   = _estimate_key_from_midi(midi_path)
        if kresult is not None:
            key_root, key_mode = kresult

    ids, mask = tokenizer.encode_song(
        events, cond_tracks, target_tracks,
        tempo_bpm=tempo, chord_map=chord_map,
        key_root=key_root, key_mode=key_mode,
    )
    if sum(mask) < 8:
        return None

    _save_shard(
        out_dir, raw_name, ids, mask,
        key_root=key_root if key_root is not None else -1,
        key_mode=key_mode if key_mode is not None else -1,
    )
    return raw_name


# ---------------------------------------------------------------------------
# Main entry
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
    parser.add_argument("--target_tracks", type=str, default="bridge,piano")
    parser.add_argument("--pop909_prefix", type=str, default="pop909_")
    parser.add_argument("--lakh_prefix",   type=str, default="lakh_")
    parser.add_argument("--slakh_prefix",  type=str, default="slakh_")
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

    logger.info(f"Vocab size  : {tokenizer.vocab_size}")
    logger.info(f"Cond tracks : {cond_tracks}")
    logger.info(f"Target      : {target_tracks}")
    logger.info(f"Output dir  : {out_dir}")

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

    if args.lakh_dir:
        midis = _find_lakh_midis(Path(args.lakh_dir))
        if not midis:
            raise SystemExit(f"No MIDI files found under {args.lakh_dir}.")
        logger.info(f"Lakh: {len(midis):,} files — core 9-quality chord set.")
        new_stems, n_skip = [], 0
        for p in tqdm(midis):
            stem = _encode_lakh_one(p, tokenizer, cond_tracks, target_tracks, out_dir,
                                    name_prefix=args.lakh_prefix)
            if stem:
                new_stems.append(stem)
            else:
                n_skip += 1
        write_meta(out_dir, tokenizer, cond_tracks, target_tracks, new_stems)
        logger.info(f"Lakh: saved {len(new_stems):,}  skipped {n_skip:,}")
        return

    if args.slakh_dir:
        song_dirs = _find_slakh_dirs(Path(args.slakh_dir))
        if not song_dirs:
            raise SystemExit(f"No all_src.mid files under {args.slakh_dir}.")
        logger.info(f"Slakh: {len(song_dirs):,} songs — full 12-quality chord set.")
        new_stems, n_skip = [], 0
        for d in tqdm(song_dirs):
            stem = _encode_slakh_one(d, tokenizer, cond_tracks, target_tracks, out_dir,
                                     name_prefix=args.slakh_prefix)
            if stem:
                new_stems.append(stem)
            else:
                n_skip += 1
        write_meta(out_dir, tokenizer, cond_tracks, target_tracks, new_stems)
        logger.info(f"Slakh: saved {len(new_stems):,}  skipped {n_skip:,}")
        return

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
                           name_prefix=name_prefix)
        if stem:
            new_stems.append(stem)
        else:
            n_skip += 1
    write_meta(out_dir, tokenizer, cond_tracks, target_tracks, new_stems)
    logger.info(f"Saved {len(new_stems)}/{len(midis)}  skipped {n_skip}")


if __name__ == "__main__":
    main()
