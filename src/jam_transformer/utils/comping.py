"""Inference-time accompaniment re-rhythming (temporary, no retraining).

The current model collapses accompaniment into a single large note-cluster on
the downbeat. This module treats each bar's cluster as a *chord* and
re-articulates it with:

  1. Chord voicing  — tension above chord tones, avoid-note removal.
  2. Voice leading  — choose the register that minimises movement from the
                      previous bar so chords connect smoothly.
  3. Context-aware  — melody density guides figuration choice per bar:
     pattern choice   dense melody → simpler accompaniment (long notes, space)
                       sparse melody → busier accompaniment (arpeggios)
  4. Stochastic     — small random variations so adjacent bars don't sound
     variation         identical (beat omissions, density nudge).

This is a Band-Aid for polyphony-collapse. The real fix is training-side.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

import numpy as np

from jam_transformer.tokenizer import NoteEvent

# ---------------------------------------------------------------------------
# Harmonic function tables
# ---------------------------------------------------------------------------
_CHORD_TONE = {3: "3", 4: "3", 7: "5", 10: "7", 11: "7"}
_TENSION    = {1: "b9", 2: "9", 5: "11", 6: "#11", 8: "b13", 9: "13"}
_PRIORITY   = ["3", "7", "13", "9", "#11", "5", "b9", "b13", "11"]
_CHORD_TONE_FN = {"3", "5", "7"}

AVAILABLE_PATTERNS = (
    "auto",                                 # context-aware (recommended)
    "arp_up", "arp_updown", "alberti",      # arpeggiated
    "broken",                               # sustained bass + upper arp
    "quarter", "backbeat",                  # block comping
)

# ---------------------------------------------------------------------------
# Voicing: reduce cluster → (bass, upper_tones) ordered by harmonic function
# ---------------------------------------------------------------------------
def _voice_chord(pitches: list[int], max_voices: int) -> tuple[int, list[int]]:
    """Reduce a cluster to (bass, upper_tones) with tension handling.

    • chord tones (3/5/7) close-stacked in mid register
    • tensions (9/11/13) one octave above chord tones (upper structure)
    • natural-11 over major 3rd dropped (avoid note)
    • capped to max_voices-1 upper voices, guide tones kept first
    """
    pitches = sorted(pitches)
    bass = pitches[0]
    root_pc = bass % 12

    present: dict[str, int] = {}
    for p in pitches:
        rel = (p % 12 - root_pc) % 12
        if rel == 0:
            continue
        fn = _CHORD_TONE.get(rel) or _TENSION.get(rel)
        if fn:
            present.setdefault(fn, rel)

    # drop avoid note: natural 11 over major 3rd
    if present.get("3") == 4 and present.get("11") == 5:
        present.pop("11", None)

    chosen = [fn for fn in _PRIORITY if fn in present][: max(0, max_voices - 1)]

    def _stack(rels: list[int], start: int) -> tuple[list[int], int]:
        out, cur = [], start
        for rel in sorted(rels):
            target_pc = (root_pc + rel) % 12
            n = cur + ((target_pc - cur) % 12)
            if n <= cur:
                n += 12
            out.append(n); cur = n
        return out, cur

    chord_rels   = [present[fn] for fn in chosen if fn in _CHORD_TONE_FN]
    tension_rels = [present[fn] for fn in chosen if fn not in _CHORD_TONE_FN]
    chord_notes, top = _stack(chord_rels, max(bass + 7, 55))
    tension_notes, _ = _stack(
        tension_rels,
        max(top, chord_notes[-1] if chord_notes else top) + 1,
    )
    return bass, [*chord_notes, *tension_notes]


# ---------------------------------------------------------------------------
# Voice leading: re-voice upper tones to minimise movement from prev bar
# ---------------------------------------------------------------------------
def _voice_lead(bass: int, upper: list[int],
                prev_bass: int | None, prev_upper: list[int]) -> tuple[int, list[int]]:
    """Shift upper voices by ±1 octave to minimise total movement from prev.

    Bass is shifted toward prev_bass by 0/±1 octave so it stays in range.
    Upper voices are each independently moved to the closest octave version
    of the same pitch-class to the corresponding previous voice.
    """
    if not prev_upper or not upper:
        return bass, upper

    # Bass: prefer the octave closest to prev_bass while staying 36..60
    if prev_bass is not None:
        candidates = [bass + 12 * k for k in range(-2, 3) if 28 <= bass + 12 * k <= 62]
        bass = min(candidates, key=lambda x: abs(x - prev_bass), default=bass)

    # Upper: greedily assign each voice to the previous voice it's closest to
    new_upper = list(upper)
    for i, u in enumerate(upper):
        pc = u % 12
        # all octave candidates in 48..88
        cands = [pc + 12 * k for k in range(3, 8) if 48 <= pc + 12 * k <= 88]
        if not cands:
            continue
        if i < len(prev_upper):
            target = prev_upper[i]
        else:
            target = prev_upper[-1]
        new_upper[i] = min(cands, key=lambda x: abs(x - target))

    # Ensure upper voices stay above bass and are sorted ascending
    new_upper = sorted(max(bass + 1, x) for x in new_upper)
    return bass, new_upper


# ---------------------------------------------------------------------------
# Melody density helper
# ---------------------------------------------------------------------------
def _melody_density(melody_events: list[NoteEvent] | None, bar: int,
                    ppb: int) -> float:
    """Fraction of 16th positions in this bar that have a melody note (0..1)."""
    if not melody_events:
        return 0.5
    positions = {e.position for e in melody_events
                 if e.bar == bar and e.track in ("melody", "MELODY")}
    return len(positions) / ppb


# ---------------------------------------------------------------------------
# Figuration (pattern → list of (position, pitches, duration))
# ---------------------------------------------------------------------------
def _figure(bass: int, upper: list[int], pattern: str,
            ppb: int, density: float, rng: random.Random
            ) -> list[tuple[int, list[int], int]]:
    """Build the list of (position, [pitches], duration) for one bar."""
    voices = [bass, *upper]
    n      = len(voices)
    beat   = ppb // 4          # 4 sixteenth notes per beat
    eighth = list(range(0, ppb, 2))

    if pattern == "quarter":
        # stochastic: ~20% chance to drop a beat (except beat 1)
        out = []
        for b, pos in enumerate(range(0, ppb, beat)):
            if b != 0 and rng.random() < 0.20:
                continue
            out.append((pos, voices, beat))
        return out

    if pattern == "backbeat":
        chord = upper or [bass]
        out = [(0, [bass], beat), (beat, chord, beat),
               (2 * beat, [bass], beat), (3 * beat, chord, beat)]
        # occasional extra 8th-note chord hit
        if rng.random() < 0.30:
            extra_pos = rng.choice([2, 6, 10, 14])
            out.append((extra_pos, chord, 2))
        return sorted(out)

    if pattern == "broken":
        out = [(0, [bass], ppb)]   # sustained bass
        k = 0
        for pos in eighth:
            if pos == 0 or not upper:
                continue
            # occasionally skip an 8th-note hit for breathing room
            if rng.random() < 0.15:
                continue
            out.append((pos, [upper[k % len(upper)]], 2))
            k += 1
        return out

    if pattern == "alberti":
        top = n - 1
        mid = max(1, n // 2)
        order = [0, top, mid, top]
        out = []
        for i, pos in enumerate(eighth):
            if rng.random() < 0.10:
                continue
            out.append((pos, [voices[order[i % len(order)] % n]], 2))
        return out

    if pattern == "arp_updown":
        seq = list(range(n)) + list(range(n - 2, 0, -1))
        out = []
        for i, pos in enumerate(eighth):
            if rng.random() < 0.10:
                continue
            out.append((pos, [voices[seq[i % len(seq)]]], 2))
        return out

    if pattern == "long":
        # whole-bar chord — used in very dense melody bars
        return [(0, voices, ppb)]

    # default / arp_up
    out = []
    for i, pos in enumerate(eighth):
        if rng.random() < 0.10:
            continue
        out.append((pos, [voices[i % n]], 2))
    return out


# ---------------------------------------------------------------------------
# Context-aware pattern selection
# ---------------------------------------------------------------------------
def _choose_pattern(density: float, bar_in_phrase: int,
                    rng: random.Random) -> str:
    """Select figuration based on melodic density + phrase position.

    density 0..1: fraction of 16th positions that have a melody note.
    bar_in_phrase 0-based position within a 4-bar phrase.
    """
    if density > 0.65:
        # melody very busy → long notes / sparse (call-and-response)
        return rng.choice(["long", "broken"])
    if density < 0.15:
        # melody very sparse → busier movement fills the space
        return rng.choice(["arp_up", "arp_updown", "alberti"])
    # moderate density: vary by phrase position
    phrase_map = {
        0: ["broken", "alberti"],         # phrase start: establish
        1: ["arp_up", "broken"],          # build
        2: ["arp_updown", "alberti"],     # peak / most active
        3: ["broken", "quarter"],         # cadence: settle back
    }
    return rng.choice(phrase_map.get(bar_in_phrase % 4, ["broken", "arp_up"]))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def redistribute_accompaniment(
    events: list[NoteEvent],
    cfg,
    pattern: str = "auto",
    max_voices: int = 4,
    min_chord: int = 2,
    melody_events: list[NoteEvent] | None = None,
    seed: int | None = None,
) -> list[NoteEvent]:
    """Re-articulate each bar's chord as a broken-chord / arpeggio figure.

    pattern     : one of AVAILABLE_PATTERNS. "auto" = context-aware selection.
    max_voices  : cap on distinct chord tones per onset.
    min_chord   : bars with fewer distinct pitches are left untouched.
    melody_events: if provided, used for density-guided pattern selection.
    seed        : random seed for reproducible stochastic variation.
    """
    if pattern not in AVAILABLE_PATTERNS:
        return events

    ppb   = int(cfg.positions_per_bar)
    track = cfg.tracks[-1]
    rng   = random.Random(seed)

    by_bar: dict[int, list[NoteEvent]] = defaultdict(list)
    for e in events:
        by_bar[e.bar].append(e)

    sorted_bars = sorted(by_bar.keys())
    first_bar   = sorted_bars[0] if sorted_bars else 0

    prev_bass: int | None = None
    prev_upper: list[int] = []

    out: list[NoteEvent] = []
    for bar in sorted_bars:
        notes   = by_bar[bar]
        pitches = sorted({n.pitch for n in notes})
        if len(pitches) < min_chord:
            out.extend(notes)
            prev_bass, prev_upper = None, []
            continue

        vel = int(np.median([n.velocity for n in notes]))
        bass, upper = _voice_chord(pitches, max_voices)

        # voice-lead from previous bar
        bass, upper = _voice_lead(bass, upper, prev_bass, prev_upper)
        prev_bass, prev_upper = bass, list(upper)

        # choose figuration
        if pattern == "auto":
            density      = _melody_density(melody_events, bar, ppb)
            bar_in_phrase = (bar - first_bar) % 4
            pat = _choose_pattern(density, bar_in_phrase, rng)
        else:
            pat = pattern

        for pos, pset, dur in _figure(bass, upper, pat, ppb, 0.0, rng):
            for p in pset:
                out.append(NoteEvent(track, bar, pos, p, dur, vel))

    out.sort(key=lambda e: (e.bar, e.position, e.pitch))
    return out
