"""REMI-style tokenizer with fully relative harmonic encoding.

Design rationale
----------------
All harmonic information is expressed relative to the piece's key:

  Chord root  → SCALE_DEGREE_{0-11}   semitones above key root (mod 12)
  Chord type  → QUALITY_{name}        one of 12 chord qualities
  Note pitch  → CHROMA_{0-11}         pitch class relative to key root
                OCTAVE_{1-9}          abs_pitch // 12  (register)

The KEY_{root}_{mode} token (24 variants) is the sole absolute harmonic anchor.

Key invariance proof
---------------------
Given transposition by N semitones (new_key_root = (key_root+N)%12):
  new_CHROMA      = ((P+N)%12 - (K+N)%12) % 12  =  (P%12 - K%12) % 12  =  CHROMA  ✓
  new_SCALE_DEG   = ((C+N)%12 - (K+N)%12) % 12  =  (C%12 - K%12) % 12  =  SD      ✓

CHROMA and SCALE_DEGREE are invariant. Only KEY root and OCTAVE need updating
during pitch augmentation.

Vocabulary layout (174 tokens)
--------------------------------
  [0]          PAD
  [1]          BOS
  [2]          SEP
  [3]          EOS
  [4]          BAR
  [5..20]      POS_0..POS_15           (16 positions per bar)
  [21..36]     TEMPO_0..TEMPO_15       (16 bins, log-scale 50–200 BPM)
  [37..39]     TRACK_melody/bridge/piano
  [40..51]     CHROMA_0..CHROMA_11     pitch class relative to key root
  [52..60]     OCTAVE_1..OCTAVE_9      abs_pitch // 12
  [61..92]     DUR_1..DUR_32
  [93..124]    VEL_0..VEL_31
  [125..136]   SCALE_DEGREE_0..11      chord root relative to key root
  [137..148]   QUALITY_{name} × 12    chord quality
  [149]        CHORD_N                 no chord / unknown
  [150..173]   KEY_{r}_maj/min × 24   global key anchor

Total: 174 tokens.

Tempo log-scale rationale
--------------------------
Human tempo JND ≈ 4-6%; 50→200 BPM spans log2(200/50)=2 octaves.
16 log-spaced bins give ~9% spacing (slightly above JND), covering all
musical tempo categories (Larghetto→Prestissimo) without the density
imbalance of linear binning (which over-resolves 50-100 BPM and
under-resolves 150-200 BPM).

Sequence format
---------------
  <BOS> KEY_{r}_{mode} TEMPO_{bin}
    TRACK_melody
      BAR  [SCALE_DEGREE_x QUALITY_y | CHORD_N]
      POS_n  CHROMA_c OCTAVE_o  DUR_d  VEL_v
      ...
    <SEP>
    TRACK_piano
      BAR  [SCALE_DEGREE_x QUALITY_y | CHORD_N]
      POS_n  CHROMA_c OCTAVE_o  DUR_d  VEL_v
      ...
  <EOS>

Augmentation contract
---------------------
To transpose by N semitones (dataset._augment):
  1. Update KEY root:      new_root = (key_root + N) % 12
  2. Update OCTAVE tokens: new_abs  = old_abs + N  (clamped); new_OCTAVE = new_abs // 12
  3. CHROMA tokens:        unchanged (key-invariant — see proof above)
  4. SCALE_DEGREE tokens:  unchanged (key-invariant)

Chord map format (prepare_data → encode_song)
----------------------------------------------
  chord_map: dict[(bar, pos_in_resolution_units), (chord_root_0_11, quality_idx) | None]
  None  → emit CHORD_N
  (root, q) → emit SCALE_DEGREE_{(root-key_root)%12}  QUALITY_{q}
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from jam_transformer.config import TokenizerConfig
from jam_transformer.logger import logger


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: 12 chord qualities in token-index order.
#: Indices 0-8 (core): reliably extractable from any MIDI source.
#:   add9 (index 7) absorbs sus2 — add9 retains the major 3rd.
#: Indices 9-11 (extended / Slakh-tier): dim7, hdim7, dom9.
CHORD_QUALITIES: List[str] = [
    "maj",    # 0
    "min",    # 1
    "dom7",   # 2
    "maj7",   # 3
    "min7",   # 4
    "dim",    # 5
    "aug",    # 6
    "add9",   # 7  (sus2 collapsed here)
    "sus4",   # 8
    "dim7",   # 9
    "hdim7",  # 10
    "dom9",   # 11
]

CHORD_ROOTS_STR: List[str] = [
    "C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B",
]

KEY_MODES_STR: List[str] = ["maj", "min"]

#: OCTAVE token range: abs_pitch // 12 for MIDI 21-108.
OCTAVE_MIN: int = 1   # MIDI 21 (A0) → 21 // 12 = 1
OCTAVE_MAX: int = 9   # MIDI 108 (C8) → 108 // 12 = 9


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class BaseTokenizer(ABC):
    cfg: TokenizerConfig

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...
    @property
    @abstractmethod
    def pad_id(self) -> int: ...
    @property
    @abstractmethod
    def bos_id(self) -> int: ...
    @property
    @abstractmethod
    def sep_id(self) -> int: ...
    @property
    @abstractmethod
    def eos_id(self) -> int: ...

    @abstractmethod
    def encode_song(
        self,
        events: "Sequence[NoteEvent]",
        condition_tracks: Sequence[str],
        target_tracks: Sequence[str],
        tempo_bpm: float | None = None,
        chord_map: "dict[tuple[int, int], tuple[int, int] | None] | None" = None,
        key_root: int | None = None,
        key_mode: int | None = None,
    ) -> Tuple[List[int], List[bool]]: ...

    @abstractmethod
    def decode(self, ids: Iterable[int]) -> "List[NoteEvent]": ...

    # --- pitch class / register helpers (relative encoding) -----------------
    @property
    @abstractmethod
    def chroma_min_id(self) -> int: ...
    @property
    @abstractmethod
    def chroma_max_id(self) -> int: ...
    @property
    @abstractmethod
    def octave_min_id(self) -> int: ...
    @property
    @abstractmethod
    def octave_max_id(self) -> int: ...
    @property
    @abstractmethod
    def octave_base(self) -> int: ...

    # --- velocity / duration / tempo helpers --------------------------------
    @property
    @abstractmethod
    def vel_min_id(self) -> int: ...
    @property
    @abstractmethod
    def vel_max_id(self) -> int: ...
    @property
    @abstractmethod
    def dur_min_id(self) -> int: ...
    @property
    @abstractmethod
    def dur_max_id(self) -> int: ...
    @property
    @abstractmethod
    def tempo_min_id(self) -> int: ...
    @property
    @abstractmethod
    def tempo_max_id(self) -> int: ...

    # --- harmony helpers ----------------------------------------------------
    @property
    @abstractmethod
    def sd_min_id(self) -> int: ...
    @property
    @abstractmethod
    def sd_max_id(self) -> int: ...
    @property
    @abstractmethod
    def quality_min_id(self) -> int: ...
    @property
    @abstractmethod
    def quality_max_id(self) -> int: ...
    @property
    @abstractmethod
    def chord_n_id(self) -> int: ...
    @property
    @abstractmethod
    def key_min_id(self) -> int: ...
    @property
    @abstractmethod
    def key_max_id(self) -> int: ...

    @abstractmethod
    def key_token_id(self, root: int, mode: int) -> int: ...

    # --- structural token id helpers ----------------------------------------
    @property
    @abstractmethod
    def bar_id(self) -> int: ...
    @property
    @abstractmethod
    def pos_id_range(self) -> Tuple[int, int]: ...
    @property
    @abstractmethod
    def track_id_range(self) -> Tuple[int, int]: ...

    # --- CFG helper ---------------------------------------------------------
    def make_uncond_prompt(self, prompt_ids) -> "list[int]":
        import torch as _torch
        ids = prompt_ids.tolist() if isinstance(prompt_ids, _torch.Tensor) else list(prompt_ids)
        try:
            sep_idx = ids.index(self.sep_id)
        except ValueError:
            return ids
        ids = ids[:]
        for i in range(1, sep_idx):
            ids[i] = self.pad_id
        return ids


_TOKENIZER_REGISTRY: Dict[str, Type[BaseTokenizer]] = {}


def register_tokenizer(name: str) -> Callable[[Type[BaseTokenizer]], Type[BaseTokenizer]]:
    def deco(cls: Type[BaseTokenizer]) -> Type[BaseTokenizer]:
        key = name.lower()
        if key in _TOKENIZER_REGISTRY:
            raise ValueError(f"Tokenizer '{name}' already registered")
        _TOKENIZER_REGISTRY[key] = cls
        return cls
    return deco


def build_tokenizer(cfg: TokenizerConfig) -> BaseTokenizer:
    key = cfg.name.lower()
    if key not in _TOKENIZER_REGISTRY:
        raise KeyError(f"Unknown tokenizer '{cfg.name}'. Known: {sorted(_TOKENIZER_REGISTRY)}")
    return _TOKENIZER_REGISTRY[key](cfg)


def available_tokenizers() -> list[str]:
    return sorted(_TOKENIZER_REGISTRY)


# ---------------------------------------------------------------------------
# NoteEvent
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    track: str
    bar: int
    position: int          # 0..positions_per_bar-1
    pitch: int             # MIDI pitch (absolute)
    duration: int          # resolution units
    velocity: int          # 1..127

    def __post_init__(self):
        self.duration = max(1, int(self.duration))
        self.velocity = max(1, min(127, int(self.velocity)))


# ---------------------------------------------------------------------------
# REMITokenizer — relative harmonic encoding
# ---------------------------------------------------------------------------

@register_tokenizer("remi_v1")
class REMITokenizer(BaseTokenizer):
    """REMI tokenizer with fully relative harmonic encoding.

    See module docstring for full vocabulary layout and design rationale.
    """

    PAD = "<PAD>"
    BOS = "<BOS>"
    SEP = "<SEP>"
    EOS = "<EOS>"

    def __init__(self, cfg: TokenizerConfig):
        self.cfg = cfg
        self.id_to_token: List[str] = []
        self.token_to_id: Dict[str, int] = {}

        def _add(tok: str) -> None:
            if tok in self.token_to_id:
                raise ValueError(f"Duplicate token: {tok}")
            self.token_to_id[tok] = len(self.id_to_token)
            self.id_to_token.append(tok)

        # ---- Specials --------------------------------------------------------
        for t in (self.PAD, self.BOS, self.SEP, self.EOS):
            _add(t)

        # ---- Structure -------------------------------------------------------
        _add("BAR")
        for i in range(cfg.positions_per_bar):
            _add(f"POS_{i}")
        tempo_first = len(self.id_to_token)
        for i in range(cfg.tempo_bins):
            _add(f"TEMPO_{i}")
        tempo_last = len(self.id_to_token) - 1

        # ---- Tracks ----------------------------------------------------------
        for name in cfg.tracks:
            _add(f"TRACK_{name}")

        # ---- Pitch: CHROMA + OCTAVE (relative encoding) ----------------------
        chroma_first = len(self.id_to_token)
        for c in range(12):
            _add(f"CHROMA_{c}")
        chroma_last = len(self.id_to_token) - 1

        octave_first = len(self.id_to_token)
        for o in range(OCTAVE_MIN, OCTAVE_MAX + 1):
            _add(f"OCTAVE_{o}")
        octave_last = len(self.id_to_token) - 1

        # ---- Duration / Velocity ---------------------------------------------
        dur_first = len(self.id_to_token)
        for d in range(cfg.duration_min, cfg.duration_max + 1):
            _add(f"DUR_{d}")
        dur_last = len(self.id_to_token) - 1

        vel_first = len(self.id_to_token)
        for v in range(cfg.velocity_bins):
            _add(f"VEL_{v}")
        vel_last = len(self.id_to_token) - 1

        # ---- Harmony: SCALE_DEGREE + QUALITY + CHORD_N -----------------------
        n_q = min(len(CHORD_QUALITIES), max(1, cfg.chord_qualities))
        self._n_chord_qualities = n_q

        sd_first = len(self.id_to_token)
        for deg in range(12):
            _add(f"SCALE_DEGREE_{deg}")
        sd_last = len(self.id_to_token) - 1

        quality_first = len(self.id_to_token)
        for q_idx in range(n_q):
            _add(f"QUALITY_{CHORD_QUALITIES[q_idx]}")
        quality_last = len(self.id_to_token) - 1

        _add("CHORD_N")
        chord_n = len(self.id_to_token) - 1

        # ---- Key token (optional) --------------------------------------------
        if cfg.use_key_tokens:
            key_first = len(self.id_to_token)
            for mode_str in KEY_MODES_STR:
                for r in range(12):
                    _add(f"KEY_{r}_{mode_str}")
            key_last = len(self.id_to_token) - 1
            self._key_min_id = key_first
            self._key_max_id = key_last
        else:
            self._key_min_id = -1
            self._key_max_id = -1

        # Cache id ranges
        self._tempo_min_id   = tempo_first
        self._tempo_max_id   = tempo_last
        self._chroma_min_id  = chroma_first
        self._chroma_max_id  = chroma_last
        self._octave_min_id  = octave_first
        self._octave_max_id  = octave_last
        self._dur_min_id     = dur_first
        self._dur_max_id     = dur_last
        self._vel_min_id     = vel_first
        self._vel_max_id     = vel_last
        self._sd_min_id      = sd_first
        self._sd_max_id      = sd_last
        self._quality_min_id = quality_first
        self._quality_max_id = quality_last
        self._chord_n_id     = chord_n

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def sep_id(self) -> int:
        return self.token_to_id[self.SEP]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    def tid(self, token: str) -> int:
        return self.token_to_id[token]

    @property
    def chroma_min_id(self) -> int:
        return self._chroma_min_id

    @property
    def chroma_max_id(self) -> int:
        return self._chroma_max_id

    @property
    def octave_min_id(self) -> int:
        return self._octave_min_id

    @property
    def octave_max_id(self) -> int:
        return self._octave_max_id

    @property
    def octave_base(self) -> int:
        return OCTAVE_MIN

    @property
    def vel_min_id(self) -> int:
        return self._vel_min_id

    @property
    def vel_max_id(self) -> int:
        return self._vel_max_id

    @property
    def dur_min_id(self) -> int:
        return self._dur_min_id

    @property
    def dur_max_id(self) -> int:
        return self._dur_max_id

    @property
    def tempo_min_id(self) -> int:
        return self._tempo_min_id

    @property
    def tempo_max_id(self) -> int:
        return self._tempo_max_id

    @property
    def sd_min_id(self) -> int:
        return self._sd_min_id

    @property
    def sd_max_id(self) -> int:
        return self._sd_max_id

    @property
    def quality_min_id(self) -> int:
        return self._quality_min_id

    @property
    def quality_max_id(self) -> int:
        return self._quality_max_id

    @property
    def chord_n_id(self) -> int:
        return self._chord_n_id

    @property
    def key_min_id(self) -> int:
        return self._key_min_id

    @property
    def key_max_id(self) -> int:
        return self._key_max_id

    def key_token_id(self, root: int, mode: int) -> int:
        if self._key_min_id < 0:
            return -1
        return self._key_min_id + mode * 12 + root

    # ------------------------------------------------------------------
    # Structure / content id helpers
    # ------------------------------------------------------------------
    @property
    def bar_id(self) -> int:
        return self.token_to_id["BAR"]

    @property
    def pos_id_range(self) -> Tuple[int, int]:
        first = self.token_to_id["POS_0"]
        last  = self.token_to_id[f"POS_{self.cfg.positions_per_bar - 1}"]
        return first, last

    @property
    def track_id_range(self) -> Tuple[int, int]:
        first = self.token_to_id[f"TRACK_{self.cfg.tracks[0]}"]
        last  = self.token_to_id[f"TRACK_{self.cfg.tracks[-1]}"]
        return first, last

    def structural_ids(self) -> List[int]:
        ids = [self.bar_id]
        lo, hi = self.pos_id_range
        ids.extend(range(lo, hi + 1))
        lo, hi = self.track_id_range
        ids.extend(range(lo, hi + 1))
        ids.extend(range(self._tempo_min_id, self._tempo_max_id + 1))
        return ids

    def content_ids(self) -> List[int]:
        ids  = list(range(self._chroma_min_id,  self._chroma_max_id  + 1))
        ids += list(range(self._octave_min_id,  self._octave_max_id  + 1))
        ids += list(range(self._dur_min_id,     self._dur_max_id     + 1))
        ids += list(range(self._vel_min_id,     self._vel_max_id     + 1))
        return ids

    def build_token_weight_vector(
        self, struct_weight: float, content_weight: float,
    ) -> "list[float]":
        """Per-id weight vector for CE loss scaling.

        Weight assignment:
          structural (BAR/POS/TRACK/TEMPO)  → struct_weight
          content (CHROMA/OCTAVE/DUR/VEL)   → content_weight
          SCALE_DEGREE / QUALITY            → content_weight  (harmonic decisions)
          CHORD_N                           → struct_weight   (placeholder)
          KEY_*                             → struct_weight   (global anchor)
          specials (PAD/BOS/SEP/EOS)        → 1.0
        """
        w = [1.0] * self.vocab_size
        for tid in self.structural_ids():
            w[tid] = struct_weight
        for tid in self.content_ids():
            w[tid] = content_weight
        for tid in range(self._sd_min_id, self._sd_max_id + 1):
            w[tid] = content_weight
        for tid in range(self._quality_min_id, self._quality_max_id + 1):
            w[tid] = content_weight
        w[self._chord_n_id] = struct_weight
        if self._key_min_id >= 0:
            for tid in range(self._key_min_id, self._key_max_id + 1):
                w[tid] = struct_weight
        return w

    # ------------------------------------------------------------------
    # Velocity / tempo binning
    # ------------------------------------------------------------------
    def _velocity_bin(self, vel: int) -> int:
        import math
        vel = max(1, min(127, vel))
        b = int(math.log2(vel) / math.log2(128) * self.cfg.velocity_bins)
        return min(self.cfg.velocity_bins - 1, max(0, b))

    def _velocity_from_bin(self, b: int) -> int:
        import math
        b = max(0, min(self.cfg.velocity_bins - 1, b))
        return max(1, min(127, round(2 ** ((b + 0.5) * 7.0 / self.cfg.velocity_bins))))

    def tempo_bin(self, bpm: float) -> int:
        import math
        lo, hi = self.cfg.tempo_min, self.cfg.tempo_max
        bpm = max(lo, min(hi, float(bpm)))
        b = int((math.log(bpm) - math.log(lo)) / (math.log(hi) - math.log(lo)) * self.cfg.tempo_bins)
        return min(self.cfg.tempo_bins - 1, max(0, b))

    def tempo_from_bin(self, b: int) -> float:
        import math
        b = max(0, min(self.cfg.tempo_bins - 1, b))
        lo, hi = self.cfg.tempo_min, self.cfg.tempo_max
        return math.exp(math.log(lo) + (b + 0.5) / self.cfg.tempo_bins * (math.log(hi) - math.log(lo)))

    # ------------------------------------------------------------------
    # Pitch encoding / decoding helpers
    # ------------------------------------------------------------------
    def _pitch_to_chroma_octave(self, pitch: int, key_root: int) -> Tuple[int, int]:
        """Absolute MIDI pitch → (chroma_token_id, octave_token_id)."""
        chroma = (pitch % 12 - key_root) % 12
        octave = pitch // 12
        octave = max(OCTAVE_MIN, min(OCTAVE_MAX, octave))
        return (
            self._chroma_min_id + chroma,
            self._octave_min_id + (octave - OCTAVE_MIN),
        )

    def _chroma_octave_to_pitch(self, chroma_id: int, octave_id: int, key_root: int) -> int:
        """(chroma_token_id, octave_token_id) → absolute MIDI pitch."""
        chroma  = chroma_id  - self._chroma_min_id            # 0-11
        octave  = octave_id  - self._octave_min_id + OCTAVE_MIN  # actual octave
        abs_pc  = (chroma + key_root) % 12
        return octave * 12 + abs_pc

    # ------------------------------------------------------------------
    # Track emission
    # ------------------------------------------------------------------
    def _emit_track(
        self,
        events: Sequence[NoteEvent],
        track: str,
        key_root: int,
        chord_map: "dict[tuple[int, int], tuple[int, int] | None] | None" = None,
    ) -> List[int]:
        """Emit TRACK / BAR / [chord tokens] / POS / CHROMA / OCTAVE / DUR / VEL.

        chord_map: (bar, pos_resolution_units) →
            (chord_root_0_11, quality_idx)   for a known chord
            None                             for CHORD_N

        A chord token pair (SCALE_DEGREE + QUALITY) or CHORD_N is emitted:
          • Right after each BAR token (chord at beat 0), only on change.
          • Before a POS_x token (x > 0) when chord changes mid-bar.
        """
        track_events = [e for e in events if e.track == track]
        if not track_events:
            return []

        track_events = sorted(track_events, key=lambda e: (e.bar, e.position, e.pitch))

        out: List[int] = [self.tid(f"TRACK_{track}")]
        cur_bar = -1
        cur_pos = -1
        last_chord_sig: Optional[tuple] = "UNSET"  # sentinel distinct from None

        def _emit_chord(chord_val: "tuple[int,int] | None") -> None:
            nonlocal last_chord_sig
            # "UNSET" sentinel: position not present in chord_map — no change.
            if chord_val == "UNSET":
                return
            if chord_val == last_chord_sig:
                return
            last_chord_sig = chord_val
            if chord_val is None:
                out.append(self._chord_n_id)
            else:
                root, q_idx = chord_val
                if q_idx >= self._n_chord_qualities:
                    out.append(self._chord_n_id)
                    return
                sd = (root - key_root) % 12
                out.append(self._sd_min_id + sd)
                out.append(self._quality_min_id + q_idx)

        # Hard upper bound on bar index. Prevents runaway token generation from
        # MIDI files with tick-overflow or corrupted bar indices. Mirrors the
        # _MAX_BARS = 2000 guard in prepare_data._lakh_track_events().
        _MAX_BAR_GUARD = 2000

        for e in track_events:
            if e.pitch < self.cfg.pitch_min or e.pitch > self.cfg.pitch_max:
                continue
            if e.bar > _MAX_BAR_GUARD:
                logger.warning(
                    f"_emit_track: skipping note at bar={e.bar} > {_MAX_BAR_GUARD} "
                    f"(tick overflow / corrupted MIDI). track={track}"
                )
                continue

            while cur_bar < e.bar:
                out.append(self.tid("BAR"))
                cur_bar += 1
                cur_pos = -1
                if chord_map is not None:
                    _emit_chord(chord_map.get((cur_bar, 0), "UNSET"))

            if e.position != cur_pos:
                if chord_map is not None and e.position != 0:
                    _emit_chord(chord_map.get((cur_bar, e.position), "UNSET"))
                out.append(self.tid(f"POS_{e.position}"))
                cur_pos = e.position

            dur = min(self.cfg.duration_max, max(self.cfg.duration_min, e.duration))
            chroma_id, octave_id = self._pitch_to_chroma_octave(e.pitch, key_root)
            out.append(chroma_id)
            out.append(octave_id)
            out.append(self.tid(f"DUR_{dur}"))
            out.append(self.tid(f"VEL_{self._velocity_bin(e.velocity)}"))

        return out

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_song(
        self,
        events: Sequence[NoteEvent],
        condition_tracks: Sequence[str],
        target_tracks: Sequence[str],
        tempo_bpm: float | None = None,
        chord_map: "dict[tuple[int, int], tuple[int, int] | None] | None" = None,
        key_root: int | None = None,
        key_mode: int | None = None,
    ) -> Tuple[List[int], List[bool]]:
        """Return (token_ids, target_mask).

        key_root: 0-11 (C=0 … B=11). Used to compute CHROMA and SCALE_DEGREE.
                  Defaults to 0 (C) when None.
        key_mode: 0=major, 1=minor.  Defaults to 0 when None.
        chord_map: see module docstring for format.
        """
        kr = int(key_root) if key_root is not None else 0
        km = int(key_mode) if key_mode is not None else 0

        ids: List[int] = [self.bos_id]

        if self._key_min_id >= 0 and key_root is not None:
            ids.append(self.key_token_id(kr, km))

        if tempo_bpm is not None:
            ids.append(self.tid(f"TEMPO_{self.tempo_bin(tempo_bpm)}"))

        for tr in condition_tracks:
            ids.extend(self._emit_track(events, tr, kr, chord_map=chord_map))
        ids.append(self.sep_id)
        cond_len = len(ids)

        for tr in target_tracks:
            ids.extend(self._emit_track(events, tr, kr, chord_map=chord_map))
        ids.append(self.eos_id)

        mask = [False] * cond_len + [True] * (len(ids) - cond_len)
        return ids, mask

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode(self, ids: Iterable[int]) -> List[NoteEvent]:
        """Turn token ids back into NoteEvents.

        Extracts key_root from the first KEY_* token in the sequence.
        Falls back to key_root=0 (C) if no KEY token is present.
        CHORD_N / SCALE_DEGREE / QUALITY tokens are skipped (harmonic context).
        """
        id_list = [i for i in ids if 0 <= i < self.vocab_size]

        # Extract key_root from first KEY token
        key_root = 0
        if self._key_min_id >= 0:
            for tid in id_list:
                if self._key_min_id <= tid <= self._key_max_id:
                    key_root = (tid - self._key_min_id) % 12
                    break

        events: List[NoteEvent] = []
        cur_track = self.cfg.tracks[0]
        cur_bar   = 0
        cur_pos   = 0
        pending: Dict[str, int] = {}

        def _flush() -> None:
            if "chroma_id" in pending and "octave_id" in pending and "dur" in pending:
                pitch = self._chroma_octave_to_pitch(
                    pending["chroma_id"], pending["octave_id"], key_root
                )
                pitch = max(self.cfg.pitch_min, min(self.cfg.pitch_max, pitch))
                events.append(NoteEvent(
                    track=cur_track,
                    bar=cur_bar,
                    position=cur_pos,
                    pitch=pitch,
                    duration=pending["dur"],
                    velocity=self._velocity_from_bin(pending.get("vel", 16)),
                ))
            pending.clear()

        for tid in id_list:
            tok = self.id_to_token[tid]
            if tok == self.EOS:
                break
            if tok in (self.PAD, self.BOS, self.SEP):
                continue
            if tok.startswith("TRACK_"):
                _flush()
                cur_track = tok[len("TRACK_"):]
                cur_bar = 0
                cur_pos = 0
            elif tok == "BAR":
                _flush()
                cur_bar += 1
                cur_pos = 0
            elif tok.startswith("POS_"):
                _flush()
                cur_pos = int(tok.split("_", 1)[1])
            elif tok.startswith("CHROMA_"):
                if "chroma_id" in pending:
                    _flush()
                pending["chroma_id"] = tid
            elif tok.startswith("OCTAVE_"):
                pending["octave_id"] = tid
            elif tok.startswith("DUR_"):
                pending["dur"] = int(tok.split("_", 1)[1])
            elif tok.startswith("VEL_"):
                pending["vel"] = int(tok.split("_", 1)[1])
                _flush()
            # SCALE_DEGREE / QUALITY / CHORD_N / KEY / TEMPO: skip in note decode
        _flush()
        return events
