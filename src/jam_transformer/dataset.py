"""Dataset over pre-tokenized .pt shards (relative harmonic encoding).

Each .pt shard is a dict::

    "ids":      LongTensor of token ids
    "mask":     BoolTensor; True where the position is accompaniment target
    "name":     str (source song id)
    "key_root": int (0-11, or -1 if unknown)
    "key_mode": int (0=major, 1=minor, or -1 if unknown)

key_root / key_mode are stored in the shard so every windowed chunk can
correctly augment OCTAVE tokens (which depend on the absolute key to
reconstruct).  See tokenizer module for the key-invariance proof.

Augmentation changes vs the absolute-encoding version
------------------------------------------------------
Pitch transposition now only needs to:
  1. Shift the KEY token root by `offset` (mod 12).
  2. Update OCTAVE tokens for notes whose abs_pitch crosses an octave
     boundary (new_octave = (old_abs + offset) // 12).
  CHROMA and SCALE_DEGREE tokens are mathematically invariant to
  transposition and must NOT be modified.

The range `pitch_transpose_semitones` can be reduced to ~5 (vs 12 before)
since all harmonic content is already key-relative in the .pt shards.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from jam_transformer.config import AppConfig
from jam_transformer.tokenizer import BaseTokenizer
from jam_transformer.logger import logger


META_FILENAME = "_dataset_meta.json"


def _fingerprint(cfg) -> str:
    return hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True).encode()
    ).hexdigest()[:16]


def load_dataset_meta(data_dir: str | Path) -> Optional[dict]:
    p = Path(data_dir) / META_FILENAME
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def assert_data_matches_config(data_dir: str | Path, tokenizer: BaseTokenizer) -> None:
    meta = load_dataset_meta(data_dir)
    if meta is None:
        logger.warning(
            f"No {META_FILENAME} found under {data_dir}. "
            "Re-run prepare_data.py with the current config."
        )
        return
    have = _fingerprint(tokenizer.cfg)
    want = meta.get("tokenizer_fingerprint", "")
    if have != want:
        raise SystemExit(
            f"Tokenizer fingerprint mismatch.\n"
            f"  data dir: {data_dir} → {want}\n"
            f"  current:  configs/... → {have}\n"
            f"Re-run prepare_data.py with the current config."
        )
    if meta.get("vocab_size") != tokenizer.vocab_size:
        raise SystemExit(
            f"Vocab size mismatch (data={meta.get('vocab_size')} "
            f"vs current={tokenizer.vocab_size}). Re-prepare the data."
        )


class JamTokenDataset(Dataset):
    """Loads tokenized songs from `data_dir/*.pt` and returns fixed-length crops."""

    def __init__(
        self,
        data_dir: str | Path,
        config: AppConfig,
        tokenizer: BaseTokenizer,
        train: bool = True,
    ):
        self.config    = config
        self.tokenizer = tokenizer
        self.train     = train
        self.max_len   = config.tokenizer.max_seq_len
        self.data_dir  = Path(data_dir)

        self.shards: List[Path] = sorted(
            p for p in self.data_dir.glob("*.pt") if not p.name.startswith("_")
        )
        if not self.shards:
            logger.warning(f"No .pt shards found under {self.data_dir}")

        self._chunks: List[Tuple[int, int]] = []
        stride = max(1, self.max_len // 2) if train else self.max_len

        cache_path = self.data_dir / "_chunk_index.json"
        shard_lens: Optional[dict] = None
        if cache_path.exists():
            try:
                shard_lens = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Couldn't read {cache_path}: {e}. Falling back.")

        for si, shard in enumerate(self.shards):
            if shard_lens is not None and shard.name in shard_lens:
                entry = shard_lens[shard.name]
                if isinstance(entry, dict):
                    n        = int(entry["n"])
                    cond_len = int(entry.get("sep", 0))
                else:
                    # legacy format: plain int (no sep info)
                    n        = int(entry)
                    cond_len = 0
            else:
                data     = torch.load(shard, map_location="cpu", weights_only=False)
                n        = int(data["ids"].numel())
                nz       = data["mask"].nonzero()
                cond_len = int(nz[0].item()) if nz.numel() > 0 else n

            if n < 8:
                continue
            for o in range(0, max(1, n - 1), stride):
                # Skip windows that end entirely within the condition (melody-only)
                # — those produce target_mask=all-False, contributing zero loss.
                if cond_len > 0 and o + self.max_len <= cond_len:
                    continue
                self._chunks.append((si, o))

        self._poly_scores: Optional[List[float]] = None
        if train and getattr(config.training, "polyphony_sample_weight_alpha", 0.0) > 0.0:
            self._poly_scores = self._compute_polyphony_scores()

    def __len__(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Polyphony scoring (CHROMA tokens replace PITCH tokens)
    # ------------------------------------------------------------------
    def _compute_polyphony_scores(self) -> List[float]:
        pmin    = self.tokenizer.chroma_min_id
        pmax    = self.tokenizer.chroma_max_id
        bar_id  = self.tokenizer.bar_id
        pos_lo, pos_hi = self.tokenizer.pos_id_range
        tr_lo,  tr_hi  = self.tokenizer.track_id_range

        def _score_segment(ids_seg: torch.Tensor) -> float:
            n_seg, n_poly, n_in_seg = 0, 0, 0
            for tid in ids_seg.tolist():
                if tid == bar_id or (pos_lo <= tid <= pos_hi) or (tr_lo <= tid <= tr_hi):
                    if n_in_seg > 0:
                        n_seg += 1
                        if n_in_seg >= 2:
                            n_poly += 1
                    n_in_seg = 0
                elif pmin <= tid <= pmax:
                    n_in_seg += 1
            if n_in_seg > 0:
                n_seg += 1
                if n_in_seg >= 2:
                    n_poly += 1
            return n_poly / max(n_seg, 1)

        cache: dict = {}
        scores: List[float] = []
        for si, start in self._chunks:
            if si not in cache:
                data = torch.load(self.shards[si], map_location="cpu", weights_only=False)
                cache[si] = data["ids"]
            ids_seg = cache[si][start: start + self.max_len].long()
            scores.append(_score_segment(ids_seg))
        return scores

    def get_sample_weights(self) -> Optional["torch.Tensor"]:
        alpha = float(getattr(self.config.training, "polyphony_sample_weight_alpha", 0.0))
        if alpha <= 0.0 or self._poly_scores is None:
            return None
        import torch as _t
        scores  = _t.tensor(self._poly_scores, dtype=_t.float32)
        weights = (scores + 0.01).pow(alpha)
        return weights

    # ------------------------------------------------------------------
    # Augmentation — relative encoding version
    # ------------------------------------------------------------------
    def _augment(self, ids: torch.Tensor, key_root: int = 0) -> torch.Tensor:
        """Apply train-time augmentations.

        1. **Pitch transposition** (pitch_transpose_semitones):
           a. Shift KEY token root by `offset` (mod 12).
           b. Update OCTAVE tokens where abs_pitch crosses octave boundary.
           CHROMA and SCALE_DEGREE are invariant — not modified.

        2. **Velocity jitter** (velocity_jitter_bins): per-token.
        3. **Tempo jitter** (tempo_jitter_bins): single delta for all TEMPO_*.
        4. **Duration jitter** (duration_jitter_bins): per-token.
        5. **Condition dropout** (condition_dropout_prob): PAD melody section.
        """
        aug = getattr(self.config, "augment", None)
        if aug is None:
            return ids
        import random as _r

        # ---- 1. Pitch transposition -----------------------------------------
        half = int(getattr(aug, "pitch_transpose_semitones", 0) or 0)
        if half > 0:
            kmin = self.tokenizer.key_min_id
            kmax = self.tokenizer.key_max_id
            omin = self.tokenizer.octave_min_id
            omax = self.tokenizer.octave_max_id
            obase = self.tokenizer.octave_base
            cmin = self.tokenizer.chroma_min_id
            pitch_lo = self.config.tokenizer.pitch_min
            pitch_hi = self.config.tokenizer.pitch_max

            # Compute safe shift range from (CHROMA, OCTAVE) pairs
            abs_pitches: list[int] = []
            for i in range(1, len(ids)):
                oct_id    = int(ids[i].item())
                chroma_id = int(ids[i - 1].item())
                if omin <= oct_id <= omax and cmin <= chroma_id < omin:
                    c = chroma_id - cmin
                    o = oct_id - omin + obase
                    abs_pitches.append(o * 12 + (c + key_root) % 12)

            if abs_pitches:
                lo = -min(half, min(abs_pitches) - pitch_lo)
                hi =  min(half, pitch_hi - max(abs_pitches))
                if hi >= lo:
                    offset = _r.randint(lo, hi)
                    if offset != 0:
                        ids = ids.clone()
                        # 1a. Update KEY root
                        if kmin >= 0:
                            key_mask = (ids >= kmin) & (ids <= kmax)
                            if key_mask.any():
                                rel = ids[key_mask] - kmin
                                mode = rel // 12
                                new_root = (rel % 12 + offset) % 12
                                ids[key_mask] = kmin + mode * 12 + new_root
                        # 1b. Update OCTAVE tokens (CHROMA unchanged)
                        for i in range(1, len(ids)):
                            oct_id    = int(ids[i].item())
                            chroma_id = int(ids[i - 1].item())
                            if omin <= oct_id <= omax and cmin <= chroma_id < omin:
                                c = chroma_id - cmin
                                o = oct_id - omin + obase
                                abs_p   = o * 12 + (c + key_root) % 12
                                new_abs = max(pitch_lo, min(pitch_hi, abs_p + offset))
                                new_o   = new_abs // 12
                                new_oid = omin + (new_o - obase)
                                new_oid = max(omin, min(omax, new_oid))
                                ids[i]  = new_oid

        # ---- 2. Velocity jitter --------------------------------------------
        vel_jitter = int(getattr(aug, "velocity_jitter_bins", 0) or 0)
        if vel_jitter > 0:
            vmin = self.tokenizer.vel_min_id
            vmax = self.tokenizer.vel_max_id
            is_vel = (ids >= vmin) & (ids <= vmax)
            if bool(is_vel.any()):
                n_vel = int(is_vel.sum().item())
                deltas = torch.randint(-vel_jitter, vel_jitter + 1, (n_vel,))
                new_ids = ids.clone()
                vel_pos = is_vel.nonzero(as_tuple=False).squeeze(1)
                new_ids[vel_pos] = (ids[vel_pos] + deltas).clamp(vmin, vmax)
                ids = new_ids

        # ---- 3. Tempo jitter -----------------------------------------------
        tempo_jitter = int(getattr(aug, "tempo_jitter_bins", 0) or 0)
        if tempo_jitter > 0:
            tmin = self.tokenizer.tempo_min_id
            tmax = self.tokenizer.tempo_max_id
            is_tempo = (ids >= tmin) & (ids <= tmax)
            if bool(is_tempo.any()):
                offset = _r.randint(-tempo_jitter, tempo_jitter)
                if offset != 0:
                    ids = ids.clone()
                    ids[is_tempo] = (ids[is_tempo] + offset).clamp(tmin, tmax)

        # ---- 4. Duration jitter --------------------------------------------
        dur_jitter = int(getattr(aug, "duration_jitter_bins", 0) or 0)
        if dur_jitter > 0:
            dmin = self.tokenizer.dur_min_id
            dmax = self.tokenizer.dur_max_id
            is_dur = (ids >= dmin) & (ids <= dmax)
            if bool(is_dur.any()):
                n_dur = int(is_dur.sum().item())
                deltas = torch.randint(-dur_jitter, dur_jitter + 1, (n_dur,))
                dur_pos = is_dur.nonzero(as_tuple=False).squeeze(1)
                ids = ids.clone()
                ids[dur_pos] = (ids[dur_pos] + deltas).clamp(dmin, dmax)

        # ---- 5. Condition dropout (CFG) ------------------------------------
        p_drop = float(getattr(aug, "condition_dropout_prob", 0.0) or 0.0)
        if p_drop > 0.0 and _r.random() < p_drop:
            sep_id  = self.tokenizer.sep_id
            sep_pos = (ids == sep_id).nonzero(as_tuple=False)
            if sep_pos.numel() > 0:
                sep_idx = int(sep_pos[0].item())
                if sep_idx > 1:
                    ids = ids.clone()
                    ids[1:sep_idx] = self.tokenizer.pad_id

        return ids

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        shard_idx, start = self._chunks[idx]
        if not hasattr(self, "_shard_cache"):
            self._shard_cache: dict = {}
        if shard_idx not in self._shard_cache:
            data = torch.load(self.shards[shard_idx], map_location="cpu", weights_only=False)
            self._shard_cache[shard_idx] = (
                data["ids"].long(),
                data["mask"].bool(),
                int(data.get("key_root", -1)),
                int(data.get("key_mode", -1)),
            )
        ids_full, mask_full, key_root, key_mode = self._shard_cache[shard_idx]
        end  = start + self.max_len
        ids  = ids_full[start:end].clone()
        mask = mask_full[start:end].clone()
        return ids, mask, key_root, key_mode

    def __getitem__(self, idx: int):
        ids, target_mask, key_root, key_mode = self._load(idx)
        kr = key_root if key_root >= 0 else 0  # fallback C when unknown

        if self.train:
            ids = self._augment(ids, key_root=kr)

        pad_id = self.tokenizer.pad_id
        cur = ids.shape[0]
        if cur < self.max_len:
            pad = self.max_len - cur
            ids = torch.cat([ids, torch.full((pad,), pad_id, dtype=ids.dtype)])
            target_mask = torch.cat([
                target_mask, torch.zeros(pad, dtype=target_mask.dtype)
            ])

        x = ids[:-1].contiguous()
        y = ids[1:].contiguous()
        loss_mask = target_mask[1:].contiguous() & (y != pad_id)
        return x, y, loss_mask
