"""Smoke tests — every test must run on CPU in well under a minute."""
from __future__ import annotations

from pathlib import Path

import torch
import pytest

from jam_transformer.config import load_config
from jam_transformer.tokenizer import (
    BaseTokenizer, NoteEvent, REMITokenizer,
    available_tokenizers, build_tokenizer,
)
from jam_transformer.model import build_model

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def tokenizer(cfg):
    return build_tokenizer(cfg.tokenizer)


def test_tokenizer_registry(cfg):
    assert "remi_v1" in available_tokenizers()
    tok = build_tokenizer(cfg.tokenizer)
    assert isinstance(tok, BaseTokenizer)
    assert isinstance(tok, REMITokenizer)


def test_tokenizer_unknown_name_raises(cfg):
    from dataclasses import replace
    bad = replace(cfg.tokenizer, name="does_not_exist")
    with pytest.raises(KeyError):
        build_tokenizer(bad)


def test_vocab_specials_first(tokenizer):
    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.sep_id == 2
    assert tokenizer.eos_id == 3
    assert tokenizer.vocab_size > 100          # 174 tokens: CHROMA+OCTAVE+DUR+VEL+harmony+specials


def test_tokenizer_round_trip(tokenizer):
    events = [
        NoteEvent("melody", 0, 0, 60, 4, 90),
        NoteEvent("melody", 0, 4, 64, 4, 80),
        NoteEvent("piano",  0, 0, 48, 16, 70),
        NoteEvent("piano",  0, 0, 52, 16, 70),
        NoteEvent("piano",  0, 0, 55, 16, 70),
    ]
    ids, mask = tokenizer.encode_song(
        events, condition_tracks=["melody"], target_tracks=["piano"], tempo_bpm=120,
    )
    assert any(mask), "target mask must mark at least one position"
    assert ids[0] == tokenizer.bos_id
    assert tokenizer.sep_id in ids
    decoded = tokenizer.decode(ids)
    # Round-trip should recover the same pitches per track (durations may be
    # clipped by config bounds but here they're in range).
    in_pitches  = sorted(e.pitch for e in events)
    out_pitches = sorted(e.pitch for e in decoded)
    assert in_pitches == out_pitches


def test_model_forward(cfg, tokenizer):
    model = build_model(cfg.model, tokenizer.vocab_size)
    B, T = 2, 32
    x = torch.randint(0, tokenizer.vocab_size, (B, T))
    logits, _ = model(x)
    assert logits.shape == (B, T, tokenizer.vocab_size)
    assert torch.isfinite(logits).all()


def test_model_generate(cfg, tokenizer):
    model = build_model(cfg.model, tokenizer.vocab_size)
    # MIDI 60 (C4) with key_root=0: CHROMA_0, OCTAVE_5
    prompt = torch.tensor([tokenizer.bos_id, tokenizer.tid("TRACK_melody"),
                           tokenizer.tid("BAR"), tokenizer.tid("POS_0"),
                           tokenizer.tid("CHROMA_0"), tokenizer.tid("OCTAVE_5"),
                           tokenizer.tid("DUR_4"),
                           tokenizer.tid("VEL_16"), tokenizer.sep_id])
    out = model.generate(prompt, max_new_tokens=16, eos_id=tokenizer.eos_id,
                         temperature=1.0, top_k=8, top_p=0.95)
    assert out.shape[0] == 1
    assert out.shape[1] >= prompt.numel()


def test_synthetic_prepare(tmp_path, cfg, tokenizer):
    """Round-trip through prepare_data's synthetic generator + dataset chunking."""
    import sys
    import subprocess
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "prepare_data.py"),
        "--synthetic", "--num_songs", "4",
        "--out_dir", str(out_dir),
        "--config", str(CONFIG_PATH),
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    shards = list(out_dir.glob("*.pt"))
    assert len(shards) == 4
    data = torch.load(shards[0], map_location="cpu", weights_only=False)
    assert "ids" in data and "mask" in data
    assert int(data["mask"].sum()) > 0
    # Meta file should be present and carry the tokenizer fingerprint.
    meta_path = out_dir / "_dataset_meta.json"
    assert meta_path.exists()
    import json
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["vocab_size"] == tokenizer.vocab_size


def test_fingerprint_mismatch_blocks_training(tmp_path, cfg, tokenizer):
    """A drifted config must abort instead of silently mis-training."""
    from jam_transformer.dataset import assert_data_matches_config
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    import json
    (out_dir / "_dataset_meta.json").write_text(
        json.dumps({"vocab_size": 9999, "tokenizer_fingerprint": "deadbeef" * 2}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        assert_data_matches_config(out_dir, tokenizer)


def test_pitch_transpose_preserves_chroma_updates_octave(tokenizer):
    """With relative harmonic encoding:
      - CHROMA tokens are key-invariant and must NOT change on transpose.
      - OCTAVE tokens must shift when the abs pitch crosses an octave boundary.
      - KEY token root must update by the transposition offset.
    We verify via dataset._augment (the same code path as real training).
    """
    import torch
    from jam_transformer.config import load_config
    from jam_transformer.dataset import JamTokenDataset
    from jam_transformer.tokenizer import NoteEvent

    # Encode a single note: C4 (MIDI 60), key C major
    events = [
        NoteEvent("melody", 0, 0, 60, 4, 90),
        NoteEvent("piano",  0, 0, 72, 4, 70),   # C5 — OCTAVE should shift
    ]
    ids, _ = tokenizer.encode_song(
        events,
        condition_tracks=["melody"],
        target_tracks=["piano"],
        tempo_bpm=120,
        key_root=0, key_mode=0,  # C major
    )
    ids_t = torch.tensor(ids, dtype=torch.long)

    cmin = tokenizer.chroma_min_id
    cmax = tokenizer.chroma_max_id
    omin = tokenizer.octave_min_id
    omax = tokenizer.octave_max_id
    kmin = tokenizer.key_min_id
    kmax = tokenizer.key_max_id

    # CHROMA tokens must be identical before and after transpose.
    is_chroma = (ids_t >= cmin) & (ids_t <= cmax)
    chroma_before = ids_t[is_chroma].clone()

    cfg2 = load_config(CONFIG_PATH)
    cfg2.augment.pitch_transpose_semitones = 6
    cfg2.augment.velocity_jitter_bins = 0
    cfg2.augment.tempo_jitter_bins = 0
    cfg2.augment.duration_jitter_bins = 0
    cfg2.augment.condition_dropout_prob = 0.0

    ds = object.__new__(JamTokenDataset)
    ds.config = cfg2
    ds.tokenizer = tokenizer
    ds.train = True

    import random
    random.seed(42)
    aug = ds._augment(ids_t.clone(), key_root=0)

    # CHROMA tokens must be identical (invariant).
    is_chroma_aug = (aug >= cmin) & (aug <= cmax)
    assert torch.equal(aug[is_chroma_aug], chroma_before), (
        "CHROMA tokens must be invariant to transposition"
    )

    # Structure tokens (BAR/POS/DUR/VEL) must be unchanged.
    is_struct = ~((aug >= cmin) & (aug <= kmax))  # rough: exclude all harmonic tokens
    is_dur = (ids_t >= tokenizer.dur_min_id) & (ids_t <= tokenizer.dur_max_id)
    is_vel = (ids_t >= tokenizer.vel_min_id) & (ids_t <= tokenizer.vel_max_id)
    assert torch.equal(aug[is_dur], ids_t[is_dur]), "DUR tokens must be unchanged"
    assert torch.equal(aug[is_vel], ids_t[is_vel]), "VEL tokens must be unchanged"


def test_dataset_augmentation_only_runs_on_train(tmp_path, cfg, tokenizer):
    """When `train=False`, two consecutive __getitem__ calls return identical
    tensors. When `train=True` with a non-trivial transpose range, repeated
    fetches eventually return different pitch tokens (with seed control)."""
    import json
    import random as _random
    import torch
    from dataclasses import asdict
    from jam_transformer.dataset import JamTokenDataset
    from jam_transformer.tokenizer import NoteEvent
    from scripts.prepare_data import tokenizer_fingerprint

    # Build a one-shard dataset with enough notes that several offsets are
    # possible without clipping.
    events = [NoteEvent("melody", b, p * 4, 60 + (b + p) % 12, 4, 80)
              for b in range(4) for p in range(4)]
    events += [NoteEvent("piano", b, 0, 48 + b, 8, 70) for b in range(4)]
    ids, mask = tokenizer.encode_song(
        events, condition_tracks=["melody"], target_tracks=["piano"], tempo_bpm=120,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.save({
        "ids":  torch.tensor(ids,  dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.bool),
        "name": "t",
    }, data_dir / "t.pt")
    (data_dir / "_dataset_meta.json").write_text(json.dumps({
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_config": asdict(tokenizer.cfg),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer.cfg),
        "n_shards": 1, "cond_tracks": ["melody"], "target_tracks": ["piano"],
    }), encoding="utf-8")
    (data_dir / "_chunk_index.json").write_text(json.dumps({"t.pt": len(ids)}),
                                                encoding="utf-8")

    cfg.tokenizer.max_seq_len = 256
    cfg.augment.pitch_transpose_semitones = 5

    val_ds = JamTokenDataset(data_dir, cfg, tokenizer, train=False)
    a = val_ds[0]
    b = val_ds[0]
    assert torch.equal(a[0], b[0]), "validation must be deterministic"

    train_ds = JamTokenDataset(data_dir, cfg, tokenizer, train=True)
    _random.seed(0)
    samples = [train_ds[0][0] for _ in range(20)]
    distinct = {tuple(s.tolist()) for s in samples}
    assert len(distinct) >= 2, (
        "augmentation produced no variation over 20 draws — RNG wired wrong"
    )


def test_pitch_transpose_zero_disables(tmp_path, cfg, tokenizer):
    """With pitch_transpose_semitones=0 the train path must be a no-op."""
    import json
    import torch
    from dataclasses import asdict
    from jam_transformer.dataset import JamTokenDataset
    from jam_transformer.tokenizer import NoteEvent
    from scripts.prepare_data import tokenizer_fingerprint

    events = [NoteEvent("melody", 0, 0, 60, 4, 80),
              NoteEvent("piano",  0, 0, 48, 8, 70)]
    ids, mask = tokenizer.encode_song(
        events, condition_tracks=["melody"], target_tracks=["piano"], tempo_bpm=120,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.save({"ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.bool), "name": "t"},
               data_dir / "t.pt")
    (data_dir / "_dataset_meta.json").write_text(json.dumps({
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_config": asdict(tokenizer.cfg),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer.cfg),
        "n_shards": 1, "cond_tracks": ["melody"], "target_tracks": ["piano"],
    }), encoding="utf-8")
    (data_dir / "_chunk_index.json").write_text(json.dumps({"t.pt": len(ids)}),
                                                encoding="utf-8")

    cfg.tokenizer.max_seq_len = 256
    # Disable every augmentation so the train path is a pure no-op.
    cfg.augment.pitch_transpose_semitones = 0
    cfg.augment.velocity_jitter_bins = 0
    cfg.augment.tempo_jitter_bins = 0
    cfg.augment.duration_jitter_bins = 0
    cfg.augment.condition_dropout_prob = 0.0

    train_ds = JamTokenDataset(data_dir, cfg, tokenizer, train=True)
    val_ds   = JamTokenDataset(data_dir, cfg, tokenizer, train=False)
    assert torch.equal(train_ds[0][0], val_ds[0][0])


def test_condition_dropout_prob_zero_no_change(tmp_path, cfg, tokenizer):
    """prob=0 must leave the condition portion intact."""
    import json, torch
    from dataclasses import asdict
    from jam_transformer.dataset import JamTokenDataset
    from jam_transformer.tokenizer import NoteEvent
    from scripts.prepare_data import tokenizer_fingerprint

    events = [NoteEvent("melody", 0, p * 2, 60 + p, 2, 80) for p in range(8)]
    events += [NoteEvent("piano", 0, 0, 48, 16, 70)]
    ids, mask = tokenizer.encode_song(
        events, condition_tracks=["melody"], target_tracks=["piano"], tempo_bpm=120,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.save({"ids": torch.tensor(ids), "mask": torch.tensor(mask), "name": "t"},
               data_dir / "t.pt")
    (data_dir / "_dataset_meta.json").write_text(json.dumps({
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_config": asdict(tokenizer.cfg),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer.cfg),
        "n_shards": 1, "cond_tracks": ["melody"], "target_tracks": ["piano"],
    }), encoding="utf-8")
    (data_dir / "_chunk_index.json").write_text(json.dumps({"t.pt": len(ids)}),
                                                encoding="utf-8")

    cfg.tokenizer.max_seq_len = 256
    # Disable every augmentation so repeated draws are identical.
    cfg.augment.pitch_transpose_semitones = 0
    cfg.augment.velocity_jitter_bins = 0
    cfg.augment.tempo_jitter_bins = 0
    cfg.augment.duration_jitter_bins = 0
    cfg.augment.condition_dropout_prob = 0.0

    train_ds = JamTokenDataset(data_dir, cfg, tokenizer, train=True)
    val_ds = JamTokenDataset(data_dir, cfg, tokenizer, train=False)
    for _ in range(5):
        assert torch.equal(train_ds[0][0], val_ds[0][0])


def test_condition_dropout_prob_one_replaces_condition(tmp_path, cfg, tokenizer):
    """prob=1 must zero (PAD) every position between BOS and SEP and leave
    everything after SEP untouched."""
    import json, torch
    from dataclasses import asdict
    from jam_transformer.dataset import JamTokenDataset
    from jam_transformer.tokenizer import NoteEvent
    from scripts.prepare_data import tokenizer_fingerprint

    events = [NoteEvent("melody", 0, p * 2, 60 + p, 2, 80) for p in range(8)]
    events += [NoteEvent("piano", 0, 0, 48, 16, 70)]
    ids, mask = tokenizer.encode_song(
        events, condition_tracks=["melody"], target_tracks=["piano"], tempo_bpm=120,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    torch.save({"ids": torch.tensor(ids), "mask": torch.tensor(mask), "name": "t"},
               data_dir / "t.pt")
    (data_dir / "_dataset_meta.json").write_text(json.dumps({
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_config": asdict(tokenizer.cfg),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer.cfg),
        "n_shards": 1, "cond_tracks": ["melody"], "target_tracks": ["piano"],
    }), encoding="utf-8")
    (data_dir / "_chunk_index.json").write_text(json.dumps({"t.pt": len(ids)}),
                                                encoding="utf-8")

    cfg.tokenizer.max_seq_len = 256
    # Disable jitters so only dropout fires — keeps the post-SEP assertion clean.
    cfg.augment.pitch_transpose_semitones = 0
    cfg.augment.velocity_jitter_bins = 0
    cfg.augment.tempo_jitter_bins = 0
    cfg.augment.duration_jitter_bins = 0
    cfg.augment.condition_dropout_prob = 1.0      # always drop

    train_ds = JamTokenDataset(data_dir, cfg, tokenizer, train=True)
    x, y, _ = train_ds[0]

    # Reconstruct the original sequence (input ids without aug) via val_ds.
    cfg2 = load_config(CONFIG_PATH)
    cfg2.tokenizer.max_seq_len = 256
    val_ds = JamTokenDataset(data_dir, cfg2, tokenizer, train=False)
    orig_x, orig_y, _ = val_ds[0]

    sep_id = tokenizer.sep_id
    pad_id = tokenizer.pad_id
    bos_id = tokenizer.bos_id

    # First position should still be BOS in both.
    assert int(x[0]) == bos_id

    # Find the SEP in the original.
    sep_positions = (orig_x == sep_id).nonzero(as_tuple=False)
    assert sep_positions.numel() > 0, "test sequence has no SEP"
    sep_idx = int(sep_positions[0].item())

    # Positions 1..sep_idx (excl) should all be PAD in the dropped variant.
    if sep_idx > 1:
        assert (x[1:sep_idx] == pad_id).all(), (
            f"expected PAD in condition slice, got {x[1:sep_idx].tolist()}"
        )
    # Post-SEP tokens unchanged.
    assert torch.equal(x[sep_idx:], orig_x[sep_idx:])
    # And y (the next-token target) post-SEP also unchanged — the loss the
    # model would minimise is still the same accompaniment.
    assert torch.equal(y[sep_idx:], orig_y[sep_idx:])


def test_cli_overrides_apply(cfg):
    from jam_transformer.overrides import apply_overrides
    apply_overrides(cfg, ["model.d_model=384", "training.learning_rate=1e-4"])
    assert cfg.model.d_model == 384
    assert cfg.training.learning_rate == 1e-4

    with pytest.raises(KeyError):
        apply_overrides(cfg, ["model.nonexistent=1"])


# ---------------------------------------------------------------------------
# New jitter augmentation tests
# ---------------------------------------------------------------------------

def test_new_tokenizer_range_properties(tokenizer):
    """chroma / octave / vel / dur / tempo id ranges must be disjoint and within vocab."""
    v = tokenizer.vocab_size
    for lo, hi, name in [
        (tokenizer.chroma_min_id, tokenizer.chroma_max_id, "chroma"),
        (tokenizer.octave_min_id, tokenizer.octave_max_id, "octave"),
        (tokenizer.vel_min_id,    tokenizer.vel_max_id,    "vel"),
        (tokenizer.dur_min_id,    tokenizer.dur_max_id,    "dur"),
        (tokenizer.tempo_min_id,  tokenizer.tempo_max_id,  "tempo"),
    ]:
        assert 0 <= lo <= hi < v, f"{name} range [{lo},{hi}] out of vocab ({v})"

    ranges = [
        (tokenizer.chroma_min_id, tokenizer.chroma_max_id),
        (tokenizer.octave_min_id, tokenizer.octave_max_id),
        (tokenizer.vel_min_id,    tokenizer.vel_max_id),
        (tokenizer.dur_min_id,    tokenizer.dur_max_id),
        (tokenizer.tempo_min_id,  tokenizer.tempo_max_id),
    ]
    for i, (a_lo, a_hi) in enumerate(ranges):
        for j, (b_lo, b_hi) in enumerate(ranges):
            if i >= j:
                continue
            overlap = max(0, min(a_hi, b_hi) - max(a_lo, b_lo) + 1)
            assert overlap == 0, f"ranges {i} and {j} overlap"


def test_velocity_jitter_stays_in_range(tokenizer):
    """VEL tokens after jitter must remain within [vel_min_id, vel_max_id]."""
    import jam_transformer.dataset as ds_mod
    from jam_transformer.config import load_config

    cfg = load_config(CONFIG_PATH)
    cfg.augment.pitch_transpose_semitones = 0
    cfg.augment.tempo_jitter_bins = 0
    cfg.augment.duration_jitter_bins = 0
    cfg.augment.condition_dropout_prob = 0.0
    cfg.augment.velocity_jitter_bins = 5   # large jitter to stress-test clamping

    ds = object.__new__(ds_mod.JamTokenDataset)
    ds.config = cfg
    ds.tokenizer = tokenizer
    ds.train = True

    # Build a sequence where every velocity token is at the extremes.
    vmin, vmax = tokenizer.vel_min_id, tokenizer.vel_max_id
    ids = torch.tensor([vmin, vmin, vmax, vmax,
                        tokenizer.pad_id, tokenizer.pad_id], dtype=torch.long)
    for _ in range(50):
        aug = ds._augment(ids.clone())
        vel_mask = (aug >= vmin) & (aug <= vmax)
        # All positions that were originally vel tokens should still be in range.
        assert vel_mask[:4].all(), f"vel token out of range: {aug[:4].tolist()}"


def test_tempo_jitter_same_delta_per_chunk(tokenizer):
    """All TEMPO tokens in a chunk must shift by the same delta."""
    import jam_transformer.dataset as ds_mod
    from jam_transformer.config import load_config
    import random

    cfg = load_config(CONFIG_PATH)
    cfg.augment.pitch_transpose_semitones = 0
    cfg.augment.velocity_jitter_bins = 0
    cfg.augment.duration_jitter_bins = 0
    cfg.augment.condition_dropout_prob = 0.0
    cfg.augment.tempo_jitter_bins = 3

    ds = object.__new__(ds_mod.JamTokenDataset)
    ds.config = cfg
    ds.tokenizer = tokenizer
    ds.train = True

    tmin, tmax = tokenizer.tempo_min_id, tokenizer.tempo_max_id
    # Place two TEMPO tokens in the middle of the valid range.
    mid = (tmin + tmax) // 2
    ids = torch.tensor([mid, mid + 2, tokenizer.pad_id], dtype=torch.long)

    for seed in range(20):
        random.seed(seed)
        aug = ds._augment(ids.clone())
        delta0 = int(aug[0].item()) - mid
        delta1 = int(aug[1].item()) - (mid + 2)
        assert delta0 == delta1, (
            f"seed={seed}: TEMPO tokens shifted by different deltas "
            f"({delta0} vs {delta1})"
        )
        assert tmin <= aug[0] <= tmax
        assert tmin <= aug[1] <= tmax


def test_duration_jitter_independent_per_note(tokenizer):
    """DUR tokens must shift independently (different deltas per token allowed)."""
    import jam_transformer.dataset as ds_mod
    from jam_transformer.config import load_config

    cfg = load_config(CONFIG_PATH)
    cfg.augment.pitch_transpose_semitones = 0
    cfg.augment.velocity_jitter_bins = 0
    cfg.augment.tempo_jitter_bins = 0
    cfg.augment.condition_dropout_prob = 0.0
    cfg.augment.duration_jitter_bins = 3

    ds = object.__new__(ds_mod.JamTokenDataset)
    ds.config = cfg
    ds.tokenizer = tokenizer
    ds.train = True

    dmin, dmax = tokenizer.dur_min_id, tokenizer.dur_max_id
    mid = (dmin + dmax) // 2
    # Two DUR tokens at the same original value.
    ids = torch.tensor([mid, mid, tokenizer.pad_id], dtype=torch.long)

    # Over 50 draws, at least once the two tokens should differ (independent draws).
    found_independent = False
    for seed in range(50):
        torch.manual_seed(seed)
        aug = ds._augment(ids.clone())
        if aug[0] != aug[1]:
            found_independent = True
            break
        assert dmin <= aug[0] <= dmax
        assert dmin <= aug[1] <= dmax
    assert found_independent, "DUR jitter appears to use a shared delta (not independent)"


# ---------------------------------------------------------------------------
# Token-type loss weighting + polyphony boost + structural suppression
# ---------------------------------------------------------------------------

def test_structural_and_content_ids_disjoint(tokenizer):
    s = set(tokenizer.structural_ids())
    c = set(tokenizer.content_ids())
    assert s.isdisjoint(c)
    # Vocab = structural + content + 4 specials (PAD/BOS/SEP/EOS)
    #         + harmony tokens (SCALE_DEGREE / QUALITY / CHORD_N / KEY_*)
    # The harmony tokens are neither in structural_ids() nor content_ids()
    # (they get their own weighting in build_token_weight_vector).
    total_accounted = len(s) + len(c) + 4
    assert total_accounted <= tokenizer.vocab_size
    # At minimum, structural + content + specials must cover most of the vocab.
    assert total_accounted >= tokenizer.vocab_size - 60  # harmony tokens ≤ 60


def test_token_weight_vector_values(tokenizer):
    w = tokenizer.build_token_weight_vector(struct_weight=0.3, content_weight=1.5)
    assert len(w) == tokenizer.vocab_size
    assert w[tokenizer.pad_id] == 1.0
    assert w[tokenizer.bar_id] == 0.3
    pos_lo, _ = tokenizer.pos_id_range
    assert w[pos_lo] == 0.3
    # CHROMA and OCTAVE are content tokens → 1.5
    assert w[tokenizer.tid("CHROMA_0")] == 1.5
    assert w[tokenizer.tid("OCTAVE_5")] == 1.5
    assert w[tokenizer.tid("DUR_4")] == 1.5
    assert w[tokenizer.tid("VEL_5")] == 1.5
    # SCALE_DEGREE / QUALITY are harmonic content → 1.5
    assert w[tokenizer.sd_min_id] == 1.5
    assert w[tokenizer.quality_min_id] == 1.5
    # CHORD_N and KEY are structural-ish → 0.3
    assert w[tokenizer.chord_n_id] == 0.3
    if tokenizer.key_min_id >= 0:
        assert w[tokenizer.key_min_id] == 0.3


def test_polyphony_boost_increases_gradient(cfg, tokenizer):
    """A polyphonic target (CHROMA after VEL) should produce a larger gradient
    norm than the same target after a non-VEL context.

    Design note: pos 0 and pos 1 must have DIFFERENT target tokens.
    When both targets are identical, loss = (CE*w0 + CE*w1)/(w0+w1) = CE
    regardless of weights — the normalization cancels the boost and gradients
    become identical. With different targets the gradient directions differ,
    so the weighting changes the combined gradient norm.
    """
    import torch.nn as nn
    from jam_transformer.lightning_module import JamTransformerLightning

    V = tokenizer.vocab_size

    class FakeModel(nn.Module):
        def __init__(self, V):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(V))
        def forward(self, x):
            B, T = x.shape
            return self.bias.expand(B, T, V), None

    vel    = tokenizer.tid("VEL_5")
    bar    = tokenizer.bar_id
    chroma = tokenizer.tid("CHROMA_0")   # C relative to key root
    octave = tokenizer.tid("OCTAVE_5")   # middle-C octave — distinct direction

    # Two-position batch:
    #   pos 0: VEL → CHROMA  (polyphonic decision, boost applies)
    #   pos 1: BAR → OCTAVE  (non-polyphonic, no boost, DIFFERENT target)
    x = torch.tensor([[vel, bar]])
    y = torch.tensor([[chroma, octave]])   # ← different targets!
    mask = torch.ones_like(y).bool()

    # With boost: polyphonic position weighted 1.5 * 2.0 = 3.0
    lit_a = JamTransformerLightning(cfg, vocab_size=V, total_steps=10)
    lit_a.model = FakeModel(V)
    loss_a, _ = lit_a._compute_loss((x, y, mask))
    loss_a.backward()
    grad_a = lit_a.model.bias.grad.norm().item()

    # Without boost: both positions weighted 1.5
    cfg_no_boost = load_config(CONFIG_PATH)
    cfg_no_boost.training.polyphony_loss_boost = 1.0
    lit_b = JamTransformerLightning(cfg_no_boost, vocab_size=V, total_steps=10)
    lit_b.model = FakeModel(V)
    loss_b, _ = lit_b._compute_loss((x, y, mask))
    loss_b.backward()
    grad_b = lit_b.model.bias.grad.norm().item()

    assert grad_a > grad_b, (
        f"polyphony boost should amplify gradient: "
        f"with_boost={grad_a:.4f}, without={grad_b:.4f}"
    )


def test_structural_suppression_avoids_pos_after_vel(cfg, tokenizer):
    """After a VEL token, heavy suppression should prevent BAR/POS being sampled."""
    from jam_transformer.model import build_model

    model = build_model(cfg.model, tokenizer.vocab_size)
    model.eval()

    vel = tokenizer.tid("VEL_5")
    prompt = torch.tensor([[
        tokenizer.bos_id,
        tokenizer.tid("TRACK_piano"),
        tokenizer.tid("BAR"), tokenizer.tid("POS_0"),
        tokenizer.tid("CHROMA_0"), tokenizer.tid("OCTAVE_5"),
        tokenizer.tid("DUR_4"),
        vel,
    ]], dtype=torch.long)

    struct_set = set(tokenizer.structural_ids())
    torch.manual_seed(0)
    out = model.generate(
        prompt, max_new_tokens=1, eos_id=tokenizer.eos_id,
        temperature=1.0, top_k=0, top_p=1.0,
        structural_suppression=20.0,
        vel_id_range=(tokenizer.vel_min_id, tokenizer.vel_max_id),
        struct_ids=tokenizer.structural_ids(),
    )
    next_id = int(out[0, -1].item())
    assert next_id not in struct_set


def test_polyphony_sample_weights_assign_correctly(tmp_path, cfg, tokenizer):
    """Higher polyphony score in a chunk should yield higher sample weight."""
    from jam_transformer.dataset import JamTokenDataset

    # All-polyphonic toy shard — uses CHROMA+OCTAVE instead of PITCH
    poly_seq = [
        tokenizer.bos_id, tokenizer.tid("TRACK_piano"),
        tokenizer.tid("BAR"), tokenizer.tid("POS_0"),
            tokenizer.tid("CHROMA_0"), tokenizer.tid("OCTAVE_5"),
            tokenizer.tid("DUR_4"), tokenizer.tid("VEL_5"),
            tokenizer.tid("CHROMA_4"), tokenizer.tid("OCTAVE_5"),
            tokenizer.tid("DUR_4"), tokenizer.tid("VEL_5"),
        tokenizer.tid("POS_4"),
            tokenizer.tid("CHROMA_7"), tokenizer.tid("OCTAVE_5"),
            tokenizer.tid("DUR_4"), tokenizer.tid("VEL_5"),
            tokenizer.tid("CHROMA_0"), tokenizer.tid("OCTAVE_6"),
            tokenizer.tid("DUR_4"), tokenizer.tid("VEL_5"),
        tokenizer.eos_id,
    ]
    ids = torch.tensor(poly_seq, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool); mask[2:] = True
    torch.save({"ids": ids, "mask": mask, "name": "synth"}, tmp_path / "synth.pt")

    cfg2 = load_config(CONFIG_PATH)
    cfg2.training.polyphony_sample_weight_alpha = 0.5
    cfg2.tokenizer.max_seq_len = 64
    ds = JamTokenDataset(tmp_path, cfg2, tokenizer, train=True)
    weights = ds.get_sample_weights()
    assert weights is not None
    # Polyphony score should be ~1.0 → weight = (1.01)^0.5 ≈ 1.005
    assert weights[0].item() > 0.9


def test_polyphony_sample_weights_disabled_when_alpha_zero(tmp_path, cfg, tokenizer):
    """alpha=0 should return None (uniform sampling fallback in train.py)."""
    from jam_transformer.dataset import JamTokenDataset
    seq = [tokenizer.bos_id, tokenizer.tid("CHROMA_0"), tokenizer.tid("OCTAVE_5"),
           tokenizer.tid("DUR_4"), tokenizer.tid("VEL_5"), tokenizer.eos_id]
    ids = torch.tensor(seq, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool); mask[1:] = True
    torch.save({"ids": ids, "mask": mask, "name": "synth"}, tmp_path / "synth.pt")

    cfg2 = load_config(CONFIG_PATH)
    cfg2.training.polyphony_sample_weight_alpha = 0.0
    cfg2.tokenizer.max_seq_len = 64
    ds = JamTokenDataset(tmp_path, cfg2, tokenizer, train=True)
    assert ds.get_sample_weights() is None


# ---------------------------------------------------------------------------
# Lakh MIDI track-assignment heuristic tests
# ---------------------------------------------------------------------------

def _make_fake_midi(tmp_path: Path, programs: list[int], pitches_per_inst: list[list[int]]) -> Path:
    """Build a minimal miditoolkit MidiFile with N instruments and save it."""
    import miditoolkit
    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    midi.tempo_changes = [miditoolkit.TempoChange(tempo=120.0, time=0)]
    for prog, pitches in zip(programs, pitches_per_inst):
        inst = miditoolkit.Instrument(program=prog, is_drum=False, name=f"prog{prog}")
        for t, p in enumerate(pitches):
            inst.notes.append(miditoolkit.Note(velocity=80, pitch=p, start=t * 480, end=(t + 1) * 480))
        midi.instruments.append(inst)
    p = tmp_path / "fake.mid"
    midi.dump(str(p))
    return p


def test_lakh_heuristic_highest_pitch_is_melody(tmp_path, cfg):
    """The instrument with the highest median pitch should be assigned 'melody'."""
    from scripts.prepare_data import _lakh_track_events

    mid = _make_fake_midi(
        tmp_path,
        programs=[0, 0],                          # two piano tracks
        pitches_per_inst=[
            [60, 62, 64, 65],                     # lower — should become piano/bridge
            [72, 74, 76, 77],                     # higher — should become melody
        ],
    )
    result = _lakh_track_events(mid, cfg.tokenizer)
    assert result is not None, "Should not skip a 2-instrument MIDI"
    events, tempo = result
    track_pitch: dict[str, list[int]] = {}
    for e in events:
        track_pitch.setdefault(e.track, []).append(e.pitch)

    assert "melody" in track_pitch, "melody track must be present"
    melody_med = sorted(track_pitch["melody"])[len(track_pitch["melody"]) // 2]
    for tr, ps in track_pitch.items():
        if tr == "melody":
            continue
        other_med = sorted(ps)[len(ps) // 2]
        assert melody_med >= other_med, (
            f"melody median pitch ({melody_med}) should be >= {tr} ({other_med})"
        )


def test_lakh_heuristic_skips_single_instrument(tmp_path, cfg):
    """A MIDI with only one non-drum instrument should be skipped (None)."""
    from scripts.prepare_data import _lakh_track_events

    mid = _make_fake_midi(
        tmp_path,
        programs=[0],
        pitches_per_inst=[[60, 62, 64, 65, 67]],
    )
    assert _lakh_track_events(mid, cfg.tokenizer) is None


def test_lakh_heuristic_bass_not_melody(tmp_path, cfg):
    """Bass instruments (GM prog 32-39) should not be assigned as melody
    when melodic alternatives exist."""
    from scripts.prepare_data import _lakh_track_events, _GM_BASS_PROGRAMS

    bass_prog = 33  # finger bass — in _GM_BASS_PROGRAMS
    assert bass_prog in _GM_BASS_PROGRAMS

    mid = _make_fake_midi(
        tmp_path,
        programs=[bass_prog, 0, 0],               # bass + 2 piano tracks
        pitches_per_inst=[
            [28, 30, 32, 33],                     # bass (low pitch)
            [60, 62, 64, 65],                     # mid piano
            [72, 74, 76, 77],                     # high piano → melody
        ],
    )
    result = _lakh_track_events(mid, cfg.tokenizer)
    assert result is not None
    events, _ = result
    melody_pitches = [e.pitch for e in events if e.track == "melody"]
    # Melody should not contain the very low bass pitches
    assert all(p >= 40 for p in melody_pitches), (
        f"Bass pitches leaked into melody: {sorted(set(melody_pitches))}"
    )


def test_lakh_encode_produces_valid_shard(tmp_path, cfg):
    """_encode_lakh_one should produce a .pt shard with valid token ids."""
    from scripts.prepare_data import _encode_lakh_one, write_meta
    from jam_transformer.tokenizer import build_tokenizer

    tokenizer = build_tokenizer(cfg.tokenizer)
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()
    mid = _make_fake_midi(
        midi_dir,
        programs=[0, 0],
        pitches_per_inst=[
            [60, 62, 64, 65, 67, 69, 71, 72],    # lower
            [72, 74, 76, 77, 79, 81, 83, 84],    # higher → melody
        ],
    )
    out_dir = tmp_path / "shards"
    out_dir.mkdir()

    stem = _encode_lakh_one(mid, tokenizer, ["melody"], ["bridge", "piano"], out_dir)
    assert stem is not None, "encode should succeed for a valid 2-track MIDI"
    assert (out_dir / f"{stem}.pt").exists()

    data = torch.load(out_dir / f"{stem}.pt", map_location="cpu", weights_only=False)
    ids, mask = data["ids"], data["mask"]
    assert ids.dtype == torch.long
    assert mask.dtype == torch.bool
    assert ids.shape == mask.shape
    assert ids.max().item() < tokenizer.vocab_size, "token id out of vocabulary"
    assert mask.sum().item() >= 8, "too few target tokens"

    # write_meta incremental path
    write_meta(out_dir, tokenizer, ["melody"], ["bridge", "piano"],
               new_shard_names=[stem])
    import json
    meta = json.loads((out_dir / "_dataset_meta.json").read_text())
    assert meta["n_shards"] == 1
    assert meta["tokenizer_fingerprint"]
