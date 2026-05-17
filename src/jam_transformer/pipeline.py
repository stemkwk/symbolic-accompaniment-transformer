"""Core inference pipeline helpers shared by CLI and web demo."""
from __future__ import annotations

from pathlib import Path

import torch

from jam_transformer.logger import logger
from jam_transformer.midi_io import events_to_midi, humanize_midi, midi_to_events
from jam_transformer.tokenizer import BaseTokenizer


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def load_checkpoint(ckpt_path: str, cfg, vocab_size: int):
    """Load a checkpoint into a JamTransformerLightning module.

    Handles torch.compile'd checkpoints transparently by stripping the
    ``_orig_mod.`` prefix when the current module is not compiled.
    """
    import pytorch_lightning as pl  # noqa: F401
    from jam_transformer.lightning_module import JamTransformerLightning

    lit = JamTransformerLightning(config=cfg, vocab_size=vocab_size, total_steps=1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("state_dict", ckpt)

    has_orig_mod = any("._orig_mod." in k for k in raw_sd)
    model_is_compiled = hasattr(lit.model, "_orig_mod")
    if has_orig_mod and not model_is_compiled:
        raw_sd = {k.replace("._orig_mod.", ".", 1): v for k, v in raw_sd.items()}
        logger.info("Detected torch.compile checkpoint — stripped '_orig_mod.' prefix.")

    missing, unexpected = lit.load_state_dict(raw_sd, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if not missing and not unexpected:
        logger.info("State dict loaded cleanly.")
    return lit


def estimate_key_from_midi(midi_path: Path) -> tuple[int, int] | None:
    """Estimate global key via Krumhansl-Schmuckler pitch-class profile.

    Returns (root_0_11, mode_0_1) or None on failure.
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

    major = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    def _corr(profile: list[float], shift: int) -> float:
        rotated = profile[shift:] + profile[:shift]
        mean_p = sum(pc) / 12
        mean_r = sum(rotated) / 12
        num = sum((pc[i] - mean_p) * (rotated[i] - mean_r) for i in range(12))
        dp = sum((pc[i] - mean_p) ** 2 for i in range(12)) ** 0.5
        dr = sum((rotated[i] - mean_r) ** 2 for i in range(12)) ** 0.5
        return 0.0 if dp * dr < 1e-9 else num / (dp * dr)

    best_corr, best_root, best_mode = -999.0, 0, 0
    for root in range(12):
        for mode, profile in enumerate([major, minor]):
            c = _corr(profile, root)
            if c > best_corr:
                best_corr, best_root, best_mode = c, root, mode
    return (best_root, best_mode)


def build_prompt(
    melody_midi: Path,
    tokenizer: BaseTokenizer,
    cond_tracks: list[str],
    tempo_override: float | None = None,
    track_name_override: dict[str, str] | None = None,
) -> tuple[torch.Tensor, float]:
    """Encode the melody MIDI into a token prompt for generation.

    Returns (prompt_ids, tempo_bpm).
    """
    events, midi_tempo = midi_to_events(melody_midi, tokenizer.cfg, track_name_override)
    if not events:
        raise ValueError(f"No notes found in {melody_midi}")
    tempo = tempo_override if tempo_override is not None else midi_tempo

    key_root, key_mode = None, None
    kresult = estimate_key_from_midi(melody_midi)
    if kresult is not None:
        key_root, key_mode = kresult
        logger.info(
            f"Key estimated: {_NOTE_NAMES[key_root]} {'major' if key_mode == 0 else 'minor'}"
        )

    ids, _mask = tokenizer.encode_song(
        events, condition_tracks=cond_tracks, target_tracks=[], tempo_bpm=tempo,
        key_root=key_root, key_mode=key_mode,
    )
    if ids and ids[-1] == tokenizer.eos_id:
        ids = ids[:-1]
    return torch.tensor(ids, dtype=torch.long), tempo


def generate_accompaniment(
    melody_midi: Path,
    cfg,
    lit,
    tokenizer: BaseTokenizer,
    cond_tracks: list[str],
    tempo_override: float | None = None,
    track_name_override: dict[str, str] | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    cfg_w: float = 0.0,
    structural_suppression: float | None = None,
) -> tuple:
    """Full generation pipeline: melody MIDI → (output_midi, tempo).

    Returns (miditoolkit.MidiFile, tempo_bpm).
    """
    icfg = cfg.inference
    temperature = temperature if temperature is not None else icfg.temperature
    top_k = top_k if top_k is not None else icfg.top_k
    top_p = top_p if top_p is not None else icfg.top_p
    max_new = max_new_tokens if max_new_tokens is not None else icfg.max_new_tokens
    struct_supp = (structural_suppression if structural_suppression is not None
                   else getattr(icfg, "structural_suppression", 0.0))

    device = next(lit.parameters()).device
    prompt_ids, tempo = build_prompt(
        melody_midi, tokenizer, cond_tracks,
        tempo_override=tempo_override,
        track_name_override=track_name_override,
    )
    prompt_ids = prompt_ids.to(device)
    logger.info(f"Prompt: {prompt_ids.numel()} tokens  |  tempo={tempo:.1f} BPM")

    uncond_ids = None
    if cfg_w > 0.0:
        uncond_ids = torch.tensor(
            tokenizer.make_uncond_prompt(prompt_ids), dtype=torch.long, device=device,
        )

    generated = lit.model.generate(
        prompt_ids,
        max_new_tokens=max_new,
        eos_id=tokenizer.eos_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        uncond_prompt_ids=uncond_ids,
        cfg_w=cfg_w,
        structural_suppression=struct_supp,
        vel_id_range=(tokenizer.vel_min_id, tokenizer.vel_max_id),
        struct_ids=tokenizer.structural_ids(),
    )[0].cpu().tolist()

    all_events = tokenizer.decode(generated)
    target_track_set = set(cfg.tokenizer.tracks) - set(cond_tracks)
    target_events = [e for e in all_events if e.track in target_track_set]
    melody_events, _ = midi_to_events(melody_midi, tokenizer.cfg, track_name_override)

    midi = events_to_midi(
        [*melody_events, *target_events], cfg.tokenizer,
        tempo_bpm=tempo,
        programs=cfg.midi_output.programs,
    )

    hcfg = cfg.humanize
    if hcfg.enabled:
        midi = humanize_midi(midi,
                             velocity_std=hcfg.velocity_std,
                             timing_std_ms=hcfg.timing_std_ms,
                             duration_std_ms=hcfg.duration_std_ms)

    logger.info(f"Generated {len(target_events)} notes across target tracks.")
    return midi, tempo
