"""Sample accompaniment from a melody-only MIDI.

Pipeline
--------
1. Read the input MIDI (or transcribe from audio/mic), keep only the melody track.
2. Encode the melody as a condition prefix (ends with <SEP>).
3. Sample the rest of the sequence with the trained model.
4. Decode token ids → NoteEvents → output MIDI.
5. Optionally render the MIDI to WAV via FluidSynth (requires soundfont).
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch

from jam_transformer.audio import apply_dsp, audio_to_midi, record_from_mic, render_midi_to_wav
from jam_transformer.config import load_config
from jam_transformer.logger import logger
from jam_transformer.overrides import apply_overrides
from jam_transformer.pipeline import generate_accompaniment, load_checkpoint
from jam_transformer.tokenizer import build_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample accompaniment from a melody MIDI.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--melody_midi", type=str, default=None,
                        help="Input melody MIDI file. Required unless --audio_input or --mic_input is given.")
    parser.add_argument(
        "--audio_input", type=str, default=None,
        metavar="AUDIO_FILE",
        help="Audio file (WAV, MP3, FLAC, …) to transcribe to MIDI before inference. "
             "Requires basic-pitch: pip install 'jam_transformer[audio]'. "
             "When given, --melody_midi is optional (defaults to <output>.transcribed.mid).",
    )
    parser.add_argument(
        "--mic_input", action="store_true",
        help="Record melody from the default microphone. "
             "Recording starts immediately and stops when you press Enter. "
             "Requires sounddevice + scipy: pip install 'jam_transformer[audio]'. "
             "Recorded WAV is saved next to --output as <output>.recorded.wav.",
    )
    parser.add_argument(
        "--denoise", action="store_true",
        help="Apply spectral noise reduction (noisereduce) before MIDI transcription. "
             "Recommended with --mic_input in noisy environments. "
             "Requires noisereduce: pip install 'jam_transformer[audio]'.",
    )
    parser.add_argument("--output", type=str, default="output/accompaniment.mid")
    parser.add_argument("--cond_tracks", type=str, default="melody")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--tempo", type=float, default=None,
        help="Override BPM (e.g. 120). By default uses the BPM from the input MIDI. "
             "Changes both the TEMPO conditioning token (model input) and the "
             "output MIDI/WAV playback speed. Valid range: 50–200 BPM.",
    )
    parser.add_argument(
        "--struct_suppression", type=float, default=None,
        help="Subtract this from BAR/POS logits right after a (PITCH,DUR,VEL) "
             "triple to bias the sampler toward stacking another note at the "
             "same position (= polyphony hack). Default: inference.structural_suppression.",
    )
    parser.add_argument(
        "--cfg_w", type=float, default=0.0,
        help="Classifier-Free Guidance weight (0 = off, typical: 1.5–3.0). "
             "When > 0 the model's unconditional branch (PAD'd condition) is "
             "run alongside the conditional branch and logits are blended: "
             "logits = logits_uncond + cfg_w * (logits_cond - logits_uncond). "
             "Requires the model to have been trained with condition_dropout_prob > 0.",
    )
    parser.add_argument(
        "--track_map", type=str, default=None,
        metavar="MIDI_NAME=TRACK,...",
        help="Map MIDI track names to logical track names. "
             "Example: --track_map \"Piano Roll=melody,Strings=bridge\" "
             "(case-insensitive MIDI name match). "
             "Overrides the built-in POP909 name table (MELODY, BRIDGE, PIANO).",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE",
                        help="Override any nested config field. Repeatable. "
                             "Must match the values used during training "
                             "(model architecture especially).")
    args = parser.parse_args()

    if args.mic_input and args.audio_input:
        parser.error("--mic_input and --audio_input are mutually exclusive.")
    if not args.mic_input and not args.audio_input and args.melody_midi is None:
        parser.error("One of --melody_midi / --audio_input / --mic_input is required.")

    cfg = load_config(args.config)
    if args.overrides:
        apply_overrides(cfg, args.overrides)
    icfg = cfg.inference
    acfg = cfg.audio_input

    # ── Audio / mic input: resolve melody MIDI path ───────────────────────────
    # --denoise flag overrides audio_input.denoise from config
    denoise = args.denoise or acfg.denoise
    out_path = Path(args.output)

    if args.mic_input:
        recorded_wav = out_path.with_suffix("").with_suffix(".recorded.wav")
        record_from_mic(recorded_wav)
        transcribed_midi = (
            Path(args.melody_midi) if args.melody_midi
            else out_path.with_suffix("").with_suffix(".transcribed.mid")
        )
        audio_to_midi(recorded_wav, transcribed_midi,
                      denoise=denoise,
                      onset_threshold=acfg.onset_threshold,
                      frame_threshold=acfg.frame_threshold,
                      min_note_length_ms=acfg.min_note_length_ms,
                      min_frequency=acfg.min_frequency,
                      max_frequency=acfg.max_frequency)
        args.melody_midi = str(transcribed_midi)
    elif args.audio_input:
        transcribed_midi = (
            Path(args.melody_midi) if args.melody_midi
            else out_path.with_suffix("").with_suffix(".transcribed.mid")
        )
        audio_to_midi(Path(args.audio_input), transcribed_midi,
                      denoise=denoise,
                      onset_threshold=acfg.onset_threshold,
                      frame_threshold=acfg.frame_threshold,
                      min_note_length_ms=acfg.min_note_length_ms,
                      min_frequency=acfg.min_frequency,
                      max_frequency=acfg.max_frequency)
        args.melody_midi = str(transcribed_midi)
    temperature = args.temperature if args.temperature is not None else icfg.temperature
    top_k = args.top_k if args.top_k is not None else icfg.top_k
    top_p = args.top_p if args.top_p is not None else icfg.top_p
    max_new = args.max_new_tokens if args.max_new_tokens is not None else icfg.max_new_tokens

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = build_tokenizer(cfg.tokenizer)
    cond_tracks = [t.strip() for t in args.cond_tracks.split(",") if t.strip()]

    track_name_override: dict[str, str] | None = None
    if args.track_map:
        track_name_override = {}
        for pair in args.track_map.split(","):
            if "=" not in pair:
                raise ValueError(f"--track_map entry must be 'MIDI_NAME=TRACK', got: {pair!r}")
            midi_name, logical = pair.split("=", 1)
            track_name_override[midi_name.strip().upper()] = logical.strip()
        logger.info(f"Track name override: {track_name_override}")

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    lit = load_checkpoint(args.checkpoint, cfg, tokenizer.vocab_size)
    lit.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit.to(device)

    midi, tempo = generate_accompaniment(
        melody_midi=Path(args.melody_midi),
        cfg=cfg,
        lit=lit,
        tokenizer=tokenizer,
        cond_tracks=cond_tracks,
        tempo_override=args.tempo,
        track_name_override=track_name_override,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        cfg_w=args.cfg_w,
        structural_suppression=args.struct_suppression,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(out_path))
    logger.info(f"Wrote {out_path}")

    if icfg.render_audio:
        wav_path = out_path.with_suffix(".wav")
        render_midi_to_wav(out_path, wav_path, icfg.soundfont, icfg.sample_rate)
        if wav_path.exists():
            try:
                apply_dsp(wav_path, wav_path, cfg.dsp)
            except ImportError:
                logger.warning("pedalboard not installed — skipping DSP effects.")


if __name__ == "__main__":
    main()
