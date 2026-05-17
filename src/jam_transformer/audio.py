"""Audio I/O utilities for jam_transformer.

Four entry points:
  record_from_mic()    — record WAV from the default microphone (Enter to stop)
  denoise_audio()      — spectral noise reduction via noisereduce
  audio_to_midi()      — transcribe WAV/MP3/FLAC/… to MIDI via basic-pitch
  render_midi_to_wav() — render a MIDI file to WAV via FluidSynth
"""
from __future__ import annotations

from pathlib import Path

from jam_transformer.logger import logger


def record_from_mic(out_wav: Path, sample_rate: int = 44100) -> None:
    """Record from the default microphone until the user presses Enter.

    Saves a 16-bit mono WAV to *out_wav*.
    Requires: pip install 'jam_transformer[audio]'  (sounddevice, scipy)
    """
    try:
        import sounddevice as sd
        import numpy as np
        from scipy.io import wavfile
    except ImportError:
        raise ImportError(
            "sounddevice and scipy are required for microphone recording. "
            "Run: pip install 'jam_transformer[audio]'"
        )

    chunks: list = []

    def _callback(indata, frames, time, status):  # noqa: ARG001
        chunks.append(indata.copy())

    print("\n[REC] 마이크 녹음 시작 — 멜로디를 연주하세요.")
    print("[REC] 끝나면 Enter를 누르세요.\n")
    with sd.InputStream(samplerate=sample_rate, channels=1,
                        dtype="float32", callback=_callback):
        input()

    audio = np.concatenate(chunks, axis=0)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_wav), sample_rate, (audio * 32767).astype("int16"))
    duration = len(audio) / sample_rate
    logger.info(f"Recorded {duration:.1f}s  →  {out_wav}")


def denoise_audio(audio_path: Path, out_wav: Path) -> None:
    """Reduce stationary background noise using spectral gating (noisereduce).

    Estimates the noise profile from the first 0.5 s of the recording and
    subtracts it across the whole file — effective for room hum, fan noise,
    and mic self-noise.
    Requires: pip install 'jam_transformer[audio]'  (noisereduce, scipy)
    """
    try:
        import noisereduce as nr
        import numpy as np
        from scipy.io import wavfile
    except ImportError:
        raise ImportError(
            "noisereduce and scipy are required for denoising. "
            "Run: pip install 'jam_transformer[audio]'"
        )

    sr, data = wavfile.read(str(audio_path))
    audio = data.astype(np.float32) / 32768.0
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    reduced = nr.reduce_noise(y=audio, sr=sr)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_wav), sr, (reduced * 32767).astype("int16"))
    logger.info(f"Denoised audio → {out_wav}")


def audio_to_midi(
    audio_path: Path,
    out_midi: Path,
    denoise: bool = False,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length_ms: float = 58.0,
    min_frequency: float | None = 32.7,
    max_frequency: float | None = 2093.0,
) -> None:
    """Transcribe an audio file (WAV/MP3/FLAC/…) to MIDI using basic-pitch.

    basic-pitch is a polyphonic neural AMT model; weights ship with the package
    so no separate download is needed.

    All transcription parameters map 1-to-1 to basic-pitch's predict() args.
    Requires: pip install 'jam_transformer[audio]'
    """
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError:
        raise ImportError(
            "basic-pitch is not installed. "
            "Run: pip install 'jam_transformer[audio]'  (or: pip install basic-pitch)"
        )

    src = audio_path
    if denoise:
        import tempfile
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        denoise_audio(audio_path, tmp)
        src = tmp

    logger.info(f"Transcribing audio → MIDI: {src}")
    _, midi_data, _ = predict(
        str(src),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length_ms,
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
    )
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    midi_data.write(str(out_midi))
    logger.info(f"Transcribed MIDI saved: {out_midi}")

    if denoise and src != audio_path:
        src.unlink(missing_ok=True)


_SOUNDFONT_SEARCH_PATHS = [
    # Linux (apt install fluid-soundfont-gm / musescore-soundfont-gm)
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GS.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
    # macOS (Homebrew fluid-synth)
    "/usr/local/share/sounds/sf2/FluidR3_GM.sf2",
    "/opt/homebrew/share/sounds/sf2/FluidR3_GM.sf2",
    # Common user locations
    "~/soundfonts/FluidR3_GM.sf2",
    "~/soundfonts/GeneralUser.sf2",
    "~/soundfonts/default.sf2",
]


def _find_soundfont(configured: str | None) -> str | None:
    """Return the first usable soundfont path, or None with a helpful message."""
    if configured and Path(configured).exists():
        return configured

    for candidate in _SOUNDFONT_SEARCH_PATHS:
        p = Path(candidate).expanduser()
        if p.exists():
            logger.info(f"Auto-detected soundfont: {p}")
            return str(p)

    logger.warning(
        "No soundfont found — WAV render skipped.\n"
        "  Install one of:\n"
        "    Linux:  sudo apt install fluid-soundfont-gm\n"
        "    macOS:  brew install fluid-synth  (includes FluidR3_GM)\n"
        "    Manual: download GeneralUser GS (.sf2) and set inference.soundfont in config.yaml"
    )
    return None


def render_midi_to_wav(midi_path: Path, wav_path: Path,
                       soundfont: str, sample_rate: int) -> None:
    """Render a MIDI file to WAV using FluidSynth.

    Requires: pip install 'jam_transformer[render]'  (pyfluidsynth, scipy)
    *soundfont* is tried first; if missing, common system paths are searched.
    """
    try:
        import fluidsynth
        from scipy.io import wavfile
        import numpy as np
    except ImportError:
        logger.warning("pyfluidsynth + scipy not installed; skipping WAV render.")
        return

    sf = _find_soundfont(soundfont)
    if sf is None:
        return

    fs = fluidsynth.Synth(samplerate=float(sample_rate))
    sfid = fs.sfload(sf)
    fs.program_select(0, sfid, 0, 0)

    if hasattr(fs, "midi_to_audio"):
        fs.midi_to_audio(str(midi_path), str(wav_path))
    else:
        import miditoolkit
        midi = miditoolkit.MidiFile(str(midi_path))
        samples = []
        for inst in midi.instruments:
            for n in inst.notes:
                fs.noteon(0, n.pitch, n.velocity)
                samples.append(
                    fs.get_samples(int(sample_rate * (n.end - n.start) / midi.ticks_per_beat))
                )
                fs.noteoff(0, n.pitch)
        audio = np.concatenate(samples) if samples else np.zeros(sample_rate, dtype=np.int16)
        wavfile.write(str(wav_path), sample_rate, audio.astype("int16"))

    fs.delete()
    logger.info(f"Rendered WAV → {wav_path}")
