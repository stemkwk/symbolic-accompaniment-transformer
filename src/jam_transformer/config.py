"""YAML config loader (single source of truth).

Unknown YAML keys emit a warning and are ignored — to add a new knob, you must
add the matching field on the dataclass first."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List

import yaml

from jam_transformer.logger import logger


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TokenizerConfig:
    # Selects which tokenizer implementation to build (see tokenizer registry).
    name: str = "remi_v1"
    resolution: int = 4
    positions_per_bar: int = 16
    pitch_min: int = 21
    pitch_max: int = 108
    duration_min: int = 1
    duration_max: int = 32
    velocity_bins: int = 32
    tempo_bins: int = 16
    tempo_min: int = 50
    tempo_max: int = 200
    tracks: List[str] = field(default_factory=lambda: ["melody", "bridge", "piano"])
    max_seq_len: int = 2048
    # ------------------------------------------------------------------
    # Relative harmonic encoding settings
    # ------------------------------------------------------------------
    # chord_qualities: number of QUALITY token types.
    #   Vocab layout: SCALE_DEGREE_0..11 (12) + QUALITY_{name} × n + CHORD_N (1)
    #   9  = core set (maj min dom7 maj7 min7 dim aug add9 sus4)
    #   12 = full set (core + dim7 hdim7 dom9, Slakh-tier)
    # Note: no per-root chord tokens; root is in SCALE_DEGREE (key-relative).
    chord_qualities: int = 12
    # Prepend a single KEY_* token (24 total: 12 roots × major/minor) at the
    # start of each sequence. When True, the augmenter also shifts the KEY
    # token along with CHORD roots on pitch-transpose, so the three token
    # families stay harmonically consistent.
    use_key_tokens: bool = True


@dataclass
class ModelConfig:
    name: str = "decoder_transformer_v1"
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    use_rope: bool = True
    tie_weights: bool = True
    # RoPE cos/sin table is pre-allocated up to this length at init time so
    # the buffer address never changes — required for CUDAGraphs compatibility
    # (torch.compile reduce-overhead mode). Must be >= tokenizer.max_seq_len.
    max_seq_len: int = 4096
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    gradient_checkpointing: bool = False


@dataclass
class AugmentConfig:
    pitch_transpose_semitones: int = 6
    velocity_jitter_bins: int = 2
    # Random tempo shift: ±N bins applied to every TEMPO_* token in the sequence.
    # Simulates the same piece played at different speeds without re-tokenizing.
    tempo_jitter_bins: int = 2
    # Random duration stretch: ±N bins applied to each DUR_* token independently.
    # Teaches the model not to fixate on exact note lengths.
    duration_jitter_bins: int = 1
    condition_dropout_prob: float = 0.15


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    epochs: int = 200
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    accumulate_grad_batches: int = 1
    warmup_steps: int = 500
    min_lr_factor: float = 0.1
    optimizer: str = "adamw_fused"
    scheduler: str = "cosine_warmup"
    # Validation batch size (no backward → no need to store activations).
    # 0 = auto (2× training batch_size). Reduces val epoch time by ~half.
    val_batch_size: int = 0
    mask_condition_loss: bool = True
    # ------------------------------------------------------------------
    # Token-type-aware loss weighting
    # ------------------------------------------------------------------
    loss_struct_weight: float = 0.3       # BAR / POS / TRACK / TEMPO targets
    loss_content_weight: float = 1.5      # CHROMA / OCTAVE / DUR / VEL targets
    # ------------------------------------------------------------------
    # Polyphony loss boost
    # ------------------------------------------------------------------
    polyphony_loss_boost: float = 2.0
    # ------------------------------------------------------------------
    # Polyphony-weighted chunk sampling
    # ------------------------------------------------------------------
    polyphony_sample_weight_alpha: float = 0.5
    log_every_n_steps: int = 25
    checkpoint_dir: str = "checkpoints"
    checkpoint_monitor: str = "val_loss"
    checkpoint_monitor_mode: str = "min"
    checkpoint_every_n_epochs: int = 5
    checkpoint_every_n_train_steps: int = 100
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    early_stopping_min_epochs: int = 15
    log_to_file: bool = True
    log_dir: str = "logs"
    csv_logger_enabled: bool = True
    wandb_project: str = "jam-transformer"
    wandb_default_run_name: str = "pop909-baseline"
    dry_run_steps: int = 0


@dataclass
class InferenceConfig:
    temperature: float = 1.1
    top_k: int = 0          # 0 = disabled (top_p only)
    top_p: float = 0.92
    max_new_tokens: int = 2048
    render_audio: bool = False
    soundfont: str = ""
    sample_rate: int = 22050
    structural_suppression: float = 1.5


@dataclass
class HumanizeConfig:
    enabled: bool = True
    velocity_std: int = 6         # gaussian std for velocity jitter (±units)
    timing_std_ms: float = 8.0    # gaussian std for note-on timing jitter (ms)
    duration_std_ms: float = 5.0  # gaussian std for note duration jitter (ms)


@dataclass
class DspConfig:
    enabled: bool = True
    # Reverb — adds room/space feel (biggest improvement for dry MIDI audio)
    reverb: bool = True
    reverb_room_size: float = 0.25   # 0–1: larger = bigger room
    reverb_damping: float = 0.5      # 0–1: higher = more high-freq absorption
    reverb_wet_level: float = 0.18   # reverb send level
    reverb_dry_level: float = 0.82   # dry signal level
    # Compressor — glues mix, evens out velocity dynamics
    compressor: bool = True
    compressor_threshold_db: float = -18.0
    compressor_ratio: float = 4.0
    compressor_attack_ms: float = 5.0
    compressor_release_ms: float = 100.0
    # Limiter — prevents clipping on final output
    limiter: bool = True
    limiter_threshold_db: float = -1.0


@dataclass
class AudioInputConfig:
    # Noise reduction (noisereduce spectral gating)
    denoise: bool = False
    # basic-pitch transcription parameters
    onset_threshold: float = 0.5    # higher = fewer false positives
    frame_threshold: float = 0.3    # higher = shorter / fewer notes
    min_note_length_ms: float = 58.0
    min_frequency: float = 32.7     # Hz — C1; set higher for vocal/instrument range
    max_frequency: float = 2093.0   # Hz — C7


@dataclass
class MidiOutputConfig:
    # GM program number (0–127) per logical track name.
    # Common values: 0=acoustic grand, 25=steel guitar, 40=violin, 48=strings,
    #   56=trumpet, 65=alto sax, 73=flute, 0=piano.
    programs: Dict[str, int] = field(default_factory=lambda: {
        "melody": 40,   # violin
        "bridge": 25,   # acoustic guitar (steel)
        "piano":  0,    # acoustic grand
    })


@dataclass
class VRAMTier:
    vram_gte_gb: float
    batch_scale: int
    label: str = ""


@dataclass
class EnvScalingConfig:
    fallback_batch_scale: int = 1
    default_precision: str = "bf16-mixed"
    cpu_precision: str = "32"
    tiers: List[VRAMTier] = field(default_factory=list)
    num_workers_colab: int = 2
    num_workers_windows: int = 0
    num_workers_server_max: int = 8
    prefetch_factor: int = 2


@dataclass
class AppConfig:
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    env_scaling: EnvScalingConfig
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    humanize: HumanizeConfig = field(default_factory=HumanizeConfig)
    audio_input: AudioInputConfig = field(default_factory=AudioInputConfig)
    midi_output: MidiOutputConfig = field(default_factory=MidiOutputConfig)
    dsp: DspConfig = field(default_factory=DspConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _populate(dc_cls, raw: Dict[str, Any], section: str):
    """Instantiate a dataclass from `raw`, warning on unknown keys."""
    if raw is None:
        raw = {}
    known = {f.name for f in fields(dc_cls)}
    extra = set(raw.keys()) - known
    for k in extra:
        logger.warning(f"Unknown key '{section}.{k}' in YAML — ignored.")
    return dc_cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    tier_dicts = (raw.get("env_scaling") or {}).get("tiers") or []
    tiers = [VRAMTier(**t) for t in tier_dicts]
    env_raw = {**(raw.get("env_scaling") or {}), "tiers": tiers}

    midi_out_raw = raw.get("midi_output") or {}
    midi_output = MidiOutputConfig(
        programs={**MidiOutputConfig().programs, **(midi_out_raw.get("programs") or {})}
    )

    return AppConfig(
        tokenizer=_populate(TokenizerConfig, raw.get("tokenizer"), "tokenizer"),
        model=_populate(ModelConfig, raw.get("model"), "model"),
        training=_populate(TrainingConfig, raw.get("training"), "training"),
        inference=_populate(InferenceConfig, raw.get("inference"), "inference"),
        env_scaling=_populate(EnvScalingConfig, env_raw, "env_scaling"),
        augment=_populate(AugmentConfig, raw.get("augment"), "augment"),
        humanize=_populate(HumanizeConfig, raw.get("humanize"), "humanize"),
        audio_input=_populate(AudioInputConfig, raw.get("audio_input"), "audio_input"),
        midi_output=midi_output,
        dsp=_populate(DspConfig, raw.get("dsp"), "dsp"),
    )
