"""Environment / hardware auto-tuning. Mirrors the structure of the spectrogram
project so the same YAML pattern works."""
from __future__ import annotations

import os
from typing import Any, Dict

import torch

from jam_transformer.config import EnvScalingConfig


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _env_label() -> str:
    if _is_colab():
        return "Google Colab"
    if os.name == "nt":
        return "Windows Local"
    return "Linux Server"


def _num_workers(env_cfg: EnvScalingConfig) -> int:
    if os.name == "nt":
        return env_cfg.num_workers_windows
    if _is_colab():
        return env_cfg.num_workers_colab
    cpus = os.cpu_count() or 1
    return max(1, min(cpus - 1, env_cfg.num_workers_server_max))


def _tier_scale(env_cfg: EnvScalingConfig, vram_gb: float) -> int:
    for t in env_cfg.tiers:
        if vram_gb >= t.vram_gte_gb:
            return t.batch_scale
    return env_cfg.fallback_batch_scale


def get_env_optimizations(
    base_batch_size: int,
    base_lr: float,
    env_cfg: EnvScalingConfig,
) -> Dict[str, Any]:
    opts: Dict[str, Any] = {
        "batch_size": base_batch_size,
        "learning_rate": base_lr,
        "num_workers": _num_workers(env_cfg),
        "persistent_workers": False,
        "prefetch_factor": None,
        "precision": env_cfg.default_precision,
        "env_name": _env_label(),
    }
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        scale = _tier_scale(env_cfg, vram_gb)
        opts["batch_size"] = base_batch_size * scale
        opts["learning_rate"] = base_lr * scale
        # Warmup steps cover the same amount of *data* regardless of batch size.
        # Larger batch → fewer steps per epoch → divide warmup proportionally so
        # the warmup phase doesn't consume a disproportionate fraction of training.
        opts["batch_scale"] = scale
    else:
        opts["precision"] = env_cfg.cpu_precision
        opts["batch_scale"] = 1

    if opts["num_workers"] > 0:
        opts["persistent_workers"] = True
        opts["prefetch_factor"] = env_cfg.prefetch_factor
    if "batch_scale" not in opts:
        opts["batch_scale"] = 1
    return opts
