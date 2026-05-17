#!/usr/bin/env python3
"""Emit shell-sourceable CFG_* variable assignments from configs/config.yaml.

Used by server shell scripts to read training defaults without hardcoding
them in bash. Values are single-quoted so eval is safe even for strings that
contain hyphens or dots.

Usage (in bash):
    eval "$(python server/read_config.py)"
    echo "$CFG_EPOCHS"          # 80
    echo "$CFG_LR"              # 0.0003

Standalone (debugging):
    python server/read_config.py          # prints assignments + summary table
    VERBOSE=1 python server/read_config.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "config.yaml"

# ---------------------------------------------------------------------------
# (section.key, CFG_variable_name, shell-visible description)
# Add a row here whenever a new training knob should be exposed to scripts.
# ---------------------------------------------------------------------------
_EXPORTS: list[tuple[str, str, str]] = [
    # ── Training ────────────────────────────────────────────────────────────
    ("training.epochs",                  "CFG_EPOCHS",         "training epochs"),
    ("training.batch_size",              "CFG_BATCH_SIZE",     "base batch size (before VRAM scaling)"),
    ("training.learning_rate",           "CFG_LR",             "peak learning rate"),
    ("training.warmup_steps",            "CFG_WARMUP_STEPS",   "LR warmup steps"),
    ("training.grad_clip",               "CFG_GRAD_CLIP",      "gradient clip value"),
    ("training.accumulate_grad_batches", "CFG_ACCUM",          "gradient accumulation steps"),
    ("training.dry_run_steps",           "CFG_DRY_RUN_STEPS",  "dry-run step count (0 = disabled)"),
    ("training.wandb_project",           "CFG_WANDB_PROJECT",  "W&B project name"),
    ("training.wandb_default_run_name",  "CFG_RUN_NAME_BASE",  "base run name (timestamp appended)"),
    ("training.early_stopping_patience", "CFG_ES_PATIENCE",    "early-stopping patience (epochs)"),
    # ── Model ───────────────────────────────────────────────────────────────
    ("model.compile",                    "CFG_COMPILE",        "torch.compile on/off"),
    ("model.d_model",                    "CFG_D_MODEL",        "model width"),
    ("model.n_layers",                   "CFG_N_LAYERS",       "transformer depth"),
    # ── Inference ───────────────────────────────────────────────────────────
    ("inference.temperature",            "CFG_TEMPERATURE",    "sampling temperature"),
    ("inference.structural_suppression", "CFG_STRUCT_SUP",     "structural-suppression strength"),
]


def _get_nested(data: dict, dotted_key: str):
    """Resolve 'a.b.c' into data['a']['b']['c']. Raises KeyError on miss."""
    parts = dotted_key.split(".")
    cur = data
    for p in parts:
        cur = cur[p]
    return cur


def _shell_val(v) -> str:
    """Convert a Python value to a safe, single-quoted shell string."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        # Preserve enough decimal places; avoid 3e-04 scientific notation in
        # shell context where bash arithmetic wouldn't understand it.
        # Python str() handles most cases; special-case very small floats.
        s = f"{v:.10g}"  # up to 10 significant digits, no trailing zeros
        return s
    return str(v)


def main() -> None:
    try:
        import yaml  # PyYAML — installed as part of the project deps
    except ImportError:
        print("# read_config.py: PyYAML not installed; shell scripts will use fallbacks.",
              file=sys.stderr)
        sys.exit(1)

    if not CONFIG.exists():
        print(f"# read_config.py: config not found at {CONFIG}", file=sys.stderr)
        sys.exit(1)

    with CONFIG.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    lines: list[str] = []
    verbose = os.environ.get("VERBOSE", "0") not in ("0", "false", "")

    col_w = max(len(v) for _, v, _ in _EXPORTS) + 2

    for dotted, varname, desc in _EXPORTS:
        try:
            raw = _get_nested(data, dotted)
        except (KeyError, TypeError):
            print(f"# read_config.py: key '{dotted}' not found — skipping {varname}",
                  file=sys.stderr)
            continue
        val = _shell_val(raw)
        lines.append(f"{varname}='{val}'")
        if verbose or sys.stdout.isatty():
            print(f"  {varname:<{col_w}} = {val!s:<14}  # {desc}", file=sys.stderr)

    # Always print the assignments to stdout (for eval)
    if sys.stdout.isatty():
        print("\n# Paste the block below into your shell or eval the whole output:",
              file=sys.stderr)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
