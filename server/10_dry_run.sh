#!/usr/bin/env bash
#
# 10_dry_run.sh — measure ms/step and VRAM, print a budget estimate. Always
# run this BEFORE committing to a long training run on a paid GPU.
#
# Reports:
#   - throughput (ms/step)
#   - estimated epoch length and full-run cost
#   - peak VRAM
# Does NOT touch checkpoints — it exits before Lightning's Trainer starts.
#
# ── Overridable variables (ENV > config.yaml > built-in default) ─────────────
#
#   DRY_RUN_STEPS=100   steps to time. Alias: STEPS (backward compat).
#                       If not set and config.dry_run_steps is 0, defaults to 100.
#   BATCH_SIZE=...      base batch size (VRAM-tier scaling still applies).
#   COMPILE=true|false  toggle torch.compile.
#   EXTRA="..."         extra arguments forwarded verbatim to scripts/train.py
#                       (e.g. EXTRA="--set model.n_layers=6").
#
# Examples:
#   ./server/10_dry_run.sh
#   DRY_RUN_STEPS=200 BATCH_SIZE=32 ./server/10_dry_run.sh
#   COMPILE=true ./server/10_dry_run.sh

source "$(dirname "$0")/_common.sh"
require_project_layout
require_processed_data

load_config_defaults

# ── Resolve variables: ENV > config.yaml > script default ───────────────────
_dflt_steps="${CFG_DRY_RUN_STEPS:-0}"
[[ "${_dflt_steps}" == "0" ]] && _dflt_steps=100   # config disables dry-run by default; use 100 here
DRY_RUN_STEPS="${DRY_RUN_STEPS:-${STEPS:-${_dflt_steps}}}"

BATCH_SIZE="${BATCH_SIZE:-${CFG_BATCH_SIZE}}"
COMPILE="${COMPILE:-${CFG_COMPILE}}"

log_section "Dry run"
log_step "steps      : ${DRY_RUN_STEPS}"
log_step "batch_size : ${BATCH_SIZE}  (before VRAM-tier scaling)"
log_step "compile    : ${COMPILE}"
[[ -n "${EXTRA:-}" ]] && log_step "extra      : ${EXTRA}"

ARGS=(
    "--dry_run_steps" "${DRY_RUN_STEPS}"
    "--batch_size"    "${BATCH_SIZE}"
    "--set"           "model.compile=${COMPILE}"
    "--set"           "training.log_to_file=false"
)
if [[ -n "${EXTRA:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${EXTRA} )
    ARGS+=( "${EXTRA_ARGS[@]}" )
fi

WANDB_DISABLED="${WANDB_DISABLED:-true}" python scripts/train.py "${ARGS[@]}"

log_section "✔ Dry run complete"
echo "Multiply 'est N ep' by your GPU's hourly rate to get the run cost."
echo "If you're happy, kick off the real training:   ./server/20_train.sh"
