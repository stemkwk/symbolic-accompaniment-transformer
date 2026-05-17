#!/usr/bin/env bash
#
# 21_resume.sh — resume training from the best available checkpoint.
#
# Searches in priority order:
#   1. Network Volume (/runpod-volume/checkpoints) — auto-detected
#   2. Local project checkpoints/
#
# Within each location:
#   a. last.ckpt       — clean epoch boundary (preferred)
#   b. last_step.ckpt  — mid-epoch crash guard (fallback)
#
# All EPOCHS, BATCH_SIZE, LR, COMPILE, RUN_NAME, AUTO_SHUTDOWN, EXTRA env
# vars are forwarded to 20_train.sh unchanged.

source "$(dirname "$0")/_common.sh"

# detect_volume sets CKPT_DIR to /runpod-volume/checkpoints (if volume is
# attached) or ${PROJECT_ROOT}/checkpoints (local fallback).  This MUST match
# the path used during the original training run.
detect_volume   # sets CKPT_DIR / TRAIN_LOG_DIR / VOLUME_MOUNTED

log_section "Looking for checkpoint"
log_step "search path: ${CKPT_DIR}"

if [[ -f "${CKPT_DIR}/last.ckpt" ]]; then
    export RESUME="${CKPT_DIR}/last.ckpt"
    log_step "Resuming from epoch checkpoint: ${RESUME}"
elif [[ -f "${CKPT_DIR}/last_step.ckpt" ]]; then
    export RESUME="${CKPT_DIR}/last_step.ckpt"
    log_warn "last.ckpt not found — falling back to mid-epoch step checkpoint: ${RESUME}"
    log_warn "This checkpoint has no val_loss. Training will re-run the current epoch from this step."
else
    log_fail "No checkpoint found in ${CKPT_DIR}. Start fresh with 20_train.sh."
fi

exec "$(dirname "$0")/20_train.sh" "$@"
