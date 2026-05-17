#!/usr/bin/env bash
#
# 05_inspect_data.sh — sample N tokenized shards back into listenable MIDI
# before kicking off training. Optional step (between 00_bringup and
# 10_dry_run); useful when you uploaded a new tokenization to the server
# and want to confirm the data is sane.
#
# Environment variables:
#   N=4         number of random shards to sample
#   SEED=...    RNG seed (omit for fresh random)
#   NO_AUGMENT=1   skip the augmented variants

source "$(dirname "$0")/_common.sh"
require_project_layout
require_processed_data "data/processed"

ARGS=( "--n" "${N:-4}" )
[[ -n "${SEED:-}" ]] && ARGS+=( "--seed" "${SEED}" )
[[ "${NO_AUGMENT:-0}" == "1" ]] && ARGS+=( "--no_augment" )

log_section "Inspecting data"
python scripts/inspect_data.py "${ARGS[@]}"

log_section "✔ Inspection ready"
echo "Pull the inspection/<timestamp>/ folder to your laptop and listen:"
echo "  SSH_HOST=user@server PATHS=inspection ./server/90_fetch_artifacts.sh"
