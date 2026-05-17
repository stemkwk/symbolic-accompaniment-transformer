#!/usr/bin/env bash
#
# 00_bringup.sh — one-time setup once the bundle is extracted.
#
# Run this immediately after extracting the bundle and cd-ing into the project:
#   zstd -d jam_tx_bundle.tar.zst | tar x && cd project_transformer   # zst (default)
#   tar xzf jam_tx_bundle.tgz     && cd project_transformer           # gz
# It is idempotent; safe to re-run after a crash or container restart.
#
# Verifies:
#   - bash strict-mode prerequisites
#   - GPU is visible (nvidia-smi)
#   - Project layout is intact
#   - Tokenized data is present and meta files exist
# Installs:
#   - `pip install -e ".[train,dev]"` (skipped if already installed)
#
# Environment variables:
#   SKIP_INSTALL=1   skip pip install (when the host image already has it)
#   PIP_EXTRA=...    extra args passed to pip install (e.g. "--no-deps")

source "$(dirname "$0")/_common.sh"

log_section "Project layout"
require_project_layout
log_step  "PROJECT_ROOT = ${PROJECT_ROOT}"

log_section "GPU"
require_gpu

log_section "Python / torch"
require_cmd python
python --version
python - <<'PY'
import torch
print(f"torch       : {torch.__version__}")
print(f"cuda avail. : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device : {torch.cuda.get_device_name(0)}")
    print(f"vram total  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PY

log_section "Install package"
if [[ "${SKIP_INSTALL:-0}" == "1" ]]; then
    log_warn "SKIP_INSTALL=1 — not running pip install."
else
    log_step "pip install -e .[train,dev] ${PIP_EXTRA:-}"
    python -m pip install -e ".[train,dev]" ${PIP_EXTRA:-}
fi

log_section "Tokenized data"
require_processed_data "data/processed"

# Print the integrity fingerprint so you can spot config drift before training.
python - <<'PY'
import json, pathlib
meta = json.loads(pathlib.Path("data/processed/_dataset_meta.json").read_text(encoding="utf-8"))
print(f"vocab_size            : {meta['vocab_size']}")
print(f"tokenizer_fingerprint : {meta['tokenizer_fingerprint']}")
print(f"n_shards              : {meta['n_shards']}")
print(f"cond_tracks           : {meta['cond_tracks']}")
print(f"target_tracks         : {meta['target_tracks']}")
PY

log_section "Runtime dependencies"
# curl is required by the AUTO_SHUTDOWN monitor in 20_train.sh to call
# the RunPod GraphQL API. Fail here, before training, so you're not stuck
# with a running pod after training finishes and curl is missing.
if command -v curl >/dev/null 2>&1; then
    log_step "curl : $(curl --version | head -1)"
else
    log_warn "curl not found — installing now (needed for AUTO_SHUTDOWN)."
    apt-get install -y curl 2>/dev/null || \
    yum install -y curl 2>/dev/null || \
    log_fail "Could not install curl. Install manually: apt-get install -y curl"
fi

log_section "Network Volume"
# detect_volume aborts (exit 1) when REQUIRE_VOLUME=1 (default) and no
# writable volume is found at /runpod-volume. Override for local testing:
#   REQUIRE_VOLUME=0 ./server/00_bringup.sh
detect_volume

log_section "Smoke test"
log_step "pytest -q -k 'not integration'   (skips the slow end-to-end test)"
python -m pytest -q -k "not integration" || log_fail "smoke tests failed — investigate before training."

log_section "✔ Ready"
echo "Next step: ./server/10_dry_run.sh"
echo ""
if [[ "${VOLUME_MOUNTED}" == "1" ]]; then
    echo "Volume detected: checkpoints will persist after pod stop."
    echo "After training:  SSH_HOST=root@<ip> ./server/90_fetch_artifacts.sh"
else
    echo "!! No Network Volume — attach one in the RunPod console before training."
    echo "   Without it, checkpoints are lost when the pod stops."
fi
