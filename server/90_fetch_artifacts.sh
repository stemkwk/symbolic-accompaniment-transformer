#!/usr/bin/env bash
#
# 90_fetch_artifacts.sh — pull training artifacts back to your local machine
# AFTER a training run is done (or anytime during, to grab partial results).
#
# **Run this from your LOCAL machine** (Windows: WSL or Git Bash), not the server.
#
# ── SSH settings ──────────────────────────────────────────────────────────────
#   Compression=no        checkpoints are float32/bf16 tensors — ssh compression
#                         would burn CPU with near-zero gain. Skip it.
#   aes128-gcm@openssh    hardware AES-NI: encrypt + MAC in one pass.
#                         Fastest safe cipher available on all current RunPod images.
#   --no-compress         rsync-level compression also skipped (same reason).
#   --partial             resume interrupted transfer instead of restarting.
#   -P                    show per-file progress and keep partial files.
#
# ── Required environment ──────────────────────────────────────────────────────
#   SSH_HOST=user@1.2.3.4[:port]     SSH target. Port suffix is optional.
#
# ── Optional environment ──────────────────────────────────────────────────────
#   REMOTE_DIR=~/project_transformer  remote project root
#   LOCAL_DIR=./pulled                local destination
#   PATHS="checkpoints logs output"   whitespace-separated subdirs to fetch
#
# Examples:
#   SSH_HOST=root@1.2.3.4 ./server/90_fetch_artifacts.sh
#   SSH_HOST=root@1.2.3.4:2222 PATHS="checkpoints" ./server/90_fetch_artifacts.sh
#   SSH_HOST=root@1.2.3.4 LOCAL_DIR=./run42 ./server/90_fetch_artifacts.sh
#
# Re-running is safe: rsync only downloads bytes that differ (delta transfer).

set -euo pipefail

: "${SSH_HOST:?Set SSH_HOST=user@host[:port]}"
LOCAL_DIR="${LOCAL_DIR:-./pulled}"
PATHS="${PATHS:-checkpoints logs output sweep_results}"

# ── Parse optional :port suffix (must happen first — probe uses these) ────────
SSH_PORT=""
HOST="${SSH_HOST}"
if [[ "${SSH_HOST}" == *:* ]]; then
    SSH_PORT="${SSH_HOST##*:}"
    HOST="${SSH_HOST%:*}"
fi

# ── Remote path resolution ────────────────────────────────────────────────────
# Training saves checkpoints to Network Volume (/runpod-volume/checkpoints)
# when a volume is attached, or to ~/project_transformer/checkpoints otherwise.
# We auto-detect which was used so the user doesn't need to know.
#
# Override: REMOTE_DIR=/some/explicit/path  to skip detection entirely.
if [[ -z "${REMOTE_DIR:-}" ]]; then
    _VOLUME_ROOT="/runpod-volume"
    _WORKSPACE="/workspace"
    _LOCAL_PROJ="~/project_transformer"

    # Quick SSH probe: check candidate storage roots in priority order.
    # Must mirror detect_volume() in _common.sh:
    #   1. /runpod-volume  — RunPod Network Volume
    #   2. /workspace      — Vast.ai persistent workspace
    #   3. ~/project_transformer — local project dir (bare-metal / no volume)
    _RSH_PROBE="ssh -o Compression=no -o ConnectTimeout=5 -c aes128-gcm@openssh.com"
    [[ -n "${SSH_PORT}" ]] && _RSH_PROBE="${_RSH_PROBE} -p ${SSH_PORT}"

    if ${_RSH_PROBE} "${HOST}" \
        "ls ${_VOLUME_ROOT}/checkpoints/*.ckpt 2>/dev/null | head -1" 2>/dev/null \
        | grep -q '\.ckpt'; then
        REMOTE_DIR="${_VOLUME_ROOT}"
        echo "→ RunPod Network Volume — remote root: ${REMOTE_DIR}"
        echo "  checkpoints: ${REMOTE_DIR}/checkpoints/"
        echo "  logs:        ${REMOTE_DIR}/logs/"
    elif ${_RSH_PROBE} "${HOST}" \
        "ls ${_WORKSPACE}/checkpoints/*.ckpt 2>/dev/null | head -1" 2>/dev/null \
        | grep -q '\.ckpt'; then
        REMOTE_DIR="${_WORKSPACE}"
        echo "→ Vast.ai /workspace — remote root: ${REMOTE_DIR}"
        echo "  checkpoints: ${REMOTE_DIR}/checkpoints/"
        echo "  logs:        ${REMOTE_DIR}/logs/"
    else
        REMOTE_DIR="${_LOCAL_PROJ}"
        echo "→ No volume checkpoints found — remote root: ${REMOTE_DIR}"
    fi
fi

# ── Build SSH and rsync option strings ───────────────────────────────────────
# aes128-gcm: single-pass authenticated encryption — fastest safe cipher.
# Compression=no: checkpoints are binary float tensors, re-compression wastes CPU.
_SSH_CIPHER="aes128-gcm@openssh.com"
_RSH="ssh -o Compression=no -c ${_SSH_CIPHER}"
[[ -n "${SSH_PORT}" ]] && _RSH="${_RSH} -p ${SSH_PORT}"

echo "→ source : ${HOST}:${REMOTE_DIR}"
echo "→ dest   : ${LOCAL_DIR}"
echo "→ paths  : ${PATHS}"
echo "→ cipher : ${_SSH_CIPHER}  (hardware AES)"
echo ""

mkdir -p "${LOCAL_DIR}"

if command -v rsync >/dev/null 2>&1; then
    for p in ${PATHS}; do
        echo "── rsync ${p}/ ──────────────────────────────────────────"
        rsync \
            -avhP \
            --partial \
            --no-compress \
            -e "${_RSH}" \
            "${HOST}:${REMOTE_DIR}/${p}/" \
            "${LOCAL_DIR}/${p}/" \
        || echo "  (skipped — ${p}/ does not exist on remote yet)"
    done
else
    # scp fallback (no delta transfer, no partial resume)
    echo "rsync not found — falling back to scp (slower: no delta transfer or resume)."
    _SCP_OPTS="-o Compression=no -c ${_SSH_CIPHER}"
    [[ -n "${SSH_PORT}" ]] && _SCP_OPTS="${_SCP_OPTS} -P ${SSH_PORT}"
    for p in ${PATHS}; do
        echo "── scp ${p} ────────────────────────────────────────────"
        scp ${_SCP_OPTS} -r \
            "${HOST}:${REMOTE_DIR}/${p}" \
            "${LOCAL_DIR}/" \
        || echo "  (skipped — ${p} does not exist on remote yet)"
    done
fi

echo ""
echo "✔ artifacts saved to ${LOCAL_DIR}/"
echo ""
echo "Quick checkpoint stats:"
if ls "${LOCAL_DIR}/checkpoints/"*.ckpt 2>/dev/null | head -5; then
    du -sh "${LOCAL_DIR}/checkpoints/"*.ckpt 2>/dev/null | sort -h || true
fi
