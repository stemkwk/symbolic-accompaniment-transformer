#!/usr/bin/env bash
#
# upload_bundle.sh — transfer the local bundle to a GPU server.
#
# Run this from your LOCAL machine (Windows: WSL or Git Bash).
#
# ── Two upload modes ─────────────────────────────────────────────────────────
#
#   [기본] Bundle mode — 초기 업로드 또는 data/processed 변경 시
#     python scripts/package_for_server.py   # .tar.zst 생성 (~수십초)
#     SSH_HOST=root@<ip>:<port> ./server/upload_bundle.sh
#
#   [빠름] RESYNC=1 — 코드/config만 수정한 경우 (data 재전송 없음)
#     RESYNC=1 SSH_HOST=root@<ip>:<port> ./server/upload_bundle.sh
#     → data/processed (~GB) 제외, 변경된 파일만 delta 전송. 수 초~수십 초.
#     → src/ 변경 시 서버에서 pip 재설치 필요 (스크립트가 명령 출력해줌).
#
# ── Destination auto-detection ───────────────────────────────────────────────
#   1. If /runpod-volume is writable on the remote → upload there (RunPod Volume).
#   2. If /workspace is writable → upload there (Vast.ai).
#   3. Otherwise → fall back to ~/ (container local disk).
#   Override: set REMOTE_DIR explicitly to skip the auto-detection probe.
#
# ── Why aes128-gcm ───────────────────────────────────────────────────────────
#   AES-NI hardware: encrypt + MAC in one pass. Fastest safe SSH cipher.
#   Compression=no: bundle is already compressed — SSH re-compression is slower.
#
# ── Required environment variable ────────────────────────────────────────────
#   SSH_HOST=user@1.2.3.4[:port]
#
# ── Optional environment variables ───────────────────────────────────────────
#   RESYNC=1           코드만 delta 재전송 (data 제외). 가장 빠른 재배포 방법.
#   BUNDLE=...         번들 경로 지정 (기본: 자동 탐색)
#   REMOTE_DIR=...     원격 경로 지정 (기본: 자동 탐색)
#   SKIP_VERIFY=1      SHA-256 검증 생략
#
# Examples:
#   SSH_HOST=root@1.2.3.4 ./server/upload_bundle.sh                 # 초기 업로드
#   RESYNC=1 SSH_HOST=root@1.2.3.4:2222 ./server/upload_bundle.sh   # 코드 재배포
#   SSH_HOST=root@1.2.3.4 SKIP_VERIFY=1 ./server/upload_bundle.sh

set -euo pipefail

: "${SSH_HOST:?Set SSH_HOST=user@host[:port]}"

SKIP_VERIFY="${SKIP_VERIFY:-0}"
RESYNC="${RESYNC:-0}"

# ── Parse optional :port suffix ───────────────────────────────────────────────
SSH_PORT=""
_HOST="${SSH_HOST}"
if [[ "${SSH_HOST}" == *:* ]]; then
    SSH_PORT="${SSH_HOST##*:}"
    _HOST="${SSH_HOST%:*}"
fi

# -o Compression=no  — 이미 압축된 번들을 SSH가 재압축하지 않도록 (느려짐)
# -c aes128-gcm@...  — AES-NI 하드웨어 가속: 가장 빠른 안전한 SSH 암호
_SSH_OPTS="-o Compression=no -c aes128-gcm@openssh.com"
[[ -n "${SSH_PORT}" ]] && _SSH_OPTS="${_SSH_OPTS} -p ${SSH_PORT}"

# ── Remote destination auto-detection ────────────────────────────────────────
# Priority: /runpod-volume (RunPod) > /workspace (Vast.ai) > ~/ (local disk)
_VOLUME_MOUNT="/runpod-volume"
_WORKSPACE="/workspace"
if [[ -z "${REMOTE_DIR:-}" ]]; then
    echo "→ probing remote storage..."
    if ssh ${_SSH_OPTS} -o ConnectTimeout=8 "${_HOST}" \
        "mkdir -p '${_VOLUME_MOUNT}' && touch '${_VOLUME_MOUNT}/.upload_probe' && rm -f '${_VOLUME_MOUNT}/.upload_probe'" \
        2>/dev/null; then
        REMOTE_DIR="${_VOLUME_MOUNT}"
        echo "→ RunPod Network Volume → ${REMOTE_DIR}"
    elif ssh ${_SSH_OPTS} -o ConnectTimeout=8 "${_HOST}" \
        "mkdir -p '${_WORKSPACE}' && touch '${_WORKSPACE}/.upload_probe' && rm -f '${_WORKSPACE}/.upload_probe'" \
        2>/dev/null; then
        REMOTE_DIR="${_WORKSPACE}"
        echo "→ Vast.ai /workspace → ${REMOTE_DIR}"
    else
        REMOTE_DIR="~"
        echo "→ No persistent volume — container home (${REMOTE_DIR})"
        echo "  Warning: container 종료 시 데이터 소멸. Volume 연결 권장."
    fi
fi

# ── RESYNC: 코드/config만 변경된 경우 — data 제외 delta 전송 ─────────────────
# Bundle 생성 없이 변경된 파일만 rsync로 전송. 수 초~수십 초.
# 초기 업로드 또는 data/processed 변경 시: RESYNC=0(기본) 사용.
if [[ "${RESYNC}" == "1" ]]; then
    _REMOTE_PROJ="${REMOTE_DIR}/project_transformer"
    echo ""
    echo "┌─ RESYNC mode (data/processed 제외 — delta 전송) ──────────────────"
    echo "│  remote: ${_HOST}:${_REMOTE_PROJ}"
    echo "└────────────────────────────────────────────────────────────────────"
    echo ""
    rsync \
        -avhP \
        --no-compress \
        --rsh="ssh ${_SSH_OPTS}" \
        --exclude='.env' \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        --exclude='data/' \
        --exclude='checkpoints/' \
        --exclude='logs/' \
        --exclude='output/' \
        --exclude='pulled*/' \
        --exclude='inspection/' \
        --exclude='sweep_results/' \
        --exclude='lightning_logs/' \
        --exclude='*.tgz' \
        --exclude='*.tar.zst' \
        --exclude='*.tar' \
        --exclude='*.sha256' \
        ./ "${_HOST}:${_REMOTE_PROJ}/"
    echo ""
    echo "✔ 코드 재배포 완료 → ${_HOST}:${_REMOTE_PROJ}"
    echo ""
    echo "src/jam_transformer/ 변경 시 서버에서 pip 재설치:"
    echo "  ssh ${SSH_HOST} 'cd ${_REMOTE_PROJ} && pip install -e . -q'"
    exit 0
fi

# ── Auto-detect bundle ──────────────────────────────────────────────────────
if [[ -n "${BUNDLE:-}" ]]; then
    BUNDLE_PATH="${BUNDLE}"
else
    # Prefer .tgz; fall back to .tar.zst, .tar
    # zst를 먼저 탐색 — package_for_server.py 기본 출력이 .tar.zst
    for _candidate in jam_tx_bundle.tar.zst jam_tx_bundle.tgz jam_tx_bundle.tar; do
        if [[ -f "${_candidate}" ]]; then
            BUNDLE_PATH="${_candidate}"
            break
        fi
    done
    : "${BUNDLE_PATH:?No bundle found. Run: python scripts/package_for_server.py}"
fi

[[ -f "${BUNDLE_PATH}" ]] || { echo "Bundle not found: ${BUNDLE_PATH}"; exit 1; }

BUNDLE_SIZE="$(du -sh "${BUNDLE_PATH}" | cut -f1)"
echo "→ bundle      : ${BUNDLE_PATH}  (${BUNDLE_SIZE})"
echo "→ destination : ${_HOST}:${REMOTE_DIR}"
echo "→ ssh cipher  : aes128-gcm@openssh.com  (hardware AES)"
echo "→ compression : disabled (bundle is pre-compressed)"
echo ""

# ── Transfer ────────────────────────────────────────────────────────────────
rsync \
    --rsh="ssh ${_SSH_OPTS}" \
    --whole-file \
    --progress \
    --stats \
    "${BUNDLE_PATH}" \
    "${_HOST}:${REMOTE_DIR}"

echo ""
echo "✔ Transfer complete."

# ── SHA-256 integrity check ─────────────────────────────────────────────────
if [[ "${SKIP_VERIFY}" != "1" ]]; then
    # Look for a .sha256 file next to the bundle (written by package_for_server.py)
    _SHA_FILE="${BUNDLE_PATH%.*}.sha256"
    # Handle double extension like .tar.zst → strip both
    if [[ ! -f "${_SHA_FILE}" ]]; then
        _SHA_FILE="${BUNDLE_PATH%.*}"
        _SHA_FILE="${_SHA_FILE%.*}.sha256"
    fi

    if [[ -f "${_SHA_FILE}" ]]; then
        echo "→ uploading checksum file: ${_SHA_FILE}"
        rsync \
            --rsh="ssh ${_SSH_OPTS}" \
            --whole-file \
            "${_SHA_FILE}" \
            "${_HOST}:${REMOTE_DIR}"

        echo "→ verifying SHA-256 on remote ..."
        # ssh into the remote and run sha256sum -c from the remote dir
        ssh ${_SSH_OPTS} "${_HOST}" \
            "cd ${REMOTE_DIR} && sha256sum -c '$(basename "${_SHA_FILE}")'"
        echo "✔ Integrity verified."
    else
        echo "  (no .sha256 file found — skipping integrity check)"
        echo "  Re-run package_for_server.py to generate one,"
        echo "  or set SKIP_VERIFY=1 to suppress this message."
    fi
fi

# ── Next step ────────────────────────────────────────────────────────────────
echo ""
_bn="$(basename "${BUNDLE_PATH}")"
# Determine where the project_transformer/ tree will land after extraction.
# We extract into REMOTE_DIR, so the tree lives at REMOTE_DIR/project_transformer.
_extract_root="${REMOTE_DIR}"
_proj_path="${_extract_root}/project_transformer"
echo "On the server:"
echo "  # 1. Extract"
case "${_bn}" in
    *.tgz|*.tar.gz) echo "  cd ${_extract_root} && tar xzf ${_bn}" ;;
    *.tar.zst)       echo "  cd ${_extract_root} && zstd -d ${_bn} | tar x" ;;
    *.tar)           echo "  cd ${_extract_root} && tar xf  ${_bn}" ;;
esac
echo "  cd ${_proj_path}"
echo ""
echo "  # 2. Set up secrets (.env is excluded from the bundle — create it now)"
echo "  cp .env.example .env"
echo "  nano .env   # fill in: RUNPOD_API_KEY, WANDB_API_KEY (optional)"
echo ""
echo "  # 3. Install + verify"
echo "  bash server/00_bringup.sh"
if [[ "${REMOTE_DIR}" == "${_VOLUME_MOUNT}" ]]; then
    echo ""
    echo "Volume benefits:"
    echo "  • Project files survive container restarts and pod stop/start."
    echo "  • After training: stop the GPU pod (auto or manual) — data stays on Volume."
    echo "  • To retrieve checkpoints without GPU billing:"
    echo "      1. Attach this Volume to a CPU-only pod."
    echo "      2. Run: SSH_HOST=root@<cpu_pod_ip> ./server/90_fetch_artifacts.sh"
fi
