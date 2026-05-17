#!/usr/bin/env bash
#
# watch_and_fetch.sh — LOCAL machine에서 실행 (WSL2 / Git Bash / macOS Terminal).
#
# 서버의 학습 완료를 폴링으로 감지 → 완료 시 아티팩트 자동 다운로드.
#
# ── 동작 순서 ─────────────────────────────────────────────────────────────────
#   1. 서버에서 가장 최근 .pid 파일을 찾아 학습 프로세스 ID를 확인.
#      PID 파일이 아직 없으면 (학습 시작 전) 나타날 때까지 대기.
#   2. 프로세스가 살아있는 동안 POLL_INTERVAL마다 확인.
#   3. 프로세스 종료 후 콘솔 로그 마지막 줄로 크래시 여부 판단.
#      크래시면 경고 출력 후 종료 (수동 확인 권장).
#      정상 종료면 90_fetch_artifacts.sh로 즉시 다운로드.
#   4. AUTO_STOP=1이면 다운로드 후 RunPod API로 Pod 직접 종료.
#
# ── Required ──────────────────────────────────────────────────────────────────
#   SSH_HOST=user@ip[:port]
#
# ── Optional ──────────────────────────────────────────────────────────────────
#   POLL_INTERVAL=60    폴링 간격(초). 서버 SHUTDOWN_GRACE_SEC(기본 120)보다
#                       작게 설정해야 grace period 내에 다운로드가 완료됨.
#   LOCAL_DIR=./pulled  로컬 저장 경로 (90_fetch_artifacts.sh에 전달)
#   PATHS="..."         내려받을 서브디렉토리 (90_fetch_artifacts.sh에 전달)
#   AUTO_STOP=1         다운로드 후 이 스크립트가 RunPod API로 Pod 종료.
#                       RUNPOD_API_KEY 환경 변수 또는 .env 필요.
#                       서버 AUTO_SHUTDOWN=1(기본값) 이면 별도 설정 없어도 자동 종료됨.
#
# ── Examples ──────────────────────────────────────────────────────────────────
#   SSH_HOST=root@1.2.3.4 ./server/watch_and_fetch.sh
#   SSH_HOST=root@1.2.3.4:2222 POLL_INTERVAL=60 ./server/watch_and_fetch.sh
#   SSH_HOST=root@1.2.3.4 AUTO_STOP=1 RUNPOD_API_KEY=rp_xxx ./server/watch_and_fetch.sh
#   SSH_HOST=root@1.2.3.4 LOCAL_DIR=./run42 PATHS="checkpoints" ./server/watch_and_fetch.sh

set -euo pipefail

# Load local .env (RUNPOD_API_KEY 등)
_DOTENV="$(dirname "${BASH_SOURCE[0]}")/../.env"
if [[ -f "${_DOTENV}" ]]; then
    set -a; source "${_DOTENV}"; set +a
fi

: "${SSH_HOST:?Set SSH_HOST=user@host[:port]}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
LOCAL_DIR="${LOCAL_DIR:-./pulled}"
AUTO_STOP="${AUTO_STOP:-0}"

# ── Port parsing ───────────────────────────────────────────────────────────────
SSH_PORT=""
_HOST="${SSH_HOST}"
if [[ "${SSH_HOST}" == *:* ]]; then
    SSH_PORT="${SSH_HOST##*:}"
    _HOST="${SSH_HOST%:*}"
fi

# BatchMode=yes: 패스워드 프롬프트 없이 실패 (키 인증 강제)
# ServerAliveInterval: 폴링 사이 유휴 SSH 연결이 끊기는 것 방지
_SSH_OPTS="-o Compression=no -c aes128-gcm@openssh.com \
    -o ConnectTimeout=15 -o BatchMode=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=accept-new"
[[ -n "${SSH_PORT}" ]] && _SSH_OPTS="${_SSH_OPTS} -p ${SSH_PORT}"

_ssh() { ssh ${_SSH_OPTS} "${_HOST}" "$@"; }

echo "┌─────────────────────────────────────────────────────"
echo "│ watch_and_fetch.sh"
echo "│ server        : ${_HOST}"
echo "│ poll interval : ${POLL_INTERVAL}s"
echo "│ local dir     : ${LOCAL_DIR}"
echo "│ auto_stop     : ${AUTO_STOP}"
echo "└─────────────────────────────────────────────────────"
echo ""

# ── Helper: find most recent training PID file on server ─────────────────────
# Searches known log directories in priority order (volume → local).
# Excludes .shutdown.pid files.
_find_pid_file() {
    # Search in same priority order as detect_volume() in _common.sh:
    #   1. /runpod-volume/logs   — RunPod Network Volume
    #   2. /workspace/logs       — Vast.ai (detect_volume sets TRAIN_LOG_DIR=/workspace/logs)
    #   3. ~/project_transformer/logs — bare-metal / local fallback
    _ssh "
        for d in /runpod-volume/logs /workspace/logs ~/project_transformer/logs; do
            f=\$(ls -t \"\${d}\"/*.pid 2>/dev/null | grep -v 'shutdown' | head -1 || true)
            if [[ -n \"\${f}\" ]]; then echo \"\${f}\"; exit 0; fi
        done
    " 2>/dev/null || true
}

_pid_alive() {
    _ssh "kill -0 '${1}' 2>/dev/null" 2>/dev/null
}

# ── Phase 1: Wait for training to start ───────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Waiting for training to start..."
PID_FILE=""
while [[ -z "${PID_FILE}" ]]; do
    PID_FILE="$(_find_pid_file)"
    if [[ -z "${PID_FILE}" ]]; then
        echo "  no .pid file yet — will check again in ${POLL_INTERVAL}s  (Ctrl-C to abort)"
        sleep "${POLL_INTERVAL}"
    fi
done

TRAIN_PID="$(_ssh "cat '${PID_FILE}'" 2>/dev/null)"
CONSOLE_LOG="${PID_FILE%.pid}.console.log"
SHUTDOWN_PID_FILE="${PID_FILE%.pid}.shutdown.pid"

echo ""
echo "[$(date '+%H:%M:%S')] Training detected"
echo "  pid file   : ${PID_FILE}"
echo "  train pid  : ${TRAIN_PID}"
echo "  console log: ${CONSOLE_LOG}"
echo ""

# Sanity check: is the PID actually alive right now?
if ! _pid_alive "${TRAIN_PID}"; then
    echo "  WARNING: PID ${TRAIN_PID} is already dead — training may have finished"
    echo "  or the PID file belongs to a previous run. Proceeding to download check."
fi

# ── Phase 2: Poll until training ends ─────────────────────────────────────────
T_START="$(date +%s)"
_DOTS=0
while _pid_alive "${TRAIN_PID}"; do
    ELAPSED=$(( $(date +%s) - T_START ))
    printf "\r  [%s] training running... %dh%02dm elapsed  " \
        "$(date '+%H:%M:%S')" $(( ELAPSED/3600 )) $(( ELAPSED%3600/60 ))
    sleep "${POLL_INTERVAL}"
done
echo ""
echo ""
echo "[$(date '+%H:%M:%S')] Training process ${TRAIN_PID} has exited."

# ── Crash detection ────────────────────────────────────────────────────────────
LAST_LINES="$(_ssh "tail -15 '${CONSOLE_LOG}' 2>/dev/null" 2>/dev/null || true)"
_CRASHED=0
if echo "${LAST_LINES}" | grep -qiE \
    "(Traceback|Error:|exception|Killed|OOM|CUDA out of memory|RuntimeError|AssertionError)"; then
    _CRASHED=1
fi

if [[ "${_CRASHED}" == "1" ]]; then
    echo "!! Crash or error detected in console log. Last 15 lines:"
    echo ""
    echo "${LAST_LINES}" | sed 's/^/  │ /'
    echo ""
    echo "Training did NOT end cleanly — skipping automatic download."
    echo "Inspect the server:"
    echo "  ssh ${_HOST} 'tail -100 ${CONSOLE_LOG}'"
    echo ""
    echo "To download anyway:"
    echo "  SSH_HOST=${SSH_HOST} LOCAL_DIR=${LOCAL_DIR} ./server/90_fetch_artifacts.sh"
    exit 1
fi

echo "✔ Training ended cleanly."
echo ""

# ── AUTO_STOP prep: pause server's auto-shutdown to avoid race with download ───
_SHUTDOWN_PID=""
if [[ "${AUTO_STOP}" == "1" ]]; then
    _SHUTDOWN_PID="$(_ssh "cat '${SHUTDOWN_PID_FILE}' 2>/dev/null" 2>/dev/null || true)"
    if [[ -n "${_SHUTDOWN_PID}" ]]; then
        echo "→ pausing server auto-shutdown (pid ${_SHUTDOWN_PID}) during download..."
        _ssh "kill -STOP '${_SHUTDOWN_PID}' 2>/dev/null" 2>/dev/null || true
    fi
fi

# ── Download ───────────────────────────────────────────────────────────────────
echo "→ fetching artifacts → ${LOCAL_DIR}"
echo ""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SSH_HOST LOCAL_DIR
[[ -n "${PATHS:-}" ]] && export PATHS

bash "${SCRIPT_DIR}/90_fetch_artifacts.sh"

echo ""
echo "✔ Download complete → ${LOCAL_DIR}"

# ── Post-download pod shutdown ─────────────────────────────────────────────────
if [[ "${AUTO_STOP}" == "1" ]]; then
    _POD_ID="$(_ssh 'echo "${RUNPOD_POD_ID:-}"' 2>/dev/null || true)"
    _API_KEY="${RUNPOD_API_KEY:-}"

    if [[ -n "${_API_KEY}" && -n "${_POD_ID}" ]]; then
        echo "→ stopping RunPod pod ${_POD_ID} via API..."
        _resp="$(curl -sf -X POST \
            "https://api.runpod.io/graphql?api_key=${_API_KEY}" \
            -H "Content-Type: application/json" \
            -d "{\"query\":\"mutation{podStop(input:{podId:\\\"${_POD_ID}\\\"}){id}}\"}" \
            2>&1)" && echo "✔ Pod stopped. (${_resp})" \
                    || { echo "  API call failed: ${_resp}"; echo "  Stop the pod manually in the RunPod console."; }
    else
        # Resume server's own auto-shutdown monitor instead
        if [[ -n "${_SHUTDOWN_PID}" ]]; then
            echo "→ resuming server auto-shutdown monitor (RUNPOD_API_KEY or POD_ID not set locally)..."
            _ssh "kill -CONT '${_SHUTDOWN_PID}' 2>/dev/null" 2>/dev/null || true
        fi
        [[ -z "${_API_KEY}" ]] && echo "  Tip: export RUNPOD_API_KEY=rp_xxx to enable direct pod stop."
        [[ -z "${_POD_ID}" ]]  && echo "  Tip: RUNPOD_POD_ID is set automatically by RunPod on the server."
    fi
else
    echo ""
    echo "Note: AUTO_STOP is off. The server's auto-shutdown (if enabled) will stop the pod."
    echo "Or stop it manually in the RunPod console."
fi
