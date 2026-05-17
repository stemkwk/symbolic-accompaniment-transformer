#!/usr/bin/env bash
#
# 30_sweep.sh — run a hyperparameter sweep defined by a YAML.
#
# Each run inside the sweep gets its own logs / checkpoints subdir so
# concurrent runs never clobber each other.
# Sweep results are written to the Network Volume (if attached) so they
# survive pod stop — same priority logic as 20_train.sh.
#
# Environment variables:
#   SWEEP=configs/sweep_example.yaml   path to the sweep YAML
#   FOREGROUND=1                       run in current shell (default: detached)
#   AUTO_SHUTDOWN=1                    stop pod after sweep completes (default: 1)
#   SHUTDOWN_GRACE_SEC=120             seconds to wait before stopping pod
#   EXTRA="..."                        forwarded to scripts/sweep.py

source "$(dirname "$0")/_common.sh"
require_project_layout
require_processed_data "data/processed"
detect_volume          # sets CKPT_DIR / TRAIN_LOG_DIR / VOLUME_MOUNTED
load_config_defaults   # makes CFG_* available; sweep YAML drives actual params

# ── Auto-shutdown 사전 검증 (20_train.sh 와 동일 로직) ─────────────────────────
if [[ "${AUTO_SHUTDOWN:-1}" == "1" ]]; then
    if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
        if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
            log_fail "RunPod 감지 (RUNPOD_POD_ID=${RUNPOD_POD_ID}) 이지만 RUNPOD_API_KEY가 없습니다. " \
"Sweep이 끝나도 Pod가 자동 종료되지 않아 과금이 지속됩니다. " \
"RunPod 콘솔 → Settings → API Keys 에서 발급 후 .env에 추가하세요. " \
"과금 위험 감수 시: AUTO_SHUTDOWN=0 ./server/30_sweep.sh"
        fi
    else
        if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
            log_step "AUTO_SHUTDOWN: RunPod 이외 환경 — shutdown -h now fallback 사용"
        fi
    fi
fi

SWEEP="${SWEEP:-configs/sweep_example.yaml}"
[[ -f "${SWEEP}" ]] || log_fail "Sweep YAML not found: ${SWEEP}"

mkdir -p "${TRAIN_LOG_DIR}"
STAMP="$(date +%Y%m%d-%H%M%S)"
CONSOLE_LOG="${TRAIN_LOG_DIR}/sweep_${STAMP}.console.log"
PID_FILE="${TRAIN_LOG_DIR}/sweep_${STAMP}.pid"

# Sweep results directory: on volume if available, else local project dir.
# sweep.py pins each run's checkpoints under OUT_DIR/<run_name>/checkpoints/
# so everything ends up in one place that 90_fetch_artifacts.sh can pull.
if [[ "${VOLUME_MOUNTED}" == "1" ]]; then
    OUT_DIR="${_VOLUME_MOUNT}/sweep_results/${STAMP}"
else
    OUT_DIR="${PROJECT_ROOT}/sweep_results/${STAMP}"
fi

ARGS=(
    "--sweep"    "${SWEEP}"
    "--data_dir" "data/processed"
    "--out_dir"  "${OUT_DIR}"
)
if [[ -n "${EXTRA:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${EXTRA} )
    ARGS+=( "${EXTRA_ARGS[@]}" )
fi

log_section "Sweep launch"
log_step "sweep yaml  : ${SWEEP}"
log_step "out dir     : ${OUT_DIR}"
log_step "console log : ${CONSOLE_LOG}"
[[ "${VOLUME_MOUNTED}" == "1" ]] && log_step "results on Volume — survive pod stop ✓"

if [[ "${FOREGROUND:-0}" == "1" ]]; then
    log_step "foreground mode — Ctrl-C kills the sweep."
    exec python scripts/sweep.py "${ARGS[@]}"
fi

nohup setsid python scripts/sweep.py "${ARGS[@]}" \
    > "${CONSOLE_LOG}" 2>&1 < /dev/null &
SWEEP_PID=$!
echo "${SWEEP_PID}" > "${PID_FILE}"
sleep 3
kill -0 "${SWEEP_PID}" 2>/dev/null \
    || log_fail "Sweep process died immediately — see ${CONSOLE_LOG}"

log_section "✔ Sweep detached"
echo "pid:        ${SWEEP_PID}  (saved to ${PID_FILE})"
echo "monitor:    tail -f ${CONSOLE_LOG}"
echo "results:    ${OUT_DIR}/"
echo ""
echo "Stop sweep:  kill -TERM ${SWEEP_PID}"

# ---------------------------------------------------------------------------
# AUTO_SHUTDOWN monitor — same pattern as 20_train.sh.
# Watches the sweep PID; stops the pod when sweep ends cleanly.
# Crashes (error keyword in console log) do NOT trigger a stop.
# ---------------------------------------------------------------------------
if [[ "${AUTO_SHUTDOWN:-1}" == "1" ]]; then
    SHUTDOWN_GRACE_SEC="${SHUTDOWN_GRACE_SEC:-120}"
    SHUTDOWN_LOG="${TRAIN_LOG_DIR}/sweep_${STAMP}.shutdown.log"
    SHUTDOWN_PID_FILE="${TRAIN_LOG_DIR}/sweep_${STAMP}.shutdown.pid"

    _RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
    _RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"

    MONITOR_SCRIPT="${TRAIN_LOG_DIR}/sweep_${STAMP}.monitor.sh"
    cat > "${MONITOR_SCRIPT}" <<MONITOR_EOF
#!/usr/bin/env bash
SWEEP_PID=${SWEEP_PID}
CONSOLE_LOG=${CONSOLE_LOG}
GRACE=${SHUTDOWN_GRACE_SEC}
LOG=${SHUTDOWN_LOG}
RUNPOD_API_KEY="${_RUNPOD_API_KEY}"
RUNPOD_POD_ID="${_RUNPOD_POD_ID}"

echo "[shutdown-monitor] started at \$(date)  watching sweep pid \${SWEEP_PID}" >> "\${LOG}"

while kill -0 "\${SWEEP_PID}" 2>/dev/null; do
    sleep 30
done

LAST_LINES=\$(tail -5 "\${CONSOLE_LOG}" 2>/dev/null)
if echo "\${LAST_LINES}" | grep -qiE "(error|traceback|exception|killed|oom)"; then
    echo "[shutdown-monitor] Crash detected at \$(date) — NOT stopping pod." >> "\${LOG}"
    echo "[shutdown-monitor] Check \${CONSOLE_LOG} then stop manually." >> "\${LOG}"
    exit 1
fi

echo "[shutdown-monitor] Sweep ended cleanly at \$(date)." >> "\${LOG}"
echo "[shutdown-monitor] Waiting \${GRACE}s grace period..." >> "\${LOG}"
sleep "\${GRACE}"
echo "[shutdown-monitor] Stopping RunPod pod now." >> "\${LOG}"

_stopped=0

if [[ -n "\${RUNPOD_API_KEY}" && -n "\${RUNPOD_POD_ID}" ]]; then
    resp=\$(curl -sf -X POST \
        "https://api.runpod.io/graphql?api_key=\${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation{podStop(input:{podId:\\\"\${RUNPOD_POD_ID}\\\"}){id}}\"}" 2>&1) \
        && { echo "[shutdown-monitor] Pod stopped via API. \${resp}" >> "\${LOG}"; _stopped=1; } \
        || echo "[shutdown-monitor] API call failed: \${resp}" >> "\${LOG}"
fi

if [[ \${_stopped} -eq 0 ]] && command -v runpodctl >/dev/null 2>&1 && [[ -n "\${RUNPOD_POD_ID}" ]]; then
    runpodctl stop pod "\${RUNPOD_POD_ID}" >> "\${LOG}" 2>&1 && _stopped=1
fi

if [[ \${_stopped} -eq 0 ]]; then
    echo "[shutdown-monitor] WARNING: Could not stop via API or runpodctl." >> "\${LOG}"
    echo "[shutdown-monitor] Stop the Pod manually in the RunPod console." >> "\${LOG}"
    shutdown -h now 2>/dev/null || true
fi
MONITOR_EOF
    chmod +x "${MONITOR_SCRIPT}"

    nohup setsid bash "${MONITOR_SCRIPT}" \
        >> "${SHUTDOWN_LOG}" 2>&1 < /dev/null &
    SHUTDOWN_PID=$!
    echo "${SHUTDOWN_PID}" > "${SHUTDOWN_PID_FILE}"

    echo ""
    echo "AUTO_SHUTDOWN enabled:"
    echo "  grace period : ${SHUTDOWN_GRACE_SEC}s after sweep ends"
    echo "  monitor pid  : ${SHUTDOWN_PID}  (${SHUTDOWN_PID_FILE})"
    echo "  monitor log  : ${SHUTDOWN_LOG}"
    if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
        echo ""
        echo "  !! RUNPOD_API_KEY not set — auto-stop will NOT work !!"
        echo "     Add to .env and re-upload the bundle."
    fi
    echo ""
    echo "Cancel auto-shutdown:  kill \$(cat ${SHUTDOWN_PID_FILE})"
fi
