#!/usr/bin/env bash
#
# 20_train.sh — kick off real training. Detaches into the background via
# `nohup` so the run survives an SSH disconnect (the most common cause of
# wasted GPU hours on rented boxes).
#
# ── Overridable variables (ENV > config.yaml > built-in default) ─────────────
#
# All of the following are read from configs/config.yaml when not set.
# Override by exporting before calling this script:
#
#   EPOCHS=100          number of training epochs  (training.epochs)
#   BATCH_SIZE=32       base batch size            (training.batch_size)
#   LR=1e-4             peak learning rate         (training.learning_rate)
#   COMPILE=true        toggle torch.compile       (model.compile)
#   RUN_NAME=my-run     W&B / log / CSV run name   (training.wandb_default_run_name
#                         + auto timestamp suffix when not set)
#
# Advanced (less commonly changed — pass via EXTRA to reach train.py --set):
#   EXTRA="--set training.warmup_steps=1000 --set training.grad_clip=0.5"
#
# ── Operational flags (no config.yaml equivalent) ────────────────────────────
#   RESUME=path.ckpt   resume from this checkpoint (or "auto" for last.ckpt)
#   FOREGROUND=1       run in the current shell instead of detaching
#   AUTO_SHUTDOWN=1    halt the RunPod pod automatically when training ends.
#                      Halts only on clean exit; crashes are NOT auto-stopped.
#   SHUTDOWN_GRACE_SEC=120  seconds after training before pod stop.
#
# After launching:
#   tail -f logs/<RUN_NAME>.console.log     # what nohup captured
#   tail -f logs/<RUN_NAME>.log             # loguru file sink
#   nvidia-smi -l 5                         # monitor GPU usage
#   cat logs/<RUN_NAME>.shutdown.log        # AUTO_SHUTDOWN monitor output
#
# To stop gracefully (saves last.ckpt first):
#   kill -TERM $(cat logs/<RUN_NAME>.pid)
#
# To cancel a pending auto-shutdown:
#   kill $(cat logs/<RUN_NAME>.shutdown.pid)

source "$(dirname "$0")/_common.sh"
require_project_layout
require_processed_data "data/processed"
detect_volume   # sets CKPT_DIR / TRAIN_LOG_DIR / VOLUME_MOUNTED

load_config_defaults

# ── Auto-shutdown 사전 검증 ────────────────────────────────────────────────────
# 플랫폼별 동작:
#   RunPod  (RUNPOD_POD_ID 주입됨): GraphQL API 필수 → API key 없으면 abort
#   Vast.ai / 다른 컴퓨터           : shutdown -h now fallback 사용 → API key 불필요
if [[ "${AUTO_SHUTDOWN:-1}" == "1" ]]; then
    if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
        # 확실히 RunPod — API key 없으면 Docker 컨테이너 안에서 종료 불가
        if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
            log_fail "RunPod 감지 (RUNPOD_POD_ID=${RUNPOD_POD_ID}) 이지만 RUNPOD_API_KEY가 없습니다. " \
"학습이 끝나도 Pod가 자동 종료되지 않아 과금이 지속됩니다. " \
"RunPod 콘솔 → Settings → API Keys 에서 발급 후 .env에 추가하세요. " \
"과금 위험 감수 시: AUTO_SHUTDOWN=0 ./server/20_train.sh"
        fi
    else
        # RunPod 이외 (Vast.ai, 다른 컴퓨터, 앨리스 클라우드 등)
        # shutdown -h now 또는 클라우드 자체 설정으로 종료 → API key 불필요
        if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
            log_step "AUTO_SHUTDOWN: RunPod 이외 환경 — shutdown -h now fallback 사용"
        fi
    fi
fi

# ── Resolve variables: ENV > config.yaml ─────────────────────────────────────
EPOCHS="${EPOCHS:-${CFG_EPOCHS}}"
BATCH_SIZE="${BATCH_SIZE:-${CFG_BATCH_SIZE}}"
LR="${LR:-${CFG_LR}}"
COMPILE="${COMPILE:-${CFG_COMPILE}}"
RUN_NAME="${RUN_NAME:-${CFG_RUN_NAME_BASE}-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${TRAIN_LOG_DIR}"

# ── Data directory — 병목 주의 ────────────────────────────────────────────────
# 기본값(data/processed)이 Network Volume 위에 있으면 DataLoader가 매 epoch마다
# NFS를 통해 shard를 읽어 학습이 느려집니다.
# 로컬 SSD로 복사한 뒤 DATA_DIR로 지정하면 I/O 병목이 사라집니다:
#
#   cp -r data/processed /workspace/data_local
#   DATA_DIR=/workspace/data_local ./server/20_train.sh
#
# 체크포인트(CKPT_DIR)는 Volume에 유지 — 데이터만 로컬로 이동하면 됩니다.
DATA_DIR="${DATA_DIR:-data/processed}"

if [[ "${VOLUME_MOUNTED}" == "1" && "${DATA_DIR}" == "data/processed" ]]; then
    log_warn "데이터가 Network Volume 위에 있을 수 있습니다."
    log_warn "DataLoader I/O 병목을 피하려면:"
    log_warn "  cp -r data/processed /workspace/data_local"
    log_warn "  DATA_DIR=/workspace/data_local ./server/20_train.sh"
fi

# ── Build train.py argument list ──────────────────────────────────────────────
ARGS=(
    "--epochs"     "${EPOCHS}"
    "--batch_size" "${BATCH_SIZE}"
    "--lr"         "${LR}"
    "--run_name"   "${RUN_NAME}"
    "--data_dir"   "${DATA_DIR}"
    "--set" "model.compile=${COMPILE}"
    "--set" "training.checkpoint_dir=${CKPT_DIR}"
    "--set" "training.log_dir=${TRAIN_LOG_DIR}"
)

# Resume support: check Volume first, then local project dir.
RESUME_VAL="${RESUME:-}"
if [[ "${RESUME_VAL}" == "auto" ]]; then
    if [[ -f "${CKPT_DIR}/last.ckpt" ]]; then
        RESUME_VAL="${CKPT_DIR}/last.ckpt"
        log_step "auto-resume from ${RESUME_VAL}"
    elif [[ -f "${CKPT_DIR}/last_step.ckpt" ]]; then
        # last_step.ckpt is saved every N steps — use as crash-recovery fallback
        RESUME_VAL="${CKPT_DIR}/last_step.ckpt"
        log_warn "No last.ckpt found — resuming from crash checkpoint ${RESUME_VAL}"
    elif [[ -f "${PROJECT_ROOT}/checkpoints/last.ckpt" ]]; then
        RESUME_VAL="${PROJECT_ROOT}/checkpoints/last.ckpt"
        log_step "auto-resume from local fallback ${RESUME_VAL}"
    elif [[ -f "${PROJECT_ROOT}/checkpoints/last_step.ckpt" ]]; then
        RESUME_VAL="${PROJECT_ROOT}/checkpoints/last_step.ckpt"
        log_warn "No last.ckpt — resuming from local crash checkpoint ${RESUME_VAL}"
    else
        log_warn "RESUME=auto but no checkpoint found — starting fresh."
        RESUME_VAL=""
    fi
fi
if [[ -n "${RESUME_VAL}" ]]; then
    [[ -f "${RESUME_VAL}" ]] || log_fail "RESUME='${RESUME_VAL}' not found."
    ARGS+=("--resume" "${RESUME_VAL}")
fi

if [[ -n "${EXTRA:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${EXTRA} )
    ARGS+=( "${EXTRA_ARGS[@]}" )
fi

CONSOLE_LOG="${TRAIN_LOG_DIR}/${RUN_NAME}.console.log"
PID_FILE="${TRAIN_LOG_DIR}/${RUN_NAME}.pid"

log_section "Training launch"
log_step "run_name   : ${RUN_NAME}"
log_step "epochs     : ${EPOCHS}"
log_step "batch_size : ${BATCH_SIZE}  (before VRAM-tier scaling)"
log_step "lr         : ${LR}"
log_step "compile    : ${COMPILE}"
log_step "checkpoints: ${CKPT_DIR}"
log_step "console log: ${CONSOLE_LOG}"
log_step "loguru log : ${TRAIN_LOG_DIR}/${RUN_NAME}.log"
[[ -n "${EXTRA:-}" ]] && log_step "extra      : ${EXTRA}"

if [[ "${FOREGROUND:-0}" == "1" ]]; then
    log_step "foreground mode — Ctrl-C kills the run."
    exec python scripts/train.py "${ARGS[@]}"
fi

# Background mode. Use `nohup` + `setsid` so the process is fully detached.
# `setsid` makes the child the leader of a new session, immune to SIGHUP.
# Exit-code wrapper: write the actual exit code to a file so the monitor can
# detect crashes reliably (including SIGKILL/OOM where no log lines are emitted).
EXIT_CODE_FILE="${TRAIN_LOG_DIR}/${RUN_NAME}.exit_code"
rm -f "${EXIT_CODE_FILE}"
nohup setsid bash -c \
    "python scripts/train.py $(printf '%q ' "${ARGS[@]}"); echo \$? > $(printf '%q' "${EXIT_CODE_FILE}")" \
    > "${CONSOLE_LOG}" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"

# Give it a moment to either crash or commit to running.
sleep 3
if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    log_fail "Training process exited immediately. See ${CONSOLE_LOG}."
fi

log_section "Training detached"
echo "pid:        ${TRAIN_PID}"
echo "pid file:   ${PID_FILE}"
echo
echo "Monitor:"
echo "  tail -f ${CONSOLE_LOG}"
echo "  tail -f ${TRAIN_LOG_DIR}/${RUN_NAME}.log"
echo "  nvidia-smi -l 5"
echo
echo "Stop gracefully:"
echo "  kill -TERM ${TRAIN_PID}"
if [[ "${VOLUME_MOUNTED}" == "1" ]]; then
    echo
    echo "Volume 사용 중 — 학습 완료 후 GPU Pod를 종료해도 체크포인트가 유지됩니다."
    echo "이후 CPU Pod에 같은 Volume을 연결해 다운로드하세요 (GPU 요금 없음)."
fi

# ---------------------------------------------------------------------------
# AUTO_SHUTDOWN monitor
# ---------------------------------------------------------------------------
# Runs in its own nohup'd sub-shell so it survives SSH disconnects.
# Polls the training PID every 30 s. When the PID disappears:
#   clean exit  → waits SHUTDOWN_GRACE_SEC, then stops the RunPod pod via API
#   crash       → logs a warning and does NOT stop (so you can inspect)
#
# NOTE: `shutdown -h now` is intentionally NOT used — it is silently ignored
# inside RunPod Community Cloud Docker containers. We call the RunPod GraphQL
# API instead, which actually stops the pod and ends GPU billing.
#
# Required: RUNPOD_API_KEY in .env  (see .env.example)
# Auto-set:  RUNPOD_POD_ID         (injected by RunPod at container start)
#
# Defaults to ON (AUTO_SHUTDOWN=1). Disable with AUTO_SHUTDOWN=0 if you want
# to inspect logs or keep the pod alive after training.
# ---------------------------------------------------------------------------
if [[ "${AUTO_SHUTDOWN:-1}" == "1" ]]; then
    SHUTDOWN_GRACE_SEC="${SHUTDOWN_GRACE_SEC:-120}"
    SHUTDOWN_LOG="${TRAIN_LOG_DIR}/${RUN_NAME}.shutdown.log"
    SHUTDOWN_PID_FILE="${TRAIN_LOG_DIR}/${RUN_NAME}.shutdown.pid"

    # Capture credentials at launch time so the nohup'd monitor has them even
    # if .env is not re-sourced inside the sub-shell.
    _RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
    _RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"

    MONITOR_SCRIPT="${TRAIN_LOG_DIR}/${RUN_NAME}.monitor.sh"
    cat > "${MONITOR_SCRIPT}" <<MONITOR_EOF
#!/usr/bin/env bash
TRAIN_PID=${TRAIN_PID}
CONSOLE_LOG=${CONSOLE_LOG}
EXIT_CODE_FILE=${EXIT_CODE_FILE}
GRACE=${SHUTDOWN_GRACE_SEC}
LOG=${SHUTDOWN_LOG}
RUNPOD_API_KEY="${_RUNPOD_API_KEY}"
RUNPOD_POD_ID="${_RUNPOD_POD_ID}"

echo "[shutdown-monitor] started at \$(date)  watching pid \${TRAIN_PID}" >> "\${LOG}"

while kill -0 "\${TRAIN_PID}" 2>/dev/null; do
    sleep 30
done

# Primary crash detection: read exit code written by the wrapper shell.
# This catches SIGKILL/OOM where no error log lines are emitted.
sleep 2  # brief wait for exit code file to be flushed
if [[ -f "\${EXIT_CODE_FILE}" ]]; then
    TRAIN_EXIT=\$(cat "\${EXIT_CODE_FILE}" | tr -d '[:space:]')
else
    # Exit code file missing → process was killed before wrapper could write it
    TRAIN_EXIT="SIGKILL"
fi

if [[ "\${TRAIN_EXIT}" != "0" ]]; then
    echo "[shutdown-monitor] Crash detected (exit=\${TRAIN_EXIT}) at \$(date) — NOT stopping pod." >> "\${LOG}"
    echo "[shutdown-monitor] Check \${CONSOLE_LOG} then stop manually." >> "\${LOG}"
    exit 1
fi

# Secondary check: scan log for known error keywords as belt-and-suspenders
LAST_LINES=\$(tail -20 "\${CONSOLE_LOG}" 2>/dev/null)
if echo "\${LAST_LINES}" | grep -qiE "(traceback|exception|killed by signal|cuda error|out of memory)"; then
    echo "[shutdown-monitor] Error keywords in log despite exit=0 — NOT stopping pod." >> "\${LOG}"
    echo "[shutdown-monitor] Investigate: \${CONSOLE_LOG}" >> "\${LOG}"
    exit 1
fi

echo "[shutdown-monitor] Training ended cleanly at \$(date)." >> "\${LOG}"
echo "[shutdown-monitor] Waiting \${GRACE}s grace period for disk flush..." >> "\${LOG}"
sleep "\${GRACE}"
echo "[shutdown-monitor] Stopping RunPod pod now." >> "\${LOG}"

_stopped=0

# 1. RunPod GraphQL API (preferred — works inside Docker containers)
if [[ -n "\${RUNPOD_API_KEY}" && -n "\${RUNPOD_POD_ID}" ]]; then
    resp=\$(curl -sf -X POST \
        'https://api.runpod.io/graphql?api_key=${_RUNPOD_API_KEY}' \
        -H 'Content-Type: application/json' \
        -d '{"query":"mutation{podStop(input:{podId:\"${_RUNPOD_POD_ID}\"}){id}}"}' 2>&1)
    if [[ \$? -eq 0 ]]; then
        echo "[shutdown-monitor] Pod stopped via API. Response: \${resp}" >> "\${LOG}"
        _stopped=1
    else
        echo "[shutdown-monitor] API call failed: \${resp}" >> "\${LOG}"
    fi
fi

# 2. runpodctl CLI fallback
if [[ \${_stopped} -eq 0 ]] && command -v runpodctl >/dev/null 2>&1 && [[ -n "\${RUNPOD_POD_ID}" ]]; then
    runpodctl stop pod "\${RUNPOD_POD_ID}" >> "\${LOG}" 2>&1 && _stopped=1
fi

# 3. Bare-metal fallback (no-op in containers, but harmless)
if [[ \${_stopped} -eq 0 ]]; then
    echo "[shutdown-monitor] WARNING: Could not stop via API or runpodctl." >> "\${LOG}"
    echo "[shutdown-monitor] Add RUNPOD_API_KEY to .env and re-upload the bundle." >> "\${LOG}"
    echo "[shutdown-monitor] Please stop the Pod manually in the RunPod console." >> "\${LOG}"
    shutdown -h now 2>/dev/null || true
fi
MONITOR_EOF
    chmod +x "${MONITOR_SCRIPT}"

    nohup setsid bash "${MONITOR_SCRIPT}" \
        >> "${SHUTDOWN_LOG}" 2>&1 < /dev/null &
    SHUTDOWN_PID=$!
    echo "${SHUTDOWN_PID}" > "${SHUTDOWN_PID_FILE}"

    echo
    echo "AUTO_SHUTDOWN enabled (RunPod API):"
    echo "  grace period : ${SHUTDOWN_GRACE_SEC}s after training ends"
    echo "  monitor pid  : ${SHUTDOWN_PID}  (saved to ${SHUTDOWN_PID_FILE})"
    echo "  monitor log  : ${SHUTDOWN_LOG}"
    if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
        echo
        echo "  !! RUNPOD_API_KEY not set in .env — auto-stop will NOT work !!"
        echo "     Get key: https://www.runpod.io/console/user/settings → API Keys"
        echo "     Add to .env, then re-upload the bundle."
    fi
    echo
    echo "Disable auto-shutdown:  AUTO_SHUTDOWN=0 ./server/20_train.sh"
    echo "Cancel after launch:    kill \$(cat ${SHUTDOWN_PID_FILE})"
fi
