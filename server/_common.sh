# Sourced by every server/ script. Centralises:
#   - bash strict mode
#   - PROJECT_ROOT detection (works regardless of cwd)
#   - Pretty section / step logging
#   - Sanity checks (GPU, project layout, processed data)
#
# Sourced, not executed — never give this file a shebang or chmod +x.

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate the project root (the directory that contains pyproject.toml). The
# server/ scripts always live one level below it, so we just take the parent
# of this file's directory.
# ---------------------------------------------------------------------------
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${_COMMON_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# ---------------------------------------------------------------------------
# Load .env so RUNPOD_API_KEY / WANDB_API_KEY etc. are available to all
# server scripts without the user having to export them manually.
# `set -a` marks every sourced variable for automatic export so child
# processes (the nohup'd monitor) inherit them too.
# ---------------------------------------------------------------------------
_DOTENV="${PROJECT_ROOT}/.env"
if [[ -f "${_DOTENV}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${_DOTENV}"
    set +a
fi

log_section() { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
log_step()    { printf "\033[1;32m→\033[0m %s\n" "$*"; }
log_warn()    { printf "\033[1;33m!\033[0m %s\n" "$*" >&2; }
log_fail()    { printf "\033[1;31m✗\033[0m %s\n" "$*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 \
        || log_fail "Required command '$1' not found on PATH."
}

require_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_fail "nvidia-smi missing — this script needs a CUDA-capable host."
    fi
    if ! nvidia-smi -L | grep -q "GPU"; then
        log_fail "nvidia-smi runs but reports no GPUs."
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

require_project_layout() {
    [[ -f "${PROJECT_ROOT}/pyproject.toml"      ]] || log_fail "Not a project_transformer checkout: missing pyproject.toml at ${PROJECT_ROOT}"
    [[ -f "${PROJECT_ROOT}/configs/config.yaml" ]] || log_fail "Missing configs/config.yaml at ${PROJECT_ROOT}"
    [[ -d "${PROJECT_ROOT}/src/jam_transformer" ]] || log_fail "Missing src/jam_transformer at ${PROJECT_ROOT}"
}

require_processed_data() {
    local data_dir="${1:-data/processed}"
    [[ -f "${PROJECT_ROOT}/${data_dir}/_dataset_meta.json" ]] || log_fail \
        "No processed data at ${data_dir}. Run scripts/download_pop909.py + scripts/prepare_data.py locally first, then upload."
    local n_shards
    n_shards="$(ls "${PROJECT_ROOT}/${data_dir}"/*.pt 2>/dev/null | wc -l | tr -d ' ')"
    [[ "${n_shards}" -gt 0 ]] || log_fail "No .pt shards under ${data_dir}."
    log_step "${n_shards} tokenized shards detected under ${data_dir}"
}

# ---------------------------------------------------------------------------
# Config defaults loader
#
# 설계 원칙:
#   우선순위:  ENV var  >  configs/config.yaml  >  내장 fallback
#   패턴:      VAR="${VAR:-${CFG_VAR}}"
#
# 호출하면 CFG_* 변수들이 현재 셸에 export됨.
# 직접 매핑(전용 변수):
#   EPOCHS         ← training.epochs
#   BATCH_SIZE     ← training.batch_size
#   LR             ← training.learning_rate
#   COMPILE        ← model.compile
#   DRY_RUN_STEPS  ← training.dry_run_steps  (10_dry_run.sh 전용)
# 그 외 파라미터:  EXTRA="--set section.key=val" 로 전달
#
# 각 스크립트에서 명시적으로 호출해야 함 (00_bringup.sh 제외):
#   source "$(dirname "$0")/_common.sh"
#   load_config_defaults
# ---------------------------------------------------------------------------
load_config_defaults() {
    local py_out
    local _read_cfg="${PROJECT_ROOT}/server/read_config.py"

    if [[ ! -f "${_read_cfg}" ]]; then
        log_warn "server/read_config.py not found — using built-in fallbacks."
        _set_cfg_fallbacks
        return 0
    fi

    if ! py_out="$(python "${_read_cfg}" 2>/dev/null)"; then
        log_warn "read_config.py failed (Python/PyYAML unavailable?) — using built-in fallbacks."
        _set_cfg_fallbacks
        return 0
    fi

    if [[ -z "${py_out}" ]]; then
        log_warn "read_config.py returned empty output — using built-in fallbacks."
        _set_cfg_fallbacks
        return 0
    fi

    eval "${py_out}"
    log_step "Config defaults loaded from configs/config.yaml"
    # shellcheck disable=SC2154
    log_step "  epochs=${CFG_EPOCHS}  batch=${CFG_BATCH_SIZE}  lr=${CFG_LR}  compile=${CFG_COMPILE}"
}

# Built-in fallback values — mirrors configs/config.yaml.
# Update this whenever the YAML defaults change so servers without Python
# still get sane values.
# Built-in fallback values — mirrors configs/config.yaml.
# Update this block whenever the YAML defaults change so servers without
# Python/PyYAML still get sane values (e.g. first-boot before pip install).
_set_cfg_fallbacks() {
    CFG_EPOCHS='200'
    CFG_BATCH_SIZE='64'
    CFG_LR='0.0003'
    CFG_WARMUP_STEPS='500'
    CFG_GRAD_CLIP='1'
    CFG_ACCUM='1'
    CFG_DRY_RUN_STEPS='0'
    CFG_WANDB_PROJECT='jam-transformer'
    CFG_RUN_NAME_BASE='pop909-baseline'
    CFG_ES_PATIENCE='15'
    CFG_COMPILE='false'
    CFG_D_MODEL='512'
    CFG_N_LAYERS='12'
    CFG_TEMPERATURE='1.1'
    CFG_STRUCT_SUP='1.5'
}

# ---------------------------------------------------------------------------
# RunPod Pod 종료
#
# RunPod Community Cloud는 Docker 컨테이너라서 `shutdown -h now`가 무시됨.
# 대신 RunPod GraphQL API로 Pod를 직접 멈춰야 GPU 요금이 중단됨.
#
# 필요한 환경 변수 (서버 .env에 설정):
#   RUNPOD_API_KEY — RunPod 콘솔 → Settings → API Keys 에서 발급
#   RUNPOD_POD_ID  — RunPod가 컨테이너 시작 시 자동으로 주입
#
# 우선순위:
#   1. RunPod API (RUNPOD_API_KEY + RUNPOD_POD_ID 모두 있을 때)
#   2. runpodctl CLI (설치되어 있을 때)
#   3. shutdown -h now (bare-metal 서버 폴백)
# ---------------------------------------------------------------------------
runpod_stop() {
    local pod_id="${RUNPOD_POD_ID:-}"
    local api_key="${RUNPOD_API_KEY:-}"
    local log_prefix="[runpod_stop]"

    # --- 1. RunPod GraphQL API ---
    if [[ -n "${api_key}" && -n "${pod_id}" ]]; then
        echo "${log_prefix} Calling RunPod API to stop pod ${pod_id}..."
        local resp
        resp=$(curl -sf -X POST \
            "https://api.runpod.io/graphql?api_key=${api_key}" \
            -H "Content-Type: application/json" \
            -d "{\"query\":\"mutation{podStop(input:{podId:\\\"${pod_id}\\\"}){id}}\"}" \
            2>&1) && {
            echo "${log_prefix} Pod stop request sent. Response: ${resp}"
            return 0
        }
        echo "${log_prefix} API call failed: ${resp}" >&2
    else
        [[ -z "${api_key}" ]] && echo "${log_prefix} RUNPOD_API_KEY not set." >&2
        [[ -z "${pod_id}" ]]  && echo "${log_prefix} RUNPOD_POD_ID not set."  >&2
    fi

    # --- 2. runpodctl CLI ---
    if command -v runpodctl >/dev/null 2>&1 && [[ -n "${pod_id}" ]]; then
        echo "${log_prefix} Trying runpodctl stop pod ${pod_id}..."
        runpodctl stop pod "${pod_id}" && return 0
        echo "${log_prefix} runpodctl failed." >&2
    fi

    # --- 3. Bare-metal fallback ---
    echo "${log_prefix} WARNING: Could not stop via RunPod API or runpodctl." >&2
    echo "${log_prefix} Falling back to 'shutdown -h now' (only works on bare-metal)." >&2
    echo "${log_prefix} If the GPU is still running after this, stop the Pod manually." >&2
    shutdown -h now 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Persistent storage detection  (platform-aware)
#
# Sets and exports:
#   VOLUME_MOUNTED  — "1" if an external persistent path was found, "0" otherwise
#   CKPT_DIR        — absolute path to use for checkpoints
#   TRAIN_LOG_DIR   — absolute path to use for training logs
#
# Search order:
#   1. /runpod-volume   RunPod Network Volume
#   2. /workspace       Vast.ai default workspace (persists while instance exists)
#   3. local fallback   ${PROJECT_ROOT}/checkpoints
#
# Hard-abort rule (REQUIRE_VOLUME):
#   RunPod Community Cloud wipes /workspace on pod stop — ONLY the Network
#   Volume survives. We detect RunPod by the RUNPOD_POD_ID env var that
#   RunPod injects at container start. When on RunPod AND no external volume
#   is found, we abort to prevent silent data loss.
#
#   On Vast.ai / bare-metal / other clouds the local disk persists, so the
#   local fallback is safe. No abort is triggered.
#
#   Override (local testing without any volume):
#     REQUIRE_VOLUME=0 ./server/20_train.sh
# ---------------------------------------------------------------------------
_VOLUME_CANDIDATES=("/runpod-volume" "/workspace")

detect_volume() {
    VOLUME_MOUNTED="0"

    for _candidate in "${_VOLUME_CANDIDATES[@]}"; do
        if [[ -d "${_candidate}" ]] \
            && touch "${_candidate}/.volume_probe" 2>/dev/null; then
            rm -f "${_candidate}/.volume_probe"
            VOLUME_MOUNTED="1"
            CKPT_DIR="${_candidate}/checkpoints"
            TRAIN_LOG_DIR="${_candidate}/logs"
            log_section "Persistent storage detected"
            log_step "path        : ${_candidate}"
            log_step "checkpoints : ${CKPT_DIR}"
            log_step "logs        : ${TRAIN_LOG_DIR}"
            break
        fi
    done

    if [[ "${VOLUME_MOUNTED}" == "0" ]]; then
        # Only hard-abort when running on RunPod Community Cloud (RUNPOD_POD_ID
        # is auto-injected) AND the user hasn't explicitly opted out.
        # On Vast.ai / bare-metal / other SSH hosts the local disk is persistent.
        if [[ "${REQUIRE_VOLUME:-1}" == "1" && -n "${RUNPOD_POD_ID:-}" ]]; then
            log_fail "RunPod Pod detected (RUNPOD_POD_ID=${RUNPOD_POD_ID}) but no writable " \
"Network Volume found at /runpod-volume. " \
"Attach a Network Volume in the RunPod console — pod stop wipes local disk. " \
"To override: REQUIRE_VOLUME=0 ./server/20_train.sh"
        fi
        CKPT_DIR="${PROJECT_ROOT}/checkpoints"
        TRAIN_LOG_DIR="${PROJECT_ROOT}/logs"
        if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
            # REQUIRE_VOLUME=0 was explicitly set — warn loudly
            log_warn "REQUIRE_VOLUME=0 on RunPod — checkpoints in ${CKPT_DIR}"
            log_warn "Data WILL BE LOST when the pod stops. Use only for quick tests."
        else
            log_step "No external volume — using local disk: ${CKPT_DIR}"
            log_step "(Vast.ai / bare-metal: local disk persists. Fine to continue.)"
        fi
    fi

    export VOLUME_MOUNTED CKPT_DIR TRAIN_LOG_DIR
    mkdir -p "${CKPT_DIR}" "${TRAIN_LOG_DIR}"
}
