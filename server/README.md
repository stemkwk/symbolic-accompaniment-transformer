# Server workflow

Shell scripts for everything that has to happen on the rented GPU, plus
local-side helpers for upload, download, and monitoring.
Each script is idempotent; numeric prefix indicates the recommended order.

## Script reference

| Script | Where to run | What it does |
|---|---|---|
| `upload_bundle.sh`      | **local** | rsync bundle to server. Auto-detects Network Volume (`/runpod-volume`) vs container home. SHA-256 verified. |
| `00_bringup.sh`         | server | Verify GPU + project layout + data, `pip install -e .`, smoke tests. |
| `05_inspect_data.sh`    | server | Sample N tokenized shards back to MIDI for sanity-check. Optional. |
| `10_dry_run.sh`         | server | Measure ms/step, peak VRAM, estimated epoch time and cost. |
| `20_train.sh`           | server | Real training, detaches via `nohup setsid`. Survives SSH disconnect. |
| `21_resume.sh`          | server | Resume `20_train.sh` from `last.ckpt` (Volume → local fallback). |
| `30_sweep.sh`           | server | Hyperparameter sweep over a YAML. Results go to Volume if attached. |
| `90_fetch_artifacts.sh` | **local** | `rsync` checkpoints / logs / output to laptop. Auto-detects Volume. |
| `watch_and_fetch.sh`    | **local** | Poll server until training ends, then auto-download. No need to stay SSH'd in. |
| `_common.sh`            | n/a | Sourced helpers — never executed directly. |
| `read_config.py`        | n/a | Reads `configs/config.yaml`, emits shell-sourceable `CFG_*` vars. |

## Typical session

```bash
# ── 1. Pack and upload (local machine) ───────────────────────────────────────
python scripts/package_for_server.py          # creates jam_tx_bundle.tgz
SSH_HOST=root@1.2.3.4 ./server/upload_bundle.sh
#   → auto-detects /runpod-volume; prints extraction command

# ── 2. Set up (server) ───────────────────────────────────────────────────────
cd /runpod-volume/project_transformer         # or ~/project_transformer
cp .env.example .env && vi .env               # set RUNPOD_API_KEY, WANDB_API_KEY
bash server/00_bringup.sh                     # install + smoke test
bash server/10_dry_run.sh                     # check ms/step + VRAM

# ── 3. Train (server) ────────────────────────────────────────────────────────
AUTO_SHUTDOWN=1 EPOCHS=80 bash server/20_train.sh
# SSH 끊고 나가도 됨. 학습 완료 후 자동으로 Pod가 종료됨.
```

```bash
# ── 4a. Monitor + auto-download (local machine, runs in parallel) ────────────
SSH_HOST=root@1.2.3.4 ./server/watch_and_fetch.sh
# 학습 완료를 폴링으로 감지 → 자동 다운로드 → (AUTO_STOP=1이면) Pod 종료
# 학습이 아직 시작 안 됐어도 대기하다가 시작되면 감지함.

# ── 4b. 수동 다운로드 ────────────────────────────────────────────────────────
SSH_HOST=root@1.2.3.4 ./server/90_fetch_artifacts.sh
```

```bash
# ── 학습 중 진행 상황 확인 (SSH 재접속 후) ──────────────────────────────────
tail -n 50 logs/*.console.log     # loss 출력
nvidia-smi                        # GPU 사용 확인
cat logs/*.shutdown.log           # auto-shutdown 모니터 상태
```

```bash
# ── 자동 종료 취소 ───────────────────────────────────────────────────────────
kill $(cat logs/*.shutdown.pid)

# ── 학습 즉시 중단 (체크포인트 저장 후 종료) ─────────────────────────────────
kill -TERM $(cat logs/*.pid)
```

## Environment-variable cheat sheet

### `20_train.sh` / `21_resume.sh`

| Var | Default | Purpose |
|---|---|---|
| `EPOCHS`             | config.yaml (`80`) | training epochs (Early Stopping이 먼저 종료 가능) |
| `BATCH_SIZE`         | config.yaml | base batch size (VRAM 티어에 따라 자동 스케일) |
| `LR`                 | config.yaml | peak learning rate |
| `COMPILE`            | config.yaml | `true`/`false` — torch.compile 활성화 |
| `RUN_NAME`           | `<base>-<timestamp>` | log 파일명 + W&B run name |
| `RESUME`             | (none) | `.ckpt` 경로 또는 `auto` (last.ckpt 자동 탐색) |
| `FOREGROUND`         | `0` | `1` = 현재 셸에 붙어서 실행 (디버깅용) |
| `EXTRA`              | (none) | `train.py`에 추가 전달, e.g. `"--set model.d_model=768"` |
| `AUTO_SHUTDOWN`      | `1` | 학습 정상 종료 후 Pod 자동 halt. 크래시 시에는 halt 안 함. |
| `SHUTDOWN_GRACE_SEC` | `120` | halt 전 대기 시간(초). 체크포인트 플러시 보장용. |

### `10_dry_run.sh`

| Var | Default | Purpose |
|---|---|---|
| `DRY_RUN_STEPS` | config.yaml (없으면 `100`) | 타이밍 측정 step 수 (`STEPS` alias도 가능) |
| `BATCH_SIZE`    | config.yaml | override batch size |
| `COMPILE`       | config.yaml | torch.compile 토글 |

### `30_sweep.sh`

| Var | Default | Purpose |
|---|---|---|
| `SWEEP`          | `configs/sweep_example.yaml` | sweep YAML |
| `FOREGROUND`     | `0` | `1` = 현재 셸에 붙어서 실행 |
| `AUTO_SHUTDOWN`  | `1` | sweep 완료 후 Pod 자동 halt |

### `90_fetch_artifacts.sh` (local)

| Var | Default | Purpose |
|---|---|---|
| `SSH_HOST`   | **required** | `user@host[:port]` |
| `REMOTE_DIR` | auto-detect (Volume → local) | 서버 프로젝트 루트 경로 |
| `LOCAL_DIR`  | `./pulled` | 로컬 저장 경로 |
| `PATHS`      | `"checkpoints logs output sweep_results"` | 내려받을 서브디렉토리 |

### `watch_and_fetch.sh` (local)

| Var | Default | Purpose |
|---|---|---|
| `SSH_HOST`      | **required** | `user@host[:port]` |
| `POLL_INTERVAL` | `60` | 폴링 간격(초). `SHUTDOWN_GRACE_SEC`(120)보다 작게 설정 권장. |
| `LOCAL_DIR`     | `./pulled` | 로컬 저장 경로 |
| `PATHS`         | (90_fetch_artifacts.sh 기본값) | 내려받을 서브디렉토리 |
| `AUTO_STOP`     | `0` | `1` = 다운로드 후 이 스크립트가 직접 Pod 종료 (RUNPOD_API_KEY 필요) |

## Stopping a detached run

`20_train.sh`는 PID를 `logs/<run_name>.pid`에 저장한다. SIGTERM은 Lightning이
체크포인트를 저장한 뒤 종료하도록 허용한다:

```bash
kill -TERM "$(cat logs/<run_name>.pid)"
```

`kill -9`는 체크포인트 저장을 건너뜀 — SIGTERM이 30초 이상 멈춰있을 때만 사용.

## Network Volume vs container disk

| | Container disk | Network Volume |
|---|---|---|
| Pod 재시작 | 유지 | 유지 |
| Pod 종료(Stop) | 유지 | 유지 |
| Pod 삭제(Terminate) | **소멸** | **유지** |
| 다른 Pod에 연결 | 불가 | 가능 (CPU pod 등) |
| 경로 | `~/` or `/workspace` | `/runpod-volume` |

`upload_bundle.sh`는 `/runpod-volume`이 쓰기 가능하면 자동으로 그쪽으로 업로드한다.
Volume이 없어도 `podStop`(삭제 아님)을 사용하면 container disk가 유지되므로
학습 완료 후 재접속해서 다운로드할 수 있다.

## When something goes wrong

- **`00_bringup.sh` fails on `pytest`** — 학습 전에 반드시 해결. CUDA / torch /
  Lightning 버전 불일치가 가장 흔한 원인.
- **`10_dry_run.sh` OOM** — `BATCH_SIZE`를 낮추거나 `EXTRA="--set model.d_model=256"`.
- **Training detached but no GPU activity** — console log 확인; 가장 흔한 원인은
  tokenizer fingerprint mismatch (config 변경 후 prepare_data.py 미실행).
- **AUTO_SHUTDOWN이 작동 안 함** — `.env`에 `RUNPOD_API_KEY` 설정 여부 확인.
  `cat logs/*.shutdown.log`로 모니터 상태 확인.
- **SSH dropped, training 생존 여부 확인** —
  `ps -p $(cat logs/<run_name>.pid)` 또는
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`

## Why `nohup setsid python …` and not `screen` / `tmux`?

Both work. `nohup setsid`는 RunPod/Vast.ai의 최소 base image에서도 coreutils만
있으면 동작한다. `tmux`를 선호하면 같은 명령을 그 안에서 실행해도 된다 —
나머지 코드는 detachment 방법에 의존하지 않는다.
