# 학습 가이드 — 처음부터 끝까지

> **대상 독자**: Docker나 딥러닝 환경 설정 경험이 없는 분  
> **목표**: GitHub 클론 → 자산 다운로드 → Docker 빌드 → 학습  
> **GPU**: RTX 4060 Ti / 4070 Ti Super (VRAM 16GB) 기준

---

## ⚠️ 시작 전 필독

**모든 명령은 Windows PowerShell 에서 입력합니다.**

`docker compose run ...` 으로 컨테이너를 실행하면 내부에 bash 셸이 열릴 수 있습니다.  
프롬프트가 `root@xxxxxxxxx:/app#` 처럼 보이면 컨테이너 **안**에 들어간 것입니다.  
이 상태에서 `docker` 명령을 치면 `docker: command not found` 오류가 납니다.

```bash
exit   # 컨테이너에서 나오기
```

PowerShell 프롬프트(`PS C:\...>`)로 돌아온 뒤 명령을 입력하세요.

---

## 0. 사전 준비

### 필수 소프트웨어 설치 (한 번만)

| 소프트웨어 | 설명 | 설치 링크 |
|---|---|---|
| **Git** | 저장소 클론 | https://git-scm.com/download/win |
| **Docker Desktop** | 컨테이너 환경 | https://www.docker.com/products/docker-desktop/ |
| **NVIDIA 드라이버** | GPU 드라이버 (526.x 이상) | GeForce Experience 또는 NVIDIA 공홈 |

Docker Desktop 설치 후 **Settings → General → "Use the WSL 2 based engine" 체크** 확인.  
설치가 끝나면 PC를 재시작합니다.

---

## 1. 저장소 클론

```powershell
git clone https://github.com/<owner>/project_transformer.git
cd project_transformer
```

> `<owner>` 부분은 실제 GitHub 계정명으로 바꿔주세요.

---

## 2. GitHub Releases에서 자산 다운로드

학습에 필요한 **체크포인트(모델 가중치)** 와 **사운드폰트(음색 파일)** 는 Git에 포함되지 않습니다.

1. GitHub 저장소 상단 **Releases** 탭 클릭
2. 최신 릴리즈에서 `jam_light_assets.zip` 다운로드
3. zip 파일을 **저장소 루트 폴더** (`project_transformer/`) 에서 압축 해제

압축 해제 후 폴더 구조:

```
project_transformer/
├── checkpoints/
│   ├── best-epoch=XXX-val_loss=X.XXXX.ckpt
│   └── last.ckpt
├── soundfonts/
│   └── *.sf2
├── Dockerfile
├── docker-compose.yaml
└── ...
```

---

## 3. 필수 디렉터리 생성

```powershell
New-Item -ItemType Directory -Force data, logs, output
```

---

## 4. Docker Desktop 실행 확인

시스템 트레이(우측 하단) Docker 고래 아이콘이 **초록색**인지 확인합니다.

---

## 5. Docker 이미지 빌드

```powershell
docker compose build
```

PyTorch + CUDA 라이브러리를 다운로드하므로 **처음 한 번만 5~15분** 걸립니다.  
이후에는 캐시되어 즉시 완료됩니다.

> 빌드 전 디스크 여유 공간이 **15GB 이상** 필요합니다.  
> 부족하면 `docker system prune -f` 로 오래된 이미지·캐시를 정리하세요.

---

## 6. GPU 인식 확인

```powershell
docker compose run --rm train python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

정상 출력:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Ti
```

`CUDA: False` 가 나오면 Docker Desktop → Settings → Resources → WSL Integration 에서 Ubuntu 활성화 확인.

---

## 7. 데이터 준비

전체 학습 데이터는 **POP909 + Lakh + Slakh** 세 데이터셋을 사용합니다.  
모두 같은 `data/processed/` 에 쌓으면 됩니다. 각 실행이 끝날 때마다 인덱스에 누적됩니다.

### A. 전체 데이터셋 (실제 학습용)

아래 명령을 **순서대로** 실행합니다. 중간에 실패해도 완료된 부분은 재실행하지 않아도 됩니다.

```powershell
# POP909 (~수십 MB, 10~20분)
docker compose run --rm train python scripts/download_pop909.py --out_dir data/raw/POP909
docker compose run --rm train python scripts/prepare_data.py `
  --pop909_dir data/raw/POP909 --out_dir data/processed

# Lakh (수 GB, 1~수 시간)
docker compose run --rm train python scripts/download_lakh.py --out_dir data/raw/lmd_clean
docker compose run --rm train python scripts/prepare_data.py `
  --lakh_dir data/raw/lmd_clean --out_dir data/processed

# Slakh (~96MB, 30분~1시간)
docker compose run --rm train python scripts/download_slakh.py --out_dir data/raw/slakh2100
docker compose run --rm train python scripts/prepare_data.py `
  --slakh_dir data/raw/slakh2100 --out_dir data/processed
```

완료 후 확인:
```powershell
docker compose run --rm train python -c "
import json; d = json.load(open('data/processed/_chunk_index.json'))
prefixes = {}
for k in d:
    p = k.split('_')[0]; prefixes[p] = prefixes.get(p, 0) + 1
print(prefixes)
"
```
출력 예시: `{'pop909': 909, 'lakh': 4823, 'slakh': 1428}`

### B. 합성 데이터 (파이프라인 테스트용)

실제 학습 전에 Docker 환경만 빠르게 확인할 때 사용합니다 (약 1분):

```powershell
docker compose run --rm train python scripts/prepare_data.py `
  --synthetic --num_songs 32 --out_dir data/test_processed
```

---

## 8. 파이프라인 동작 확인 (1회용 스모크 테스트)

합성 데이터(방법 B)로 학습 루프 전체가 정상인지 1 스텝만 돌려봅니다.

```powershell
docker compose run --rm -e WANDB_DISABLED=true train `
  python scripts/train.py `
  --data_dir data/test_processed `
  --fast_dev_run `
  --set model.compile=false `
  --set training.log_to_file=false `
  --set training.csv_logger_enabled=false
```

> `--set model.compile=false` : GPU 전용 커널 컴파일을 건너뜁니다.  
> VRAM이 적은 로컬 검증 머신에서 컴파일 도중 충돌(segfault)이 생길 수 있기 때문입니다.  
> 16GB 학습 머신에서는 이 플래그 없이 그대로 실행합니다.

아래처럼 마지막 줄이 나오면 정상입니다:

```
Epoch 0: 100%|██████████| 1/1 [00:03<00:00]
```

> **시작 후 5~10분간 아무것도 안 뜨는 건 정상입니다.**  
> PyTorch 임포트 + 데이터셋 인덱싱이 끝나야 진행 표시가 나타납니다.

---

## 9. 드라이런 — VRAM·속도 확인 (권장)

**16GB 학습 머신**에서 실행합니다. 실제 데이터로 50 스텝만 돌려 시간과 VRAM을 미리 잽니다.

```powershell
docker compose run --rm -e WANDB_DISABLED=true train `
  python scripts/train.py `
  --data_dir data/processed `
  --dry_run_steps 50
```

출력 예시:
```
  measured: 50 steps in 38.2s  →  764.0 ms/step
  est epoch   : 45.8 min
  est 200 ep  : 152.6 h
  peak VRAM   : 6.31 GB
```

---

## 10. 학습 시작

### 이전 체크포인트에서 이어서 (권장)

```powershell
docker compose run --rm -e WANDB_DISABLED=true train `
  python scripts/train.py `
  --data_dir data/processed `
  --resume checkpoints/last.ckpt `
  --set training.accumulate_grad_batches=2
```

### 처음부터 새로 시작

```powershell
docker compose run --rm -e WANDB_DISABLED=true train `
  python scripts/train.py `
  --data_dir data/processed `
  --set training.accumulate_grad_batches=2
```

| 옵션 | 설명 |
|---|---|
| `--resume checkpoints/last.ckpt` | 에포크·옵티마이저 상태까지 포함해 이어서 학습 |
| `--set training.accumulate_grad_batches=2` | 실제 배치 32 × 2회 누적 → effective batch 64 |

> **last.ckpt vs best-epoch=...ckpt**  
> `last.ckpt` → 학습 재개용 / `best-epoch=...ckpt` → 추론(inference)용

---

## 11. 학습 진행 확인

```
Epoch 42/200: 100%|██████████| 312/312 [08:14<00:00, train_loss=1.234, val_loss=1.089, lr=2.8e-4]
```

체크포인트 자동 저장:
- `checkpoints/last.ckpt` — 5 에포크마다
- `checkpoints/last_step.ckpt` — 100 스텝마다 (전원 차단 대비)
- `checkpoints/best-epoch=XXX-val_loss=X.XXXX.ckpt` — val_loss 개선 시

`Ctrl+C` 로 언제든 중단. 재개 시 `--resume checkpoints/last.ckpt` 사용.

---

## 12. 반주 생성 (추론)

```powershell
docker compose run --rm train `
  python scripts/inference.py `
  --checkpoint checkpoints/best-epoch=XXX-val_loss=X.XXXX.ckpt `
  --melody_midi "path/to/melody.mid" `
  --output output/result.mid
```

> 파일명의 `XXX` 부분은 Tab 키로 자동완성됩니다.

---

## VRAM 부족 시 대처

```powershell
# batch를 줄이고 누적 횟수를 높여 동일한 effective batch 유지
docker compose run --rm -e WANDB_DISABLED=true train `
  python scripts/train.py `
  --data_dir data/processed `
  --resume checkpoints/last.ckpt `
  --set training.batch_size=16 `
  --set training.accumulate_grad_batches=4
```

---

## 자주 겪는 문제

| 증상 | 해결 방법 |
|---|---|
| `docker: command not found` | 컨테이너 안에서 명령을 치고 있음. `exit` 후 PowerShell에서 실행 |
| `CUDA: False` | Docker Desktop → Settings → Resources → WSL Integration에서 Ubuntu 활성화 확인 |
| `Segmentation fault` | `--set model.compile=false` 추가 (VRAM 4~8GB 로컬 머신에서 발생) |
| `OOM: CUDA out of memory` | `--set training.batch_size=16 --set training.accumulate_grad_batches=4` 추가 |
| 5~10분간 아무것도 안 뜸 | 정상. 데이터셋 인덱싱 중. 기다리면 됨 |
| `No module named ...` | `docker compose build` 재실행 |
| 빌드 중 디스크 오류 | `docker system prune -f` 로 공간 확보 후 재빌드 |
| `wandb` 관련 오류 | `-e WANDB_DISABLED=true` 가 `docker compose run` 명령에 포함됐는지 확인. 컨테이너 내부에서 실행하면 이 옵션이 무시됨 |
