# 학습 가이드 — 부탁받은 분을 위한 안내

> **대상 독자**: 프로젝트 소유자에게 "모델 학습 좀 돌려달라"고 부탁받은 분
> **당신이 할 일**: 처음부터(밑바닥부터) 모델을 학습시키고, **완성된 체크포인트 파일을 소유자에게 돌려주는 것**
> **권장 GPU**: NVIDIA VRAM **16GB** (RTX 4060 Ti / 4070 Ti Super 등). 12GB도 배치 조정으로 가능
> **OS**: Windows(권장) 또는 Linux 모두 가능. 명령은 Windows PowerShell 기준이며, Linux 차이는 그때그때 표기합니다.

딥러닝·Docker 경험이 없어도 **이 문서의 명령을 위에서 아래로 복사-붙여넣기**만 하면 됩니다.

---

## 🗺️ 한눈에 보기 — 전체 흐름

```
0. 소프트웨어 설치 (Git · Docker Desktop · NVIDIA 드라이버)   ── 한 번만
1. 저장소 클론
2. 학습 데이터 1개 다운로드 (jam_data_processed.zip)          ── Releases에서
3. 폴더 생성 + .env (W&B는 선택)
4. Docker 빌드 + GPU 인식 확인
─────────────────────────────────────────────────────────────
5. 검증 ① 스모크 테스트   (합성 데이터, 1 step, ~1분)          ← 여기까지 OK면 환경 정상
6. 검증 ② 드라이런        (실데이터, 50 step, VRAM·시간 측정)  ← 학습 진행 여부 결정
─────────────────────────────────────────────────────────────
7. 학습 시작 (몇 시간~며칠, 중단/재개 자유)
8. 진행 확인
9. ★ 결과(체크포인트) 압축해서 소유자에게 전달               ← 최종 목표
```

> **시간 약속**: 본 학습은 GPU에서 **수 시간~수 일**이 걸릴 수 있습니다.
> 5·6단계까지만 먼저 끝내 두면 약 20분이면 됩니다. 거기서 출력된 예상 시간을 보고
> 본 학습을 밤새 돌릴지, 며칠에 나눠 돌릴지 정하면 됩니다. **언제든 중단했다가 이어서 할 수 있습니다.**

---

## ⚠️ 시작 전 필독

**모든 명령은 Windows PowerShell** (Linux는 터미널)에서 입력합니다.

`docker compose run ...` 으로 컨테이너를 실행하면 내부에 셸이 열릴 수 있습니다.
프롬프트가 `root@xxxxxxxx:/app#` 처럼 보이면 컨테이너 **안**에 들어간 것입니다.
이 상태에서 `docker` 명령을 치면 `command not found` 오류가 납니다.

```bash
exit   # 컨테이너에서 나오기
```

PowerShell 프롬프트(`PS C:\...>`)로 돌아온 뒤 명령을 입력하세요.

> 🚫 **`configs/config.yaml`은 절대 수정하지 마세요.** 학습 데이터와 설정이 지문(fingerprint)으로
> 묶여 있어, 토크나이저·데이터 관련 설정을 바꾸면 학습이 거부됩니다.
> 배치 크기 등 조정이 필요하면 아래 안내대로 **`--set` 옵션으로만** 바꾸세요.

---

## 0. 사전 준비 (소프트웨어 설치 — 한 번만)

| 소프트웨어 | 설명 | 설치 링크 |
|---|---|---|
| **Git** | 저장소 클론 | https://git-scm.com/download/win |
| **Docker Desktop** | 컨테이너 환경 | https://www.docker.com/products/docker-desktop/ |
| **NVIDIA 드라이버** | GPU 드라이버 (526.x 이상) | GeForce Experience 또는 NVIDIA 공홈 |

- Docker Desktop 설치 후 **Settings → General → "Use the WSL 2 based engine" 체크** 확인.
- 설치가 끝나면 PC를 **재시작**합니다.
- (Linux) Docker Engine + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 설치.

---

## 1. 저장소 클론

```powershell
git clone https://github.com/stemkwk/symbolic-accompaniment-transformer.git
cd symbolic-accompaniment-transformer
```

---

## 2. 학습 데이터 다운로드 (`jam_data_processed.zip` 한 개만)

학습에 필요한 건 **전처리된 데이터 한 개뿐**입니다.
모델 가중치(`jam_checkpoints.zip`)는 **당신이 이번에 만들어 낼 결과물**이므로 받을 필요가 없습니다.

| 파일 | 학습에 필요? |
|---|---|
| **`jam_data_processed.zip`** (전처리 데이터, ~150MB) | ✅ **필요 — 이것만 받으면 됨** |
| `jam_soundfonts.zip` (피아노 음색) | ❌ 추론(반주 생성) 전용. 학습에는 불필요 |
| `jam_checkpoints.zip` (모델 가중치) | ❌ 학습으로 새로 만들 결과물 (받지 않음) |

1. GitHub 저장소 상단 **Releases** 탭 클릭
2. 최신 릴리즈에서 **`jam_data_processed.zip`** 다운로드
3. **저장소 루트 폴더**에서 압축 해제 → `data/processed/` 폴더가 자동 생성됨

압축 해제 후 구조:

```
symbolic-accompaniment-transformer/
└── data/
    └── processed/
        ├── pop909_*.pt
        ├── lakh_*.pt
        ├── slakh_*.pt
        ├── _chunk_index.json
        └── _dataset_meta.json
```

> 데이터를 직접 처음부터 빌드하고 싶다면 → **부록 A** 참고 (소유자/고급 사용자용, 보통 불필요).

---

## 3. 폴더 생성 + 환경 변수(.env)

```powershell
New-Item -ItemType Directory -Force data, logs, output, checkpoints
Copy-Item .env.example .env
```

`.env`는 **선택 사항**입니다. 아무것도 안 채워도 학습은 정상 동작합니다(로그는 CSV로 자동 저장).

**학습 진행 상황을 휴대폰/브라우저에서 실시간으로 보고 싶다면** W&B만 설정하세요:

```powershell
notepad .env
```
```
WANDB_API_KEY=여기에_https://wandb.ai/settings_에서_발급받은_키
WANDB_NAME=accompaniment-run
```

> `HF_TOKEN`·`RUNPOD_API_KEY` 등 나머지 항목은 **이 작업에선 비워두면 됩니다.**

---

## 4. Docker 빌드 + GPU 확인

Docker Desktop 트레이 고래 아이콘이 **초록색**인지 먼저 확인하세요.

```powershell
docker compose build
```

PyTorch + CUDA를 받으므로 **처음 한 번만 5~15분** 걸립니다(디스크 여유 **15GB+** 필요).
부족하면 `docker system prune -f` 로 정리 후 재시도.

GPU 인식 확인:

```powershell
docker compose run --rm train python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

정상 출력:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
```

`CUDA: False` → Docker Desktop → Settings → Resources → WSL Integration 에서 사용 중인 배포판 활성화 확인.

---

## 5. 검증 ① 스모크 테스트 (합성 데이터, ~1분)

실제 데이터를 건드리기 전에, **학습 루프 자체가 정상인지** 가짜 데이터 32곡으로 1 스텝만 돌려봅니다.

```powershell
# (1) 합성 데이터 생성 — 약 1분
docker compose run --rm train python scripts/prepare_data.py `
  --synthetic --num_songs 32 --out_dir data/test_processed

# (2) 1 스텝 스모크 테스트
docker compose run --rm -e WANDB_DISABLED=true train `
  python scripts/train.py `
  --data_dir data/test_processed `
  --fast_dev_run `
  --set training.log_to_file=false `
  --set training.csv_logger_enabled=false
```

아래처럼 **`val_loss`가 찍히고 끝나면 정상**입니다:

```
Epoch 0: 100%|██████████| 1/1 [00:02<00:00, train_loss_step=5.330, val_loss=5.250, val_ppl=183.0]
`Trainer.fit` stopped: `max_steps=1` reached.
```

> 초기 loss가 **5.x (≈ ln(173))** 근처면 모델이 올바르게 초기화된 것입니다.
> 시작 후 잠시 아무것도 안 떠도 정상 — PyTorch 임포트 + 인덱싱 중입니다.

---

## 6. 검증 ② 드라이런 — VRAM·시간 측정 (학습 머신에서)

**실제 데이터**로 50 스텝만 돌려 VRAM 사용량과 epoch당 시간을 미리 잽니다.
이 출력으로 **본 학습을 얼마나 돌릴지** 판단합니다.

```powershell
docker compose run --rm -e WANDB_DISABLED=true train `
  python scripts/train.py `
  --data_dir data/processed `
  --dry_run_steps 50
```

출력 예시:
```
  initial loss: 5.22  OK near ln(173)=5.15
  measured: 50 steps in 38.2s  →  764.0 ms/step
  est epoch   : 80 min
  est 200 ep  : ~268 h   (Early Stopping 미적용 시 상한)
  peak VRAM   : 11.2 GB
```

**읽는 법:**
- `peak VRAM` 이 GPU 용량보다 작으면 ✅ 그대로 진행. 넘치면(OOM) → 아래 *VRAM 부족* 참고.
- `est 200 ep` 는 **최대 상한**입니다. 실제로는 **조기 종료(Early Stopping)** 가 보통 훨씬 일찍 멈춥니다.
- 며칠을 돌리기 어렵다면 7단계에서 **`--epochs`로 상한을 정하거나**, 중간에 멈췄다가 이어서 돌리면 됩니다.

---

## 7. 학습 시작

### Windows + Docker Desktop (가장 흔한 경우)

```powershell
docker compose run --rm train `
  python scripts/train.py `
  --data_dir data/processed
```

- 기본 200 epoch 상한 + 조기 종료. **그냥 두면 알아서 가장 좋은 모델에서 멈춥니다.**
- 시간이 부족하면 상한을 줄여서: `... --epochs 50`
- **밤새/며칠 돌리기**: 창을 닫지 말고 두거나, 중간에 `Ctrl+C`로 멈춰도 됩니다.

**중단했다가 이어서 하기** (전원·재부팅 후에도):

```powershell
docker compose run --rm train `
  python scripts/train.py `
  --data_dir data/processed `
  --resume checkpoints/last_step.ckpt
```

### Linux 서버 / native GPU — `torch.compile`로 가속 (선택)

native Linux 환경이면 `torch.compile`을 켜서 **20~30% 빠르게** 돌릴 수 있습니다:

```bash
docker compose run --rm train \
  python scripts/train.py \
  --data_dir data/processed \
  --set model.compile=true
```

> ⚠️ **Windows + Docker Desktop(WSL2)에서는 `compile=true`를 켜지 마세요** — `Segmentation fault`로 죽습니다.
> Windows에서는 기본값(`false`) 그대로 두면 됩니다.

| 자주 쓰는 옵션 | 설명 |
|---|---|
| `--epochs 50` | epoch 상한을 줄임 (시간 절약) |
| `--resume checkpoints/last_step.ckpt` | 옵티마이저 상태까지 포함해 이어서 학습 |
| `--set training.batch_size=16 --set training.accumulate_grad_batches=4` | VRAM 부족 시 (아래 참고) |
| `--set model.compile=true` | Triton JIT 가속 (**native Linux 전용**) |

---

## 8. 학습 진행 확인

진행 줄 예시:
```
Epoch 42/200: 100%|██████████| 312/312 [08:14<00:00, train_loss=1.234, val_loss=1.089, lr=2.8e-4]
```
- `val_loss`가 **꾸준히 내려가면** 정상입니다.
- W&B 키를 넣었다면 https://wandb.ai 의 해당 run에서 그래프로도 볼 수 있습니다.

**체크포인트는 자동 저장**됩니다 (`checkpoints/` 폴더):
- `last_step.ckpt` — 1000 스텝마다 (전원 차단 대비, **재개용**)
- `last.ckpt` — 5 epoch마다
- `best-epoch=XXX-val_loss=X.XXXX.ckpt` — val_loss가 개선될 때마다 (**최종 결과물**)

---

## 9. ★ 결과물(체크포인트) 소유자에게 전달

학습이 끝나거나(또는 충분히 돌렸다고 판단되면), `checkpoints/` 폴더를 통째로 압축해 전달합니다.

**Windows PowerShell:**
```powershell
Compress-Archive -Path checkpoints\* -DestinationPath jam_checkpoints.zip
```
**Linux:**
```bash
zip -r jam_checkpoints.zip checkpoints/
```

- 생성된 **`jam_checkpoints.zip`** 을 Google Drive / WeTransfer 등으로 소유자에게 전달하면 끝입니다.
- 파일이 크면(체크포인트 1개당 약 0.5GB) `best-epoch=*.ckpt` **하나만** 보내도 추론에는 충분합니다.
  학습을 더 이어갈 가능성이 있으면 `last_step.ckpt`도 함께 보내세요.

> W&B를 썼다면, run 페이지 링크(loss 곡선)도 함께 알려주면 소유자가 학습 품질을 확인하기 좋습니다.

**여기까지 하면 부탁받은 작업 완료입니다. 감사합니다!** 🎉

---

## 🛟 VRAM 부족(OOM) 시 대처

`CUDA out of memory` 가 뜨면 배치를 줄이고 누적으로 보완합니다(유효 배치는 유지):

```powershell
docker compose run --rm train `
  python scripts/train.py `
  --data_dir data/processed `
  --set training.batch_size=16 `
  --set training.accumulate_grad_batches=4
```

여전히 부족하면 `batch_size=8 --set training.accumulate_grad_batches=8` 로 더 줄이세요.

---

## 🧰 자주 겪는 문제

| 증상 | 해결 방법 |
|---|---|
| `command not found` (docker/git) | 컨테이너 **안**에서 명령을 치고 있음. `exit` 후 PowerShell에서 실행 |
| `CUDA: False` | Docker Desktop → Settings → Resources → WSL Integration 에서 배포판 활성화 |
| `Segmentation fault` | Windows(WSL2)에서 `compile=true`로 켰을 때 발생. **옵션을 빼면 됨**(기본값 false) |
| `OOM: CUDA out of memory` | 위 *VRAM 부족* 참고 (`batch_size`↓ + `accumulate_grad_batches`↑) |
| 5~10분간 아무것도 안 뜸 | 정상. 데이터셋 인덱싱 중. 기다리면 됨 |
| `fingerprint mismatch` / 데이터 불일치 | `config.yaml`을 수정했을 가능성. `git checkout configs/config.yaml`로 원복 |
| `No module named ...` | `docker compose build` 재실행 |
| 빌드 중 디스크 오류 | `docker system prune -f` 로 공간 확보 후 재빌드 |
| `wandb` 관련 오류 | 명령에 `-e WANDB_DISABLED=true` 추가 |

---

## 부록 A. 데이터를 직접 처음부터 빌드 (보통 불필요)

Releases의 `jam_data_processed.zip` 대신 원본 데이터셋을 직접 받아 전처리하려는 경우입니다.
**부탁받은 분은 이 부록을 건너뛰고 2단계의 zip을 쓰는 것을 권장합니다.**

세 데이터셋을 모두 같은 `data/processed/` 에 누적합니다(각 실행이 인덱스에 추가됨).

```powershell
# POP909 (~수십 MB, 10~20분)
docker compose run --rm train python scripts/tools/download_pop909.py --out_dir data/raw/POP909
docker compose run --rm train python scripts/prepare_data.py `
  --pop909_dir data/raw/POP909 --out_dir data/processed

# Lakh (수 GB, 1~수 시간)
docker compose run --rm train python scripts/tools/download_lakh.py --out_dir data/raw/lmd_clean
docker compose run --rm train python scripts/prepare_data.py `
  --lakh_dir data/raw/lmd_clean --out_dir data/processed

# Slakh redux (스트리밍 추출, 30분~1시간) — HF_TOKEN 불필요(공개 Zenodo)
docker compose run --rm train python scripts/tools/download_slakh.py --redux --out_dir data/raw/slakh2100
docker compose run --rm train python scripts/prepare_data.py `
  --slakh_dir data/raw/slakh2100 --out_dir data/processed
```

완료 확인:
```powershell
docker compose run --rm train python -c "
import json; d = json.load(open('data/processed/_chunk_index.json'))
p = {}
for k in d:
    s = k.split('_')[0]; p[s] = p.get(s, 0) + 1
print(p)
"
```
예시: `{'pop909': 909, 'lakh': 15897, 'slakh': 1355}` (총 18,161 shard)

---

## 부록 B. 학습된 모델로 반주 생성 (추론) — 소유자용

체크포인트를 받은 소유자가 멜로디로 반주를 생성할 때:

```powershell
docker compose run --rm train `
  python scripts/inference.py `
  --checkpoint checkpoints/best-epoch=XXX-val_loss=X.XXXX.ckpt `
  --melody_midi "path/to/melody.mid" `
  --output output/result.mid
```

> 파일명의 `XXX`는 Tab 키로 자동완성됩니다. 추론에는 `jam_soundfonts.zip`(피아노 음색)이 필요합니다.
