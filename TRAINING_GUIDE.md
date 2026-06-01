# 학습 가이드

모델 학습을 도와주셔서 감사합니다. 아래 명령을 순서대로 실행하시면 되고, 다 끝나면 만들어진 체크포인트 파일을 전달해 주시면 됩니다.

- **GPU**: NVIDIA VRAM 16GB 권장 (12GB도 배치 조정으로 가능)
- **OS**: Windows / Linux 모두 가능 — 명령은 PowerShell 기준, Linux 차이만 따로 표기
- **소요 시간**: 환경 세팅·검증 ~20분, 본 학습은 수 시간~수 일 (중단·재개 자유)

> **두 가지만 미리 알아두시면 좋습니다.**
> - `configs/config.yaml`은 수정하지 말아 주세요. 데이터와 설정이 묶여 있어 바꾸면 학습이 거부됩니다. 조정이 필요하면 `--set` 옵션을 씁니다.
> - 학습 중단은 `Ctrl+C`로 해주세요. 강제 종료하면 저장 중이던 체크포인트가 깨질 수 있습니다.

---

## 0. 설치 (한 번만)

| | 링크 |
|---|---|
| Git | https://git-scm.com/download/win |
| Docker Desktop | https://www.docker.com/products/docker-desktop/ |
| NVIDIA 드라이버 (526+) | GeForce Experience 또는 NVIDIA 공홈 |

Docker Desktop은 Settings → General → "Use the WSL 2 based engine"을 켜고 PC를 재시작합니다.
(Linux는 Docker Engine + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

## 1. 클론

```powershell
git clone -b feat/single-stream-accompaniment https://github.com/stemkwk/symbolic-accompaniment-transformer.git
cd symbolic-accompaniment-transformer
```

`-b feat/single-stream-accompaniment`가 중요합니다. 기본 브랜치(`main`)는 데이터와 맞지 않습니다.

## 2. 데이터 다운로드

GitHub **Releases** 탭에서 **`jam_data_processed.zip`** (~150MB) 하나만 받아, 저장소 루트에서 압축을 풉니다 → `data/processed/` 가 생깁니다.

(나머지 zip은 추론·결과물용이라 학습엔 필요 없습니다.)

## 3. 폴더 + 환경 변수

```powershell
New-Item -ItemType Directory -Force data, logs, output, checkpoints
Copy-Item .env.example .env
```

`.env`는 비워둬도 학습됩니다(로그는 CSV로 저장). 진행 상황을 휴대폰/웹에서 실시간으로 보고 싶으면 W&B 키만 채우세요:

```
WANDB_API_KEY=https://wandb.ai/settings 에서 발급
WANDB_NAME=accompaniment-run
```

## 4. 빌드 + GPU 확인

```powershell
docker compose build
docker compose run --rm train python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

첫 빌드는 5~15분(디스크 15GB+ 필요). `CUDA: True`가 나오면 됩니다.
`CUDA: False`면 Docker Desktop → Settings → Resources → WSL Integration에서 배포판을 켜주세요.

---

## 5. 검증 ① 스모크 (~1분)

가짜 데이터로 학습 루프만 빠르게 확인합니다.

```powershell
docker compose run --rm train python scripts/prepare_data.py --synthetic --num_songs 32 --out_dir data/test_processed

docker compose run --rm -e WANDB_DISABLED=true train python scripts/train.py --data_dir data/test_processed --fast_dev_run
```

`val_loss`가 찍히고 끝나면 정상입니다. (초기 loss가 5.x 근처면 정상 초기화)

## 6. 검증 ② 드라이런 (실데이터, VRAM·시간 측정)

```powershell
docker compose run --rm -e WANDB_DISABLED=true train python scripts/train.py --data_dir data/processed --dry_run_steps 50
```

출력의 `peak VRAM`이 GPU 용량보다 작으면 그대로 진행하면 됩니다. 넘치면 아래 *VRAM 부족*을 참고하세요. `est epoch`로 본 학습 시간을 가늠할 수 있습니다.

## 7. 학습

```powershell
docker compose run --rm train python scripts/train.py --data_dir data/processed
```

기본 200 epoch 상한 + 조기 종료라 두면 알아서 가장 좋은 모델에서 멈춥니다. 시간이 빠듯하면 `--epochs 50`처럼 상한을 줄이면 됩니다.

이어서 하기 (중단·재부팅 후):

```powershell
docker compose run --rm train python scripts/train.py --data_dir data/processed --resume checkpoints/last_step.ckpt
```

> Linux 네이티브 GPU면 `--set model.compile=true`로 20~30% 빨라집니다. **Windows(WSL2)에서는 켜지 마세요** — Segmentation fault로 죽습니다.

체크포인트는 `checkpoints/`에 자동 저장됩니다: `last_step.ckpt`(재개용, 1000스텝마다), `best-epoch=*.ckpt`(최종 결과물, val_loss 개선 시).

## 8. 결과 전달

학습이 끝나면(또는 충분하다 싶으면) `checkpoints/`를 압축해 전달해 주세요.

```powershell
Compress-Archive -Path checkpoints\* -DestinationPath jam_checkpoints.zip   # Windows
```
```bash
zip -r jam_checkpoints.zip checkpoints/                                      # Linux
```

용량이 부담되면 `best-epoch=*.ckpt` 하나만 보내셔도 됩니다. W&B를 쓰셨다면 run 링크도 같이 주시면 좋습니다.

여기까지면 끝입니다. 감사합니다! 🎉

---

## VRAM 부족(OOM)

`CUDA out of memory`가 뜨면 배치를 줄이고 누적으로 보완합니다:

```powershell
docker compose run --rm train python scripts/train.py --data_dir data/processed --set training.batch_size=16 --set training.accumulate_grad_batches=4
```

여전히 부족하면 `batch_size=8 ... accumulate_grad_batches=8`로 더 줄이세요.

## 자주 겪는 문제

| 증상 | 해결 |
|---|---|
| `command not found` | 컨테이너 안에 들어가 있는 경우. `exit` 후 PowerShell에서 실행 |
| `CUDA: False` | Settings → Resources → WSL Integration에서 배포판 활성화 |
| `Segmentation fault` | Windows에서 `compile=true`로 켰을 때. 옵션을 빼면 됩니다 |
| `CUDA out of memory` | 위 *VRAM 부족* 참고 |
| 몇 분간 출력 없음 | 정상 (데이터 인덱싱 중) |
| `fingerprint mismatch` | `config.yaml`이 바뀐 경우. `git checkout configs/config.yaml` |
| `No module named ...` | `docker compose build` 재실행 |

---

## 부록 A. 데이터 직접 빌드 (보통 불필요)

Releases zip 대신 원본을 직접 전처리하려는 경우입니다. 세 데이터셋을 같은 `data/processed/`에 누적합니다.

```powershell
docker compose run --rm train python scripts/tools/download_pop909.py --out_dir data/raw/POP909
docker compose run --rm train python scripts/prepare_data.py --pop909_dir data/raw/POP909 --out_dir data/processed

docker compose run --rm train python scripts/tools/download_lakh.py --out_dir data/raw/lmd_clean
docker compose run --rm train python scripts/prepare_data.py --lakh_dir data/raw/lmd_clean --out_dir data/processed

docker compose run --rm train python scripts/tools/download_slakh.py --redux --out_dir data/raw/slakh2100
docker compose run --rm train python scripts/prepare_data.py --slakh_dir data/raw/slakh2100 --out_dir data/processed
```

총 18,161 shard면 완성입니다 (pop909 909 / lakh 15897 / slakh 1355).

## 부록 B. 추론

웹에서 클릭으로 시연하려면 [`INFERENCE_GUIDE.md`](INFERENCE_GUIDE.md)를 보세요. 명령줄로 한 번에 만들려면:

```powershell
docker compose run --rm train python scripts/inference.py --checkpoint checkpoints/best-epoch=XXX-val_loss=X.XXXX.ckpt --melody_midi "path/to/melody.mid" --output output/result.mid
```

추론에는 `jam_soundfonts.zip`(피아노 음색)이 필요합니다.
