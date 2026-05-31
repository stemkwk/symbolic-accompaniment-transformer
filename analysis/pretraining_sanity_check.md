# 🛡️ 학습 전 오류 검증 및 자가 진단 전략 보고서 (Pre-training Sanity Check)

본 보고서는 **Symbolic Jam Transformer** 프로젝트의 대형 학습(Training Run)을 시작하기 전, 코드의 수치적·기능적 정합성을 검증하고 자원을 낭비하기 전에 오류를 미리 감지 및 차단하는 자가 진단(Proactive Debugging) 전략을 정리한 문서입니다.

---

## 1. 🔍 데이터 누수 원천 차단 (Data Leakage Protection)

* **원리**: 오프라인 사전 전조(Pitch Shift) 빌드 방식을 기각하고, 학습 시점에 PyTorch 텐서 연산으로 수행하는 **실시간 데이터 증강(Dynamic Augmentation)**을 채택함으로써 데이터 누수를 근본적으로 해결했습니다.
* **상태**: 
  - 오프라인 전처리 결과물 디렉토리에는 곡당 오직 단 하나의 원본 파일(예: `pop909_001.pt`)만 존재합니다.
  - 데이터셋 분할(`_is_val`) 시 파일명 문자열의 SHA-256 해시값 기준으로 Train/Val을 쪼개므로, 동일한 곡의 다른 전조 버전이 학습 세트와 검증 세트에 교차 분배되는 **오염(Cross-contamination)이 100% 구조적으로 불가능**합니다.
  - 이로 인해 검증 손실값(`val_loss`) 및 Perplexity(`val_ppl`)의 일반화 성능 신뢰도가 고도로 확보됩니다.

---

## 2. 🧪 학습 전 결함 탐지 프로토콜 (Sanity Check Protocol)

대형 유료 GPU 서버나 장기 학습 구동 전, **단 1~2분 만에 결함을 발견하는 단계별 프로필**입니다.

### Step 1: PyTorch Lightning `fast_dev_run`
* **실행 명령**: `python scripts/train.py --fast_dev_run`
* **검증 내용**: 
  - 전체 학습 루프의 1 에폭 대신, 딱 **1스텝의 학습(Forward+Backward) 및 1스텝의 검증**만을 신속하게 통과시킵니다.
  - 모델의 결선 상태, 텐서 차원 불일치, FP16/BF16 정밀도 가속 상태, 옵티마이저 가중치 업데이트 등 시스템의 아키텍처 오류를 학습 시작 즉시 탐지하여 폭발을 막습니다.

### Step 2: 초기 손실값의 수학적 검수 (Initial Loss Check)
* **검수 기준**: 첫 에폭 첫 스텝(Step 1)에서 출력되는 최초 loss 값이 수학적 범위 내에 있는지 대조합니다.
* **수학적 공식**: 
  - 가중치가 초기화된 시점의 모델은 균등한 무작위 확률로 토큰을 예측하므로, 이론적 초기 크로스 엔트로피 손실값은 어휘 사전 크기 $V$에 대해 **$-\ln(1 / V) = \ln(V)$**에 근접해야 합니다.
  - 본 토크나이저의 $V = 173$ 기준, $\ln(173) \approx 5.15$ 이므로 **초기 Loss는 반드시 $4.9 \sim 5.4$ 범위** 내에 들어와야 정상입니다.
  - 만약 초기 Loss가 $8.0$을 넘거나 $NaN$이 찍힌다면, 임베딩이나 타겟 마스킹(`loss_mask`)의 심각한 구현 버그가 존재한다는 뜻입니다.

### Step 3: 데이터 정합성 육안 진단 (`inspect_data.py`)
* **실행 명령**: `python scripts/inspect_data.py`
* **검증 내용**:
  - 데이터셋 인코딩이 화성학적 룰(REMI Interleaving: BAR $\to$ POS $\to$ TRACK $\to$ NOTES)을 완벽히 지키며 멜로디와 반주 트랙이 병합되어 있는지 분포를 가시적으로 점검하고 데이터 전처리 단의 뒤틀림을 예방합니다.

### Step 4: VRAM 처리량 및 OOM 예방 (`dry_run_steps`)
* **실행 명령**: `python scripts/train.py --dry_run_steps 20`
* **검증 내용**:
  - 배치 크기(`batch_size`)와 정밀도 환경에서 VRAM Peak 용량을 측정합니다.
  - GPU 장비 한계 VRAM의 85% 선을 넘지 않는지 미리 파악하여, 학습 중반부에 메모리 부족(OOM)으로 크래시가 나는 현상을 미연에 방지합니다.

---

## 3. ⚠️ 잠재적 버그 및 설정 불안정성 경고 (Proactive Warnings)

### A. Gradio WAV 오디오 정규화 로직 DataType Loss 버그 ✅ 수정 완료
* **위치**: `app.py` L128 (`_to_f32` 함수)
* **문제 (수정 전)**:
  - `_to_f32`가 캐스팅 후 값 범위만 보고 `np.iinfo(np.int16).max`(32767)로 일괄 나눔.
  - **int32 (24/32-bit WAV)**: 32767로 나누면 최대 65536이 남아 믹싱 후 `np.clip`에 의해 전체 신호가 클리핑 → 굉음/노이즈.
  - **uint8 (8-bit WAV)**: 0-255 범위를 32767로 나누면 볼륨이 1/100 이하 → 무음.
* **수정 후 (`app.py` 반영 완료)**:
  ```python
  def _to_f32(arr: np.ndarray) -> np.ndarray:
      if arr.dtype == np.uint8:
          # uint8 WAV: 0..255, silence = 128 (DC offset)
          return (arr.astype(np.float32) - 128.0) / 128.0
      if np.issubdtype(arr.dtype, np.signedinteger):
          # int16, int32 등 — 타입 실제 최대값으로 정규화
          return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
      # float32 / float64 — 캐스팅만
      return arr.astype(np.float32)
  ```

### B. Windows 로컬 환경 구동 시 `torch.compile` 지연 — ✅ 반영됨
* **위치**: `configs/config.yaml` (`model.compile`)
* **현재 상태**: 기본값이 **`compile: false`** 로 설정되어 있어 Windows 로컬에서 안전합니다 (과거 `true` 기본값에서 변경됨). MSVC/MinGW 미비로 인한 컴파일 실패·3~5분 지연이 기본 경로에서 발생하지 않습니다.
* **참고**: Linux 서버에서 20~30% 가속을 원하면 `--set model.compile=true`로 명시적 활성화.

### C. 중간 저장(Checkpointing) 주기 및 Early Stopping 임계치 — ✅ 반영됨
* **위치**: `configs/config.yaml` (`checkpoint_every_n_train_steps`, `early_stopping_patience`, `early_stopping_min_epochs`)
* **현재 상태**: 아래 권장안이 모두 config에 반영되어 있습니다.
  1. **체크포인트 주기**: **`1000` steps** (과거 100 → I/O 병목 회피). 38M 모델 + AdamW 상태 1회 저장 ≈ 400~450MB이므로 100 steps(약 3~5초)마다 쓰면 스루풋 급락 → 1000으로 조정 완료.
  2. **Early Stopping**: **`patience: 10` + `early_stopping_min_epochs: 10`** (과거 15 → 시간·과금 절감). warmup 이후 10 epoch 정체면 plateau/overfit 판단 충분. min_epochs를 동일하게 맞춰 웜업 구간 조기종료 오작동도 방지.

