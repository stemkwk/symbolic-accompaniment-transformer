# 🕵️ Symbolic Jam Transformer: Comprehensive Project Audit & Review Report

> ⚠️ **2026-05-31 기준 — 2026-05-26 시점 감사 스냅샷.** 여기 "제안 중"으로 적힌 다수 항목이 이후
> 반영되었습니다 (예: Static KV/RoPE 제외하고 대부분). 최신 상태는 [`project_plan.md`](project_plan.md),
> 검증 현황은 [`post_training_verification.md`](post_training_verification.md) 참조.

본 보고서는 **Symbolic Jam Transformer** 프로젝트의 전체 설계와 구현에 대한 종합적인 아키텍처 감사 및 검수 보고서입니다. 이전 단계에서 수정 완료된 내역을 바탕으로, 모델의 구조적 한계점, 추가적으로 확인해야 할 잠재적 오류 요인, 성능 최적화 대안, 학술적/실무적 고려사항 및 최종 학습 전 점검 항목을 기술합니다.

---

## 1. 🏗️ 구조적 한계 (Structural Limitations)

현재 모델 설계 및 인코딩 포맷이 갖는 본질적인 한계점과 이에 따른 파급효과입니다.

### A. 토큰 밀도 및 컨텍스트 윈도우의 제약 (REMI Bottleneck)
* **상황**: 단일 음표(Note)를 표현하기 위해 `[CHROMA, OCTAVE, DUR, VEL]` 4개의 토큰을 결합하고, 시간적/공간적 흐름을 위해 `POS` 및 `TRACK` 토큰을 추가로 나열합니다. 다성부 반주가 밀집될 경우 1마디에 수십 개에서 백여 개의 토큰이 소모됩니다.
* **한계**: 토크나이저 설정의 `max_seq_len: 2560` 기준으로 모델이 한 번에 볼 수 있는 컨텍스트 길이는 약 **12~16마디** 내외입니다.
* **영향**: 인트로-A-B-후렴구로 이어지는 3~4분짜리 대중음악의 거시적 형식 구조(Song Form)나 장기적 모티프(Long-term Motif)의 발전 학습은 아키텍처적으로 불가능합니다.
* **대처 상황**: 생성 시점에 이전 윈도우의 후반부 마디를 새로운 프롬프트 접두사(Prompt Prefix)로 재사용하여 이어 나가는 **"중복/슬라이딩 생성 (Overlapping Generation)"** 전략을 통해 장기 생성을 실현하고 있습니다.

### B. 시간 양자화 그리드 한계 (Rigid Time Quantization)
* **상황**: 현재 1마디를 16분 음표 기준으로 16개 그리드(`POS_0` ~ `POS_15`)로 양자화(Quantization)하여 표현합니다.
* **한계**:
  * 셋잇단음표(Triplet, 3연음)나 셔플/스윙(Shuffle/Swing) 그루브를 토큰 수준에서 표현하는 것이 구조적으로 차단됩니다.
  * 연주자의 정교한 레이드백(Laid-back)이나 미세 타이밍(Micro-timing) 정보를 토큰화 과정에서 상실합니다.
* **대처 상황**: 추론 후처리 단계의 `Humanizer`가 Gaussian Jitter를 입혀 기계적인 느낌을 줄이고 있으나, 이는 후처리일 뿐 모델 스스로가 엇박자 그루브나 swing 스윙감을 학습하여 작곡하는 것은 불가능합니다.

### C. 바-블록 인터리빙(Bar-Block Interleaving)의 방향성
* **상황**: 곡을 $N$마디 단위의 멜로디 블록 $\to$ SEP $\to$ $N$마디 단위의 반주 블록 순으로 인코딩하여 모델에 입력합니다.
* **영향**: 멜로디 정보는 미래 $N$마디를 미리 내다보고 반주를 생성할 수 있지만(Anticipation), 반주 생성 자체는 인과적(Causal)으로만 흘러갑니다.
* **의의**: 이 구조는 실시간 라이브 연주 세션(연주자의 멜로디 입력을 받아 즉흥 반주 생성)에 매우 적합하게 작동하지만, 오프라인 마스터링/편곡과 같이 곡의 후반부 반주 흐름을 고려하여 앞부분 반주를 교정하는 양방향(Bidirectional) 채우기(Infilling) 작업에는 한계가 있습니다.

---

## 2. 🐛 잠재적 오류 요인 및 검토 사항 (Hidden Risks & Error Review)

테스트 스위트를 통과했으나, 실제 대규모 학습 및 배포 단계에서 노출될 수 있는 엣지 케이스와 검토 사항입니다.

### A. PyTorch SDPA 버전 및 마스크 호환성
* **위치**: `src/jam_transformer/model.py`
* **검토 사항**:
  - 인과성 누수를 막기 위해 `T > 1`일 때 2D 불리언 `attn_mask`를 직접 빌드하여 `F.scaled_dot_product_attention`에 전달하고 있습니다.
  - PyTorch 2.0 이하 또는 특정 CUDA 드라이버 환경에서 불리언 마스크의 형태(Shape)나 타입 캐스팅에 따라 FlashAttention 커널이 활성화되지 못하고 느린 Math 커널로 폴백(Fallback)할 가능성이 있습니다.
  - **대책**: GPU 서버 구동 초기 단계에서 `torch.compile`을 적용하거나, SDPA가 최적의 커널(FlashAttention 또는 Memory Efficient Attention)을 타는지 1단계 프로파일링을 수행할 것을 권장합니다.

### B. 오디오 샘플레이트 불일치 및 리샘플링 (Resampling Overhead)
* **위치**: `app.py` L152
* **검토 사항**:
  - Gradio 마이크로 입력된 사용자 음원(`sr_m`, 보통 48000Hz)과 FluidSynth/Pedalboard로 렌더링된 AI 반주(`sr_a`, 44100Hz)의 샘플레이트가 다를 때 리샘플링 연산을 수행합니다: `mel = resample(mel, int(len(mel) * sr_a / sr_m))`.
  - Scipy의 단순 `resample` 함수는 푸리에 변환(FFT) 기반이므로 신호 길이에 따라 메모리를 일시적으로 많이 소모하고 음질 열화가 발생할 수 있습니다.
  - **대책**: 실서비스 혹은 고품질 데모 구동 시에는 `librosa.resample`이나 `torchaudio.transforms.Resample`과 같이 대역 제한 윈도우 싱크(Band-limited Windowed Sinc) 필터 기반의 고품질 리샘플러 사용을 고려해야 합니다.

### C. 다중 전조 데이터 누수 원천 차단 (Data Leakage Protection)
* **위치**: `src/jam_transformer/dataset.py` L111
* **검토 사항**:
  - 오프라인 사전 전조 빌드를 최종 **기각(Rejected)**하고, **실시간 텐서 벡터화 전조 증강(Dynamic Augmentation)** 방식을 채택함으로써 데이터 누수(Data Leakage) 문제를 원천 차단했습니다.
  - 오프라인 전처리를 할 때 곡당 단 하나의 원본 파일(예: `pop909_001.pt`)만 생성되므로, 데이터셋 분할 시 곡 단위 해싱 문자열(`pop909_001`)이 유일합니다.
  - 따라서 해싱 분할(`_is_val`) 단계에서 동일한 곡의 다른 전조 버전이 학습(Train)과 검증(Val) 데이터셋으로 분산 쪼개져 성능이 과장되는 교차 오염(Cross-contamination) 위험이 **100% 구조적으로 불가능**합니다. 이는 아키텍처 다이어트가 가져온 매우 성공적인 사이드 이펙트입니다.

### D. Gradio WAV 오디오 정규화 로직 잠재 버그 (DataType Loss) ✅ 수정 완료
* **위치**: `app.py` L128 (`_to_f32` 함수)
* **문제**:
  - 기존 코드는 캐스팅 후 값 범위만 보고 `np.iinfo(np.int16).max`(32767)로 일괄 나눔.
  - **int32 (24/32-bit WAV)**: 32767로 나누면 최대 65536이 남아 클리핑 → 굉음/노이즈.
  - **uint8 (8-bit WAV)**: 0-255 범위를 32767로 나누면 볼륨 1/100 이하 → 무음.
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
  > `np.issubdtype(np.uint8, np.integer)` = True이므로, 기존 제안처럼 `np.integer` 분기 안에 uint8을 포함시키면 `/255`로 [0,1] 정규화가 되어버림. uint8을 먼저 명시적으로 처리한 후 signedinteger 분기를 타는 것이 올바른 순서.

### E. Windows 환경에서의 torch.compile 지연 및 불안정성
* **위치**: `configs/config.yaml` L63 (`compile: true`)
* **검토 사항**:
  - 설정 파일에는 기본적으로 `compile: true`가 적용되어 있습니다.
  - **문제점**: Windows 로컬 환경에서 PyTorch 컴파일(`torch.compile`)은 C++ 컴파일러(MSVC, MinGW 등)와의 호환성 문제로 컴파일을 시도하다 에러를 내며 실패하거나(eager mode 폴백), 최초 구동 시 3~5분 이상의 심각한 지연(Lag)을 초래합니다.
  - **대책**: Windows 로컬에서 디버깅 및 가벼운 실험 시에는 `python scripts/train.py --set model.compile=false` 또는 설정 오버라이드를 통해 컴파일 기능을 명시적으로 꺼주는 것이 시간 낭비와 렉을 방지하는 팁입니다.

---

## 3. ⚡ 성능 최적화 방안 (Performance Optimization)

추가적인 연산 가속 및 VRAM 절약을 위한 구체적인 최적화 제안입니다.

### A. Static KV cache 및 CUDAGraphs 결합
* **현재 문제**: 매 디코딩 단계마다 `torch.cat`으로 KV 캐시 텐서 크기를 확장하므로 GPU 할당 오버헤드와 메모리 단편화가 누적됩니다.
* **해결 방안**:
  1. `CausalSelfAttention` 내부에서 학습된 최대 시퀀스 길이 `(max_seq_len)` 크기의 고정 텐서를 미리 할당합니다.
  2. 매 스텝 생성된 Key/Value 벡터를 해당 버퍼의 특정 인덱스에 인플레이스(`in-place`) 덮어쓰기 형태로 채워 넣습니다.
  3. 이를 통해 메모리 복사 비용을 차단하며, 특히 `torch.compile(mode="reduce-overhead")`와 결합하여 CPU-GPU Launch Latency를 소멸시키고 추론 속도를 2배 이상 가속할 수 있습니다.

### B. In-place RoPE 캐시 슬라이딩
* **현재 문제**: 디코딩 도중 컨텍스트가 윈도우 크기(`ctx_window`)를 넘으면, 기존 KV 캐시를 완전히 파괴하고 최근 `ctx_keep` 토큰을 처음부터 다시 Forward 통과시켜 캐시를 재구축합니다 (Latency Spike 유발).
* **해결 방안**:
  * RoPE는 상대 위치 정보를 담고 있으므로, 윈도우 슬라이딩 시 기존 KV 캐시의 앞부분을 슬라이싱(`k[:, :, stride:]`)하고 뒤에 새 토큰을 갖다 붙이는 방식의 인플레이스 슬라이딩이 이론적으로 가능합니다.
  * 단, RoPE는 절대 인덱스 기준으로 pre-rotation이 적용되므로, 단순 자르기 시 위치 인코딩이 어긋납니다. 따라서 Rotary Embedding을 가할 때 캐시 슬라이딩 보정 계수(Rotational Offset Shift)를 적용하도록 Attention 모듈을 고도화하여 재연산 비용을 완전히 없앨 수 있습니다.

---

## 4. 💡 제안 사항 및 고려 사항 (Core Recommendations)

모델의 음악적 일반화 및 성능 지표 개선을 위한 제안입니다.

### A. `min_melody_coverage`의 최적값 튜닝
* **배경**: 이번에 도입된 Lakh/Slakh Preprocessing Coverage Filter (`min_melody_coverage: 0.20`)는 멜로디가 극히 비어 있는 곡들을 제외해 주는 핵심 장치입니다.
* **제안**: 너무 높은 커버리지는 오히려 다양한 인트로(Intro) 및 아웃트로(Outro) 무음 구간 학습을 방해할 수 있습니다. Lakh 데이터의 특성을 고려할 때 $0.15 \sim 0.25$ 구간에서 최적의 비율을 데이터셋 크기와 비례하여 그리드 서치할 것을 권장합니다.

### B. CFG Dropout (`condition_dropout_prob`) 활성화 레벨 제어
* **배경**: 멜로디가 없는 상황에서도 반주가 멈추지 않고 적절한 코드를 이어나가도록 돕기 위해 `condition_dropout_prob: 0.05`를 활성화하였습니다.
* **고려사항**: CFG(Classifier-Free Guidance) 효과를 극대화하기 위해서는 학습 시의 Dropout 확률(5%)과 더불어, 추론 시점(`pipeline.py`)에서 멜로디 조건 토큰을 완전히 빈 값(PAD)으로 채운 무조건부(Unconditional) 로짓과 조건부 로짓 간의 보간 연산(`guided_logits = uncond + scale * (cond - uncond)`)을 구현하는 것이 좋습니다. 이를 통해 반주의 창의성 및 멜로디 추종 강도를 유연하게 조절할 수 있습니다.

### C. 중간 저장(Checkpointing) 주기 및 Early Stopping 임계치 적절성 분석
* **위치**: `configs/config.yaml` L180 (`checkpoint_every_n_train_steps`), L187 (`early_stopping_patience`)
* **분석 및 문제점**:
  1. **체크포인트 저장 주기 (100 steps) 병목**: 
     - 본 모델(38M params)은 옵티마이저(AdamW) 상태를 포함하여 체크포인트 1회 저장 시 **약 400~450MB**의 디바이스 데이터를 디스크에 기록합니다.
     - 고성능 GPU(RTX 4090/A100) 기준 1 step은 30~50ms 내외이므로, 100 steps는 **단 3~5초**에 해당합니다. 
     - 3초마다 450MB의 대용량 쓰기가 반복되면 심각한 하드디스크 I/O 병목이 발생하여 실제 학습 스루풋(Throughput)이 5배 이상 저하됩니다.
     - **대책**: Spot 인스턴스 crash guard 목적이라 하더라도 저장 단계를 **`500` 또는 `1000` steps**로 변경하거나 로컬 디스크 속도가 보장되지 않으면 비활성화(`0`)할 것을 강력히 권장합니다.
  2. **Early Stopping 인내치 (10 epochs) 최적화 제안**:
     - 기존의 `patience: 15`는 다소 길어 학습 중단 반응이 느릴 수 있습니다. **`patience: 10`으로 단축 조정하는 것이 시간 및 GPU 과금 방지 측면에서 대단히 훌륭한 타협점**입니다.
     - 코사인 Warmup이 완료된 이후 10 에폭 동안 검증 손실이 개선되지 않는다면 모델이 완전히 정체(Plateau)했거나 오버피팅이 발생했다고 판단하기에 충분한 윈도우 크기입니다.
     - **대책**: `patience`를 `10`으로 낮출 경우, 학습 시작 초기(웜업 구간)에 조기 종료되는 오작동을 막기 위해 최소 유예 에폭 설정인 **`early_stopping_min_epochs` 역시 동일하게 `10`으로 맞추어 조율**해주어야 의도대로 최단 지점에서 자동 정지됩니다.
       * *실행 옵션*: `--set training.early_stopping_patience=10 --set training.early_stopping_min_epochs=10`

---

## 5. 📋 최종 학습 전 점검 리스트 (Training Checklist)

Paid GPU 서버에서 본격적인 대형 학습(Training Run)을 커밋하기 전에 반드시 점검해야 할 항목들입니다.

- [ ] **Dry Run 확인**: CLI 명령어 `python scripts/train.py --dry_run_steps 20`를 실행하여 1스텝당 소요 시간(ms/step)과 에폭당 예상 비용을 달러($)로 환산하여 예산 범위 내에 있는지 확인했는가?
- [ ] **VRAM Peak 검수**: Dry Run 출력 결과에서 Peak VRAM이 장비 용량(e.g., T4=16GB, RTX3090/4090=24GB)의 85% 이하로 통제되고 있는가?
- [ ] **지문 일치성(Fingerprint) 검사**: `_dataset_meta.json`의 해시 지문이 현재 `configs/config.yaml`의 토크나이저 설정과 일치하여 학습 직후 AssertionError로 터지지 않는 것을 검증했는가?
- [ ] **W&B API Key 설정**: `.env` 파일에 `WANDB_API_KEY`가 올바르게 세팅되어 실시간 학습 커브 모니터링이 준비되었는가?
- [ ] **체크포인트 경로 무결성**: Spot instance 중단 시 중단 단계부터 즉시 복구할 수 있도록 `training.checkpoint_dir` 경로가 로컬 디스크 및 외부 영구 스토리지에 안정적으로 매핑되었는가?
- [ ] **Gradio 데모 디바이스 확인**: `app.py` 구동 시 모델 및 오디오 믹싱 연산이 CPU가 아닌 CUDA 장치로 올바르게 로드되는가?
