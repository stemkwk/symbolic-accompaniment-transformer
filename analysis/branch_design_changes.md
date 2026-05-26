# 🔀 브랜치 설계 변경 비교 분석
## `main` → `feat/single-stream-accompaniment`

본 문서는 `feat/single-stream-accompaniment` 브랜치가 `main` 브랜치 대비 어떤 구조적 설계 결정을 바꿨는지, 그리고 그 이유와 트레이드오프를 기술합니다.

---

## 1. 트랙 구조: 3트랙 → 2트랙

### main 브랜치
```
tracks: ["melody", "bridge", "piano"]
```
- `melody`: 보컬 멜로디 (조건)
- `bridge`: 기타/현악 등 중간 음역 반주 (타깃 1)
- `piano`: 피아노 반주 (타깃 2)

### 현재 브랜치
```
tracks: ["melody", "accompaniment"]
```
- `melody`: 보컬 멜로디 (조건)
- `accompaniment`: 피아노 반주 (타깃, bridge + piano 병합)

### 변경 이유
- **복잡도 감소**: 두 타깃 트랙을 동시에 생성하면 모델이 두 트랙 간 조화까지 학습해야 해서 학습 난이도가 급격히 상승함.
- **데이터 일관성**: POP909 데이터셋은 사실상 멜로디 + 단일 반주 구조이며, bridge 트랙 레이블이 명확하지 않아 노이즈가 됨.
- **목표 단순화**: 논문 과제 범위에서 "멜로디 → 반주" 단일 태스크에 집중하는 것이 결과 해석과 평가 지표 설계 모두 명확해짐.

---

## 2. 시퀀스 포맷: SEP-분리형 → Temporal Interleaving

### main 브랜치 (SEP-분리형)
```
<BOS> KEY TEMPO
  TRACK_melody
    BAR [chord] POS_n CHROMA OCTAVE DUR VEL ...
    ...
  <SEP>
  TRACK_piano
    BAR [chord] POS_n CHROMA OCTAVE DUR VEL ...
    ...
<EOS>
```
멜로디 전체 블록과 피아노 전체 블록을 `<SEP>`으로 구분하여 순차 배치.

### 현재 브랜치 (Temporal Interleaving)
```
<BOS> KEY TEMPO
  BAR [chord]                              ← 공유 화음
  POS_n
    TRACK_melody        CHROMA OCTAVE DUR VEL   ← 조건
    TRACK_accompaniment CHROMA OCTAVE DUR VEL   ← 타깃
  POS_n+1 ...
<EOS>
```
각 시간 위치(POS)에서 멜로디 음표와 반주 음표를 인접하게 배치.

### 변경 이유

| 관점 | SEP-분리형 | Temporal Interleaving |
| :--- | :--- | :--- |
| **인과적 거리** | 멜로디 조건이 최대 ~1000토큰 앞에 위치 → 어텐션 거리가 멀어 약한 조건화 | 멜로디와 반주가 같은 시간 위치에서 인접 → 강한 직접 조건화 |
| **화성 학습** | 반주 전체를 한 블록으로 예측 → 마디 간 화성 일관성 학습이 어려움 | 위치 단위로 멜로디↔반주 쌍을 직접 대조 → 화성 대응 관계 명시적 학습 |
| **CFG 지원** | 멜로디 블록 전체를 PAD로 교체하면 자연스럽게 unconditional 생성 가능 | 위치마다 인터리빙되어 있어 "조건 없는 배포"와 "조건 있는 배포"를 분리하기 어려움 → CFG 추론 불가 |
| **시퀀스 효율** | 두 블록을 순차 나열 → 동일 화성 정보가 두 번 반복됨 | 화음 토큰은 BAR 단위로 1회만 발행 → 밀도 높은 정보 압축 |

### 트레이드오프 (포기한 것)
- **CFG 추론**: SEP 포맷에서 멜로디 블록 전체를 PAD로 교체하는 방식의 CFG가 자연스럽게 가능했으나, 인터리빙 포맷에서는 구현 불가.
- `condition_dropout_prob`은 유지되지만 추론 시 guidance weight `w > 1.0`은 미지원 (`cfg_w=0` 고정).

---

## 3. 다성부 제어 전략 역전

### main 브랜치
- `structural_suppression: 1.5` — 추론 시 VEL 토큰 직후 BAR/POS/TEMPO 토큰의 logit을 1.5 차감하여 시간 진행을 억제, 화음 적층을 강제.
- `polyphony_sample_weight_alpha: 0.5` — 폴리포니가 풍부한 청크를 더 자주 샘플링.

### 현재 브랜치
- `polyphony_loss_boost: 2.0` — 학습 시 화음 위치 토큰의 cross-entropy 손실에 2배 가중치 적용. 모델이 스스로 화음 쌓기를 학습.
- `structural_suppression: 0.0` — 비활성 (기본값). 모델이 이미 화음 생성을 학습했으므로 추론 시 보정 불필요.
- `polyphony_sample_weight_alpha: 0.0` — 비활성. `polyphony_loss_boost`와 중복되어 데이터 분포 왜곡만 초래하므로 제거.

### 변경 이유
- **근본 해결**: 추론 시 패널티는 모델이 배우지 못한 것을 강제하는 임시 방편. 학습 단계에서 loss를 통해 화음 생성 자체를 목표로 만드는 것이 일반화 측면에서 더 robust.
- **단일 레버**: 두 메커니즘이 동시에 작동하면 효과 귀속(Attribution)이 불분명해짐. loss_boost 하나로 제어하는 것이 평가와 ablation 모두 명확함.

---

## 4. 소스 균형 샘플링 도입

### main 브랜치
균등 샘플링 (자연 분포 그대로).
- 실제 분포: Lakh ≈ 90.3% / Slakh ≈ 4.7% / POP909 ≈ 4.9%

### 현재 브랜치
`WeightedRandomSampler`로 목표 분포 재조정.

| 소스 | 자연 비율 | 목표 비율 | 샘플링 가중치 |
| :--- | ---: | ---: | ---: |
| Lakh | 90.3% | 55% | 0.0716 (×0.61) |
| Slakh | 4.7% | 40% | 1.0000 (×8.5) |
| POP909 | 4.9% | 5% | 0.1199 (×1.0) |

### 변경 이유
- **Slakh 품질 우선**: Slakh2100은 전문 뮤지션이 편곡한 MIDI로 화음 복잡성과 정확성이 가장 높음. 자연 비율(4.7%)로는 학습에 미치는 영향이 미미함.
- **Lakh 다양성 보존**: Lakh는 수만 곡의 서양 팝/록/재즈를 커버하여 장르 다양성을 제공하지만 품질 편차가 큼. 비율을 낮춰도 절대량이 충분함.
- **POP909 장르 편향 억제**: POP909는 중국 팝으로 편향되어 있어 비율을 자연 수준으로 고정.

---

## 5. Train/Val 분할 방식 변경

### main 브랜치
스트라이드 기반 인덱스 분할 (청크 단위 80/20 분리).

**문제점**: 동일 곡의 청크들이 train과 val 양쪽에 포함될 수 있음 → 미묘한 데이터 누수(Data Leakage) 위험.

### 현재 브랜치
SHA-256 해시 기반 **곡 단위** 분할 (`val_ratio: 0.2`).
- 샤드 파일명의 SHA-256 해시값으로 결정론적으로 train/val 배정.
- 동일 곡의 모든 청크가 반드시 같은 split에 배정 → 교차 오염 원천 차단.
- 동일 config에서 재실행 시 항상 동일한 split 재현.

---

## 6. 기타 설정 변경 요약

| 항목 | main | 현재 | 이유 |
| :--- | :--- | :--- | :--- |
| `max_seq_len` | 2048 | 2560 | 12~13마디 커버로 코드 진행 학습 범위 확대 |
| `condition_dropout_prob` | 0.15 | 0.05 | 인터리빙 포맷에서 CFG 역할 소멸; 견고성 목적만 유지 |
| `checkpoint_every_n_train_steps` | 100 | 1000 | 100 steps = 약 3초 → 450MB 쓰기가 연속 발생, I/O 병목 |
| `early_stopping_patience` | 15 | 10 | 비용 절감; warmup 이후 10 epoch 정체면 수렴 판단 충분 |
| `min_melody_coverage` | 없음 | 0.20 | Lakh/Slakh의 희소 솔로 트랙이 멜로디로 잘못 탐지되는 케이스 필터링 |

---

**작성일:** 2026-05-26
