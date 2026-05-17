# AI Jam Station: 프로젝트 분석 및 성능 개선 보고서 (v2)

본 보고서는 'Real-Time AI Jam Station' 프로젝트의 현재 학습 상태를 진단하고, 초기 모델에서 발견된 'Sparse(성긴 구성)' 및 'Monophonic(단선율)' 문제를 해결하기 위한 전략과 향후 고도화 방향을 정리한 문서입니다.

---

## 1. 현재 상태 및 학습 진단 (Current Status)

### 학습 지표 분석
*   **Early Stopping 발생**: 원래 200 Epoch를 목표로 했으나, Epoch 10 부근에서 `val_loss`의 하락세가 정체(Plateau)되어 조기 종료됨.
*   **성공적인 초기 수렴**: Plateau 발생 전까지 `val_loss`가 약 1.28까지 빠르게 하락하며 기본적인 음악 규칙은 성공적으로 학습함.
*   **모델 한계 직면**: 현재의 Baseline 구조(REMI 단일 시퀀스)로는 더 이상의 손실값 하락이 어려운 상태로 판단됨. 이는 'Sparse' 및 'Monophonic' 문제를 해결하기 위해 단순 학습 시간 연장이 아닌 **구조적 개선(Chord Tokens 등)**이 필수적임을 시사함.
*   **체크포인트 확보**: `pulled_checkpoints/` 내에 최적 시점(Epoch 10)의 모델 가중치가 저장되어 있어 즉시 활용 가능.

### 해결된 문제 (Reflected Changes)
*   **Monophonic 문제 완화**: `structural_suppression` (인퍼런스 시 시간 진행 토큰 억제) 로직 도입을 통해 화음 발생 비율(Polyphony Rate)이 기존 대비 약 **58% 개선**됨 (0.113 → 0.179).
*   **학습 편향 보정**: `polyphony_loss_boost`와 `WeightedRandomSampler`를 통해 모델이 화음 구조를 더 중요하게 학습하도록 유도 중.

---

## 2. 주요 기술적 개선 및 차별화 포인트

### [핵심 제안] Chord Tokens 도입 (High Priority)
기존의 CP-Word 제안을 대체하며, 모델의 화성학적 이해도를 근본적으로 높이기 위한 전략입니다.

*   **표현 수준**: **12개 Root + 9개 Quality** (Maj, Min, dim, aug, sus4, 7, maj7, min7, m7b5) 조합.
*   **어휘 규모**: 약 110여 개의 토큰 추가 (데이터 희소성과 표현력 사이의 최적 균형).
*   **기대 효과**:
    1.  **화성적 일관성**: 모델이 예측해야 할 음표의 범위를 해당 화성 내로 좁혀주어 'Sparse' 문제 해결.
    2.  **일반화 성능 향상**: 곡을 외우는 대신 음악적 규칙을 학습하게 하여 Plateau 현상 돌파 및 과적합 방지.

### 추가 기술 고도화 제안
1.  **구조적 토큰 (Form Tokens)**: `INTRO`, `VERSE`, `CHORUS` 등 곡의 진행 단계를 명시하여 파트별 다이내믹 차별화. (롱폼 생성 시 문맥 유지 및 송폼 인지에 필수)
2.  **데이터 규모 확장**: 현재 POP909(약 900곡)에 더해 **Lakh MIDI Dataset**의 Clean subset을 추가하여 모델의 기초 체급 강화.
3.  **표현 해상도 최적화**: 벨로시티(Velocity) 빈을 Log-scale로 재설계하여 인간적인 강약 표현 유도.
4.  **모델 업스케일링**: 지식 수용량 확대를 위해 `d_model` 및 `n_layers` 상향 조정 고려.

### 음악의 7대 요소 대응 전략
모델의 음악성 고도화를 위해 다음 요소들을 체계적으로 관리합니다.
*   **반영 완료**: 리듬(DUR/POS), 멜로디(PITCH-Input), 화성(PITCH-Output/Chord), 셈여림(VEL), 빠르기(TEMPO).
*   **고도화 예정**: 형식(Form - Form Tokens), 음색(Timbre - VST/Audio Rendering).

---

## 3. 향후 개발 로드맵 및 액션 아이템

### [단기] 우선순위 1: 데이터 및 구조 고도화
1.  **Chord Detection & Encoding**: `pretty_midi` 등을 활용한 마디별 코드 추출 및 토크나이저 통합.
2.  **Loss Boost & Sampling**: `polyphony_loss_boost` 상향 및 `polyphony_sample_weight_alpha` 조정을 통해 화음 중심 학습 강제.

### [중기] 우선순위 2: 데이터셋 및 모델 체급 확장
1.  **Lakh Dataset 연동**: 대규모 외부 데이터를 통한 일반화 성능 확보.
2.  **Model Scaling**: 정체기 돌파를 위한 파라미터 수 확장 실험.

### [장기] 우선순위 3: 시연 인터페이스 및 서비스화
1.  **Gradio 기반 `app.py` 구축**: 사용자 MIDI 업로드 및 실시간 오디오 렌더링(FluidSynth) 지원.
2.  **Human-in-the-loop**: 사용자가 UI에서 코드 토큰을 직접 수정하여 결과물을 조절하는 기능.

---
**최종 업데이트:** 2026-05-15
**주제:** 기계학습지능(기학지) 기말 프로젝트 고도화 전략
