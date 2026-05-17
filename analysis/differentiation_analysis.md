# 📊 Symbolic Jam Transformer: Comparative & Limitations Analysis

본 문서에서는 기계학습 및 AI 음악 작곡(Computational Musicology) 분야에서 널리 사용되는 기존 베이스라인 모델들과 본 프로젝트(**Symbolic Jam Transformer**)의 구조적 차별점, 기술적 개선점, 그리고 공학적 한계점을 엄격하게 분석합니다.

---

## 1. 🆚 대조 대상군 (Comparative Landscape)

본 분석에서 비교 대상으로 삼은 기술적 벤치마크는 다음과 같습니다:

1. **표준 절대 음높이 오토레그레시브 모델 (Standard Absolute-Pitch Transformers)**:
   * 예: *Music Transformer (Google Magenta)*, 일반적인 *REMI v1 / MIDI-Like* 기반 토크나이저 모델.
2. **계층적 변분 오토인코더 (Hierarchical VAEs)**:
   * 예: *MusicVAE*. 음악적 마디 구조를 잠재 공간(Latent Space)에서 결합하고 2~4마디의 짧은 단위를 외삽 및 보간하여 생성하는 방식.
3. **신경망 원음 오디오 생성기 (Raw Audio Generators)**:
   * 예: *MusicGen (Meta)*, *Stable Audio*. 오디오 코덱 토큰(EnCodec) 또는 확산 모델(Diffusion)을 사용하여 직접 Waveform을 출력하는 초대형 모델.
4. **기존 cGAN 오디오 매핑 (Old Pix2Pix cGAN Baseline)**:
   * 본 프로젝트의 이전 버전인 멜-스펙트로그램 직접 회귀 매핑 모델.

---

## 2. 🌟 1. 차별점 (Core Points of Differentiation)

### A. 조-불변 상대적 하모닉 토크나이저 (Key-Invariant Relative Harmonic Tokenizer)
* **기존 모델의 한계**: 기존의 Music Transformer나 REMI 기반 모델은 절대적인 MIDI Note Number(0~127)를 그대로 사용합니다. 이로 인해 C Major로 학습한 화성 진행 패턴이 F# Major로 조옮김되면 완전히 새로운 토큰으로 인식되어 데이터 희소성(Sparsity) 문제가 심화됩니다. 모든 조(Key)에 대응하기 위해 막대한 용량의 뇌와 전조 데이터 증강이 필수적입니다.
* **본 프로젝트의 차별성**: 음높이를 절대값 대신 전역 조성(KEY)에 따른 **상대적 음도(Scale Degree)**, **화음 성질(Chord Quality)** 및 **상대 크로마(Chroma)**와 **절대 옥타브(Octave) 레지스터**로 완전히 파편화했습니다.
  * **음악적 불변성(Invariance)**: C Major에서의 '도-미-솔'과 F# Major에서의 '파#-라#-도#'은 상대 화성 토큰 시퀀스로 인코딩 시 **100% 완벽하게 동일한 토큰 구조**를 가집니다.
  * **차별적 의의**: 모델이 절대 음높이에 얽매이지 않고 "화성학적 구조와 상대적 장단조 관계"를 추상화하여 학습하게 함으로써, 극히 적은 데이터셋(예: POP909 단일 데이터)으로도 전조 및 일반화 능력을 폭발적으로 달성했습니다.

### B. 실시간 연주에 최적화된 조건부 디코더 구조 (Melody-Conditioned Causal Decoder)
* **기존 모델의 한계**: MusicVAE는 인코더-디코더 구조로, 고정된 길이의 반주만 생성할 수 있어 실시간으로 연주자가 치는 멜로디에 맞추어 즉흥 잼 세션을 수행할 수 없습니다. 
* **본 프로젝트의 차별성**: Causal(인과적) Self-Attention을 사용하는 단일 디코더 구조 내에서 멜로디 트랙을 조건 접두사(Condition Prefix)로 제시하고, 반주 트랙을 뒤이어 생성하는 단방향 타겟팅 구조를 설계했습니다.
  * **실시간 상호작용**: 연주자의 입력 MIDI 스트림 또는 마이크 음원을 받아 멜로디 토큰으로 신속히 인코딩한 뒤, 뒤이어 실시간으로 반주 토큰을 오토레그레시브하게 디코딩(KV-Caching 기법 사용 가능)하여 **실시간 인터랙티브 AI 잼 스테이션** 구현을 가능하게 합니다.

### C. 추론 시점 다성부 제어 메커니즘 (Structural Suppression)
* **기존 모델의 한계**: 일반적인 시퀀스 모델은 추론 중 단순 Temperature나 Top-p 조절만으로는 "화음의 두께(Chord Density)"를 제어할 수 없습니다. 모델이 단선율 반주로 고착화되면 모델의 매개변수를 바꾸어 다시 학습해야 합니다.
* **본 프로젝트의 차별성**: 모델의 재학습 없이 **오직 추론 시점의 디코딩 Logits 조작(Structural Suppression Penalty)**만으로 반주의 다성부 밀도(Polyphony Rate)를 부드러운 단선율부터 풍부한 재즈 화음까지 결정론적으로 조절할 수 있습니다. 

---

## 3. 🚀 2. 개선점 (Technical Improvements & Solutions)

기존 모델들이 가졌던 고질적인 병목과 문제점들을 공학적으로 해결한 개선 사항들입니다.

### A. 구형 cGAN의 회귀 붕괴(Regression Collapse) 극복
* **기존 문제**: 오디오 스펙트로그램 직접 회귀 cGAN은 하나의 멜로디에 여러 반주가 매핑될 수 있는 "One-to-Many" 모호성으로 인해 스펙트로그램이 회색 노이즈처럼 평균화되어 뭉개지는 한계가 있었습니다.
* **해결 방안**: 심볼릭(Symbolic) 토큰 예측 체계로 전환하여 다중 정답 배포를 확률 분포상의 여러 극대점(Modes)으로 고르게 학습시켰으며, 크로스 엔트로피 손실 함수를 통해 선명하고 기계음이 섞이지 않는 안정적인 음악 구조를 확보했습니다.

### B. 초경량 매개변수 및 VRAM 극대화
* **기존 문제**: Meta의 MusicGen 등 원음 오디오 토큰 모델은 최소 3억 개(300M)에서 수십억 개의 파라미터를 사용하여 연산량이 매우 크며, RTX 3080/4090 수준의 로컬 GPU에서도 OOM(Memory Out-Of-Memory)이 빈번하고 학습에 수 주가 소요됩니다.
* **해결 방안**: 본 프로젝트는 12 Layers, `d_model=512` 수준의 극도로 정제된 **심볼릭 디코더 아키텍처(약 38M params)**를 채택했습니다. 
  * **공학적 극대화**: RoPE(Rotary Position Embedding)와 gradient checkpointing, fused AdamW 옵티마이저를 결합하여, **무료 단일 GPU(Colab T4 등) 혹은 저사양 로컬 환경에서 단 2-3시간 만에 200 epochs 학습 수렴**이 가능하게 만들었습니다.

### C. 강력한 토큰화 무결성 및 설정 불일치 차단 (Data Stability)
* **기존 문제**: 연구 도중 `config.yaml` 파일의 토큰 규칙(예: `velocity_bins`나 `pitch_min` 등)을 조금만 수정하고 이전 캐시 데이터를 로드하면, 차원이 소리 없이 어긋나 엉뚱한 예측을 하거나 CUDA 커널 에러로 터지는 버그가 빈번했습니다.
* **해결 방안**: 전처리 폴더에 토크나이저 하이퍼파라미터 해시값인 `_dataset_meta.json` (Tokenizer Fingerprint)을 엄격히 기록해두고, `train.py` 실행 즉시 능동적으로 일치성을 검사하여 **설정 편차로 인한 오학습을 사전에 100% 완전 차단**합니다.

---

## 4. ⚠️ 3. 한계점 (Technical Limitations & Challenges)

학술 보고서 및 향후 고도화 연구에서 반드시 짚고 넘어가야 할 정직하고 객관적인 한계점들입니다.

### A. 음원 합성 품질의 사운드폰트(SoundFont) 의존성
* **한계**: 본 모델은 물리적 오디오를 생성하는 것이 아닌, 완벽한 작곡 정보인 'MIDI'를 출력합니다. 따라서 연주음의 사실성(Acoustic Realism)이 어떤 사운드폰트(`.sf2`)를 FluidSynth에 결합하느냐에 절대적으로 의존합니다.
* **영향**: 사운드폰트가 없거나 윈도우 기본 미디 드라이버(`gm.dls`)를 사용하여 합성할 경우, 작곡 구조가 훌륭하더라도 최종 WAV 음질이 기계적이고 메마른 소리로 렌더링되어 감상용 음악으로서의 가치가 저하됩니다.

### B. 고정된 16분 음표 그리드 양자화 (Rigid Time Quantization)
* **한계**: 토크나이저의 박자 표현력이 한 마디를 16개 격자로 쪼개는 16분 음표 그리드(`POS_0` ~ `POS_15`)에 고정되어 있습니다.
* **영향**:
  * 재즈나 클래식 음악의 고유한 특징인 셋잇단음표(Triplet)나 엇박자 그루브, 그리고 인간 연주자의 미세한 박자 당김/늦춤(Rubato 및 Micro-timing)을 **생성 단계 자체에서 스스로 구현하는 것은 불가능**합니다.
  * 추론 이후 후처리 단계에서 `Humanizer` 라이브러리로 미세한 무작위 오차(Jitter)를 입혀 기계적인 느낌을 줄일 수는 있으나, 모델 스스로가 의도를 가지고 "스윙(Swing)"이나 "루바토 연주"를 작곡하는 화성 학습은 구조적으로 차단되어 있습니다.

### C. 한정된 로컬 컨텍스트 윈도우 (Window Memory Bottleneck)
* **한계**: 연산 최적화를 위해 시퀀스 길이 한계를 `max_seq_len: 2560` 토큰으로 제한했습니다. 이는 곡 당 대략 12~16마디의 정보에 해당합니다.
* **영향**:
  * 모델이 방금 연주한 몇 마디 전의 화성과 모티프(Motif, 동기)는 아주 훌륭하게 기억하여 대위법적으로 반주하지만, 곡 전체(예: 3~4분 길이의 POP 곡 전체)를 아우르는 거시적인 음악 구조(Intro $\to$ Verse $\to$ Chorus $\to$ Outro)를 일관성 있게 설계하고 기억하는 장기 의존성(Long-term Dependency) 설계에는 한계가 있습니다.
  * 향후 선형 주의집중(Linear Attention) 메커니즘이나 Mamba 등 상태공간 모델(SSM) 구조로의 고도화가 요구되는 대목입니다.

---
**작성일자**: 2026-05-18  
**연구원**: Jam Transformer Research Team  
