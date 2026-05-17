# AI Jam Station: 고품질 오디오 렌더링 및 후처리 전략 보고서

본 보고서는 AI가 생성한 심볼릭 데이터(MIDI)를 청각적으로 완성도 높은 결과물(WAV)로 변환하기 위한 시스템 고도화 전략을 담고 있습니다. 이는 모델의 학습 성능과는 독립적인 과정으로, 시연 시의 몰입감과 프로젝트의 전반적인 품질을 결정짓는 핵심 요소입니다.

---

## 1. 현황 및 문제점 (Current Limitations)
*   **사운드 엔진**: 현재 저용량 SoundFont 기반의 FluidSynth를 사용하여 결과물이 다소 기계적이고 건조함.
*   **표현력 부족**: 벨로시티(Velocity)에 따른 음색 변화가 단조롭고, 공간감(Reverb)이나 악기 고유의 아티큘레이션(Articulation)이 반영되지 않음.

---

## 2. 단계별 오디오 품질 향상 전략 (Progressive Strategy)

### [Phase 1] 고품질 샘플 기반 렌더링 (Low Effort)
*   **핵심 아이디어**: 기존 FluidSynth 구조를 유지하되, 악기 뱅크(`.sf2`)를 전문가급 샘플로 교체.
*   **추천 자원**: Salamander Grand Piano (16개 벨로시티 레이어 지원) 등 대용량 사운드폰트.
*   **장점**: 코드 수정 없이 설정 변경만으로 즉각적인 음질 개선 가능.

### [Phase 2] Python 기반 오디오 후처리 (Mid Effort)
*   **핵심 도구**: **`Pedalboard`** (Spotify 개발 라이브러리)
*   **구현 내용**:
    1.  **Reverb & Delay**: 스테레오 공간감을 추가하여 인위적인 건조함 제거.
    2.  **Compressor & EQ**: 소리의 밀도를 높이고 저역/고역의 밸런스를 조정하여 '음반' 같은 사운드 구현.
    3.  **Limiter**: 볼륨의 피크를 잡아주어 안정적인 출력 보장.

### [Phase 3] VST 호스팅 및 Humanization (High Effort)
*   **핵심 도구**: **`Dawdreamer`** (Python VST Host)
*   **구현 내용**:
    1.  **VST 연동**: 실제 상용 피아노 VST(예: Kontakt, Addictive Keys 등)를 Python 코드에서 직접 구동.
    2.  **ADSR 제어**: MIDI 메시지를 넘어서는 미세한 ADSR(Envelope) 및 페달링 제어.
    3.  **Humanizer**: 렌더링 직전 미세한 타이밍 오차와 벨로시티 노이즈를 추가하여 인간적인 연주 느낌 부여.

---

## 3. 권장 기술 스택 및 파이프라인

```mermaid
graph LR
    A[Generated MIDI] --> B[Humanizer Logic]
    B --> C[FluidSynth / Dawdreamer]
    C --> D[Raw WAV]
    D --> E[Pedalboard FX Pipeline]
    E --> F[Final Audio Artifact]
```

*   **Humanizer**: `mido` 또는 `pretty_midi` 활용
*   **Synthesis**: `pyfluidsynth` (High-end SF2) 또는 `dawdreamer` (VST)
*   **Post-Process**: `pedalboard` (Reverb, EQ, Limiter)

---

## 4. 기대 효과
*   **시연 완성도**: AI의 논리적 우수성(MIDI 구조)을 감성적 완성도(고품질 사운드)로 뒷받침.
*   **평가 우위**: 단순히 "돌아가는 모델"을 넘어, "실제 사용 가능한 음악 도구"로서의 비전을 제시.

---
**작성일:** 2026-05-15
**주제:** 오디오 렌더링 및 사운드 디자인 고도화 전략
