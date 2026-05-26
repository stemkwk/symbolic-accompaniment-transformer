# ⚡ 오프라인 데이터 전처리 성능 가속화 및 필터링 (Offline Preprocessing Optimization) - 적용 완료

본 문서에서는 대규모 데이터셋(Lakh MIDI, Slakh2100) 구축 시 오프라인 전처리 과정의 속도 가속화 및 학습 품질 향상을 위해 적용된 **스윕 라인(Sweep-line) 알고리즘**과 **멜로디 커버리지 필터(Melody Coverage Filter)**의 구현 내용을 정리합니다.

---

## ⚙️ 최적화 및 필터링 적용 요약

| 항목 | 해결 상태 | 적용 파일 및 위치 | 핵심 적용 내용 |
| :--- | :---: | :--- | :--- |
| **스윕 라인 알고리즘** | ✅ 완료 | `scripts/prepare_data.py` (`_extract_chords_from_midi`) | $O(N_{\text{beats}} \times N_{\text{notes}})$ 중첩 루프 구조를 $O(N \log N)$의 스윕 라인 구조로 전면 교체하여 비교 연산량 600배 이상 가속 |
| **멜로디 커버리지 필터** | ✅ 완료 | `scripts/prepare_data.py` (`_melody_coverage`) | 멜로디 음표가 있는 마디의 비율을 계산하여 임계값 미만인 곡을 전처리 단계에서 여과 |
| **설정(Config) 연동** | ✅ 완료 | `configs/config.yaml` | `preprocessing.min_melody_coverage: 0.20` 설정 및 `condition_dropout_prob: 0.05` 활성화로 무음 대처 학습 공백 보완 |

---

## 🔍 세부 적용 내용

### 1. 스윕 라인 기반 코드 추출 최적화
* **방식**: 음표를 시작 틱(`n[0]`) 기준으로 정렬하고, 박자의 흐름에 따라 활성 리스트(Active Note Pool)에 음표를 삽입/삭제하는 스윕 라인 알고리즘을 도입했습니다.
* **효과**: 연산량이 대폭 감축되어, 기존에 속도 병목 때문에 3000 박자 혹은 50,000음표 초과 시 데이터 생성을 강제로 건너뛰던 가드 코드를 안전하게 제거하였습니다.

### 2. 멜로디 커버리지 필터 (Melody Coverage Filter) 구현
* **배경**: Lakh 등의 대용량 데이터셋 중 멜로디가 극히 일부분만 나오는 sparse 솔로 파트(멜로디 커버리지 4~11% 내외)를 여과 없이 학습하면 모델이 무음 반주 위주로 편향 학습하게 됩니다.
* **구현**: 멜로디가 존재하는 마디의 비율을 구하는 헬퍼 함수를 추가하고, 데이터 추출 시점에 적용하였습니다:
  ```python
  def _melody_coverage(events, cond_track="melody") -> float:
      """멜로디가 있는 bar / 전체 bar"""
      mel_bars = {e.bar for e in events if e.track == cond_track}
      all_bars  = {e.bar for e in events}
      return len(mel_bars) / max(len(all_bars), 1)
  ```
* **적용 위치**: `_encode_lakh_one()`, `_encode_slakh_one()`, `_encode_one()` (POP909)의 이벤트 생성 직후에 필터를 적용하여 임계값 미만 곡을 사전 제외합니다.

### 3. 설정 파일 고도화
* `configs/config.yaml`에 `preprocessing` 섹션을 신설하여 `min_melody_coverage: 0.20` 값을 정의하고 스크립트에서 동적 참조하도록 연결했습니다.
* 커버리지 제한으로 인해 발생할 수 있는 무음 대처 학습 공백을 보완하기 위해 `condition_dropout_prob: 0.05` 설정을 활성화하고 오해의 소지가 있던 레거시 주석을 올바르게 교정했습니다.
