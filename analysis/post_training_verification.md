# ✅ 검증 현황 + 학습 후 검증 체크리스트

이번 세션의 변경(데이터 재전처리 / 학습 설정 A / 추론 제어 B)에 대한 검증 상태.
**메커니즘 검증은 지금 완료**했고, **음악적·행동적 검증은 학습된 체크포인트가 있어야** 가능하므로
아래에 따로 적어 둔다. 학습이 끝나면 이 체크리스트대로 수행할 것.

작성: 2026-05-31 · 브랜치 `feat/single-stream-accompaniment`

---

## ✅ 지금 완료한 메커니즘 검증 (랜덤/소형 모델 + 실데이터)

| 변경 | 검증 내용 | 결과 |
|---|---|---|
| condition dropout 전체 블록 PAD | 실데이터 강제 발동: 134블록 멜로디 529→0, 반주(mask=1) 무손상, SEP/BOS 유지 | ✅ |
| `condition_dropout_prob` 0.075 / 소스 가중치 | config 로드 + 실효 분포 = 55.0/40.0/5.0% (실제 chunk 수 기준) | ✅ |
| CFG (#1) — uncond 분기 | 모델 입력 가로채기: uncond row가 멜로디 전부 PAD(헤더 `[BOS,PAD,PAD]`, forced블록 26토큰 중 25 PAD), cond와 distinct | ✅ |
| CFG (#1) — 블렌딩 공식 | `blend(w=1)` == cond logits, 오차 1.49e-08 | ✅ |
| avoid penalty (#3) — 적용 | C major 강제: 18개 코드-활성 스텝에서 **오직 CHROMA F만 정확히 −penalty** 감소 | ✅ |
| avoid penalty (#3) — 테이블 | quality별 매핑 (C maj→F, G7→C, sus4→E, min→없음) | ✅ |
| 전체 회귀 | `pytest tests/` 31 passed, 1 skipped (변경마다 반복) | ✅ |

> 즉 "코드가 의도대로 동작하는가"(no-op 버그·잘못된 인덱스·desync 등)는 닫혔다.

---

## ⏳ 학습 후 검증 (체크포인트 필요 — 지금 불가)

랜덤 모델로는 "음악적으로 옳은가"를 잴 수 없다. 학습 완료 후 best checkpoint로 아래를 수행.

### 1. CFG 음악적 효과 (#1)
- [ ] 같은 멜로디로 `cfg_w` 0 / 1.5 / 3.0 생성 비교 → w↑일수록 반주가 멜로디를 **더 강하게 추종**하는지 (화성 정합↑, 무난한 일반 반주에서 멀어짐). 청취 + 멜로디-반주 화성 일치율 측정.
- [ ] **uncond sanity**: 멜로디를 PAD한 무조건부 생성이 garbage가 아니라 **그럴듯한 반주**인지 → dropout 학습이 제대로 됐는지 확인. (garbage면 `condition_dropout_prob`를 올리거나 dropout 재검토)
- [ ] w를 너무 키웠을 때(>3) 과포화/붕괴 임계 관찰 → app 슬라이더 권장 범위 확정.

### 2. avoid-note penalty 음악적 효과 (#3)
- [ ] `avoid_note_penalty` 0 / 3 / 6 생성 비교 → 출력에서 **sustained avoid note(예: maj 코드 위 11) 빈도 감소** 측정 (생성 MIDI에서 코드 대비 avoid 음정 비율 집계).
- [ ] 색채음/경과음(텐션 9·13, 반음계)이 **과도하게 사라지지 않는지** 확인 (soft penalty의 목적). 너무 밋밋해지면 값을 낮춤.
- [ ] 권장 기본값 확정 (현재 config 0.0=off, 슬라이더 권장 2–4).

### 3. 학습 설정 (A) 효과
- [ ] `condition_dropout_prob 0.075`: 모델이 멜로디를 **충분히 추종**하는지 (너무 높아 under-conditioned 아닌지). 조건부 생성 품질 확인.
- [ ] 소스 가중치 55/40/5: 학습 로그에서 소스별 샘플 비율이 의도대로인지 + **Slakh(1,355곡) 과적합** 징후 (소스별 train/val loss 격차) 모니터링.

### 4. 일반 학습-전/후 검증 (project_plan C)
- [ ] **빌린 16GB 박스에서 `train.py --dry_run_steps 50`** → 진짜 ms/step·peak VRAM·초기 loss(≈ln 173≈5.15)·`assert_data_matches_config`. (로컬은 4GB라 throughput/VRAM 무의미)
- [ ] inference end-to-end: POP909/실멜로디 → 반주 비어있지 않고 음악적으로 유효한지 청취.
- [ ] Colab T4 `colab_train_verify.ipynb` 실데이터 1ep 정상 종료.

---

## 참고: 검증 재현 방법
- 메커니즘 검증 스크립트는 일회성으로 작성·실행 후 삭제했다 (랜덤 소형 모델 + 입력/로짓 가로채기).
  재현 시: 멜로디 MIDI 생성 → `JamTransformerLightning`(d=64,L=2) → `DecoderTransformer._sample`/`model.forward`
  를 spy 또는 scripted-sample로 감싸 logits/입력 캡처 → CFG는 uncond PAD·blend(w=1)==cond, avoid는
  C-maj 강제 후 CHROMA F만 −penalty인지 확인.
