# 🗺️ 프로젝트 진행 플랜 — 데이터 전처리 → 학습

> **목표**: 멜로디 MIDI → 반주(accompaniment) MIDI 생성 Transformer.
> POP909 / Lakh / Slakh를 전처리해 `data/processed/`에 저장하고, 빌린
> **RTX 4070 Ti Super (16GB)** 에서 학습. 빌린 장비라 로컬/Colab(T4)에서 최대한 검증.
>
> **브랜치**: `feat/single-stream-accompaniment` · **최종 갱신**: 2026-05-31

---

## ✅ 완료 (이전 세션들, ~2026-05-28)

### Workstream B — 코드 수정 5건 (검증됨)
- `compare_inference.py` KEY anchor 추가 (메인 추론과 정합)
- `pipeline.py` checkpoint 정합성 가드 (missing/unexpected 경고 + vocab shape RuntimeError)
- `inference.py` 죽은 override locals 제거
- `config.yaml` / `tokenizer.py` stale 주석·vocab(173)·SEP emit 정정
- `tests/test_integration.py` 원복 (`assert total_notes >= len(melody)`)

### Workstream A — 멜로디 추출 진단
- `diagnose_melody_agreement.py` (핑거프린트 비교), `export_melody_comparison.py` (청취셋 WAV)
- **결과**: POP909 GT 대비 정확도 weight 93.7% / miner 98.4%.
  **Lakh/Slakh weight-vs-miner 일치율 ~45%** → GT 없는 두 데이터셋에서 weight 신뢰도 낮음
  → **miner 채택 근거 확보**

### 전처리 리팩토링 + 추출 고도화
- `prepare_data.py` → `src/jam_transformer/preprocessing/` 패키지로 분리
  (shards/chords/melody/pop909/lakh/slakh/synthetic)
- Lakh miner 통합 (`--melody_method` / `--miner_fallback` / `--mm_models`)
- Slakh 악기명(instrument) 추출: `metadata.yaml`의 `inst_class`(GT급 라벨)로 멜로디 스템 식별

---

## ✅ 완료 (이번 세션, 2026-05-31) — 커밋 `00c6a9e` (push 완료)

### #1 Slakh instrument-mode fallback 버그 수정
- `_encode_slakh_one`: melody-class 스템 없는 곡(~27%)을 버리지 않고
  `_lakh_track_events`로 **fallback** (설계·docstring 의도 복원)

### #2 Slakh fallback을 Lakh와 동일하게 (miner→weight→sparse)
- `prepare_data.py` single-thread 경로도 `--melody_method miner`면 miner 로드
  (parallel init과 일치) → instrument 모드 fallback이 **miner→weight→sparse 거름**
- stale docstring/help 갱신

### #3 Slakh redux MIDI-only 다운로드
- `download_slakh.py --redux`: 104GB FLAC tarball을 **스트리밍**하며 `*.mid`+`metadata.yaml`만
  추출(디스크 ~150MB), omitted(중복) 제외 → **dedup 1710곡**. traversal 방어 + 진행률

### #4 Colab 다운로드 노트북
- `notebooks/colab_download_slakh_redux.ipynb`: Colab 스트리밍 + **추출 끝나면 자동으로
  tar 묶어 gdrive 업로드** (무인 실행 가능)

### 검증 (전처리와 무충돌, 임시 디렉토리)
- **pytest 전체: 31 passed, 1 skipped**
- **Slakh fallback 실동작 (40곡)**: 인코딩 20→32곡 (**+60%**). no-melody-class 13곡 중
  12곡 miner/weight로 복구, 1곡만 sparse 제외 → 의도대로 동작 확인

### 📌 핵심 발견 (데이터 규모)
- 현재 보유 Slakh = **866곡(yourmt3-16k 서브셋)**, 풀 2100 아님
- 중복 제거 최대 = **Slakh2100-redux 1710곡** → val(270)/test(151)은 이미 완비, **train만 463/1289**
- redux 받으면 Slakh ≈ 2배 (866 → 1710), 품질 일관성 위해 miner 설정 권장

---

## ✅ 완료 (Lakh 재전처리, 2026-05-31 06:08)

- `--melody_method miner --miner_fallback weight --num_workers 2`, 소요 ~3.2h
- **saved 15,897 / skipped 1,359** (17,256 중 ~92%)
- 인덱스 현재: `{pop909: 909, slakh: 435, lakh: 15897}` = 총 17,241 shard
- ⚠️ 경고: midi-miner 모델 sklearn 1.3.0 pickle ↔ 환경 1.8.0 (`InconsistentVersionWarning`).
  진단 때와 동일 조건이라 일관성은 유지됨

## 🔄 진행 중

- **Slakh redux 다운로드** (Colab → gdrive, 자기 전 실행 예정, ~50분 @ 35MB/s)

---

## ⏳ 앞으로 할 일

### 1. 데이터 확정
- [x] Lakh 재전처리 완료 (15,897 saved, 인덱스 합쳐짐)
- [ ] Slakh redux tar를 gdrive에서 받아 `data/raw/slakh2100_redux`에 풀기
- [ ] **기존 버그 Slakh shard 435개 삭제** (`data/processed/slakh_*.pt`) — 새 경로는 해시가
  달라 fast-skip 안 되므로, 안 지우면 옛 버그본 + 새 shard가 섞임
- [ ] **Slakh 재전처리** (fixed 코드):
  `--slakh_dir data/raw/slakh2100_redux --slakh_melody instrument --melody_method miner --miner_fallback weight --num_workers 8`
- [ ] `_chunk_index.json` 정합성: `{pop909, slakh, lakh}` 3종 다 포함 확인 (재빌드됨)

### 2. 학습 전 검증 (빌린 GPU 가기 전)
- [ ] `train.py --dry_run_steps 50` → ms/step·peak VRAM·초기 loss(≈ln 173 ≈ 5.15) +
  `assert_data_matches_config` 통과
- [ ] inference end-to-end: POP909 1곡 → 반주 비었는지
- [ ] Colab T4 `colab_train_verify.ipynb` 실데이터 1ep 정상 종료

### 3. 빌린 GPU 전송 + 본 학습
- [ ] `package_for_server.py`로 `jam_data_processed.zip` 재번들 (데이터 완전히 바뀜)
- [ ] 빌린 박스에서 본 학습 (`--set model.compile=true`로 +20~30%)

### 4. 선택 / 후순위
- [ ] 추론 최적화: Static KV cache, In-place RoPE sliding (`report.md` 5B/5C 제안)
- [ ] `min_melody_coverage` 0.15~0.25 그리드 서치 (최종 데이터 기준)
- [ ] 문서 갱신 (`branch_design_changes.md`/`report.md`에 Slakh instrument·redux 반영)

---

## 📎 참고 명령

```bash
# Slakh redux 다운로드 (빌린 박스 등 빠른 회선)
python scripts/tools/download_slakh.py --redux --out_dir data/raw

# 재전처리 (Slakh, fixed 코드)
python scripts/prepare_data.py \
  --slakh_dir data/raw/slakh2100_redux --out_dir data/processed \
  --slakh_melody instrument --melody_method miner --miner_fallback weight --num_workers 8

# 학습 전 dry-run
python scripts/train.py --data_dir data/processed --dry_run_steps 50
```
