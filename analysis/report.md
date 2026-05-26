# 📊 Symbolic Jam Transformer: Comprehensive Project Analysis (v4)

본 보고서는 **Symbolic Jam Transformer** 프로젝트의 코드베이스와 설정을 분석하여 구조적 한계점, 논리적 버그, 그리고 학습/추론 성능을 향상시키기 위한 최적화 및 개선 적용 경과를 종합 정리한 문서입니다.

이 문서는 프로젝트 검수용 종합 보고서이며, 분야별 세부 분석 및 구체적인 교정 코드는 프로젝트 루트 폴더 `analysis/` 아래의 개별 파일로 분할하여 기록해 두었습니다. 다른 코드 편집 에이전트가 각 개별 파일을 전달받아 잔여 제안 사항 수정을 직접 수행할 수 있습니다.

---

## 🚦 전체 개선 및 최적화 진행 현황판 (Status Board)

| 번호 | 과제 분류 및 세부 항목 | 진행 상태 | 관련 보고서 및 링크 |
| :--- | :--- | :---: | :--- |
| **1** | **중대한 코드 논리 버그 (7개 항목)** | | [logical_bugs.md](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/archive/logical_bugs.md) |
| | A. 추론 시점의 인과성 누수 교정 | ✅ 완료 | |
| | B. Top-P 샘플링 경계선 누락 교정 | ✅ 완료 | |
| | C. Humanizer의 시간 단위 변환 오차 교정 | ✅ 완료 | |
| | D. LR 스케줄러 스텝 과소 계산 교정 | ✅ 완료 | |
| | E. 레거시 보폭 분할의 데이터셋 인덱스 mismatch 교정 | ✅ 완료 | |
| | F. Gradio 내 오디오 믹싱 데이터 타입 불일치 교정 | ✅ 완료 | |
| | G. Gradio 시연 시 오디오 합성 우선순위 교정 | ✅ 완료 | |
| **2** | **오프라인 데이터 전처리 성능 최적화** | | [offline_preprocessing_optimization.md](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/archive/offline_preprocessing_optimization.md) |
| | A. O(N log N) 스윕 라인 코드 추출 알고리즘 적용 | ✅ 완료 | |
| | B. 멜로디 저커버리지 필터 및 설정 파일 연동 | ✅ 완료 | |
| **3** | **프로젝트 디렉토리 구조 정리** | | [directory_reorganization.md](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/archive/directory_reorganization.md) |
| | A. Docker 설정 및 배포/에셋 번들 격리 정리 | ✅ 완료 | |
| **4** | **구조적 한계점 및 표현력 고도화** | | [structural_limitations.md](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/structural_limitations.md) |
| | A. 마디 중간 코드 세분화 (Sub-bar chord callback) | ✅ 완료 | |
| | B. 덧붙여진 복합 토큰 (Compound Token) 도입 제안 | ❌ 기각 | standard 디코더의 범용성 유지 목적 |
| | C. 단일 토큰 밀도 및 장기 기억 제약 | 📌 우회 | 슬라이딩/중복 생성 전략 채택 |
| **5** | **학습 및 추론 성능 병목 최적화** | | [performance_optimization.md](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/performance_optimization.md) |
| | A. 데이터 증강 시 Transposition 텐서 벡터화 적용 | ✅ 완료 | CPU 병목 및 GPU Starvation 소멸 |
| | B. Static KV 캐시 버퍼 사전 할당 도입 | 📌 제안 중 | 매 단계 `torch.cat` 오버헤드 방지 |
| | C. In-place RoPE 캐시 슬라이딩 윈도우 도입 | 📌 제안 중 | 컨텍스트 슬라이딩 시 재연산 생략 |
| **6** | **전조 증강의 오프라인 사전 빌드** | ❌ 기각 | [offline_pitch_augmentation.md](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/archive/offline_pitch_augmentation.md) |
| | *기각 사유: Transposition 텐서 벡터화 완료로 오프라인 빌드의 실익 상실 및 용량 비효율* | | |
| **7** | **학습 전 자가 진단 및 사전 디버깅** | ✅ 완료 | [pretraining_sanity_check.md](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/pretraining_sanity_check.md) |
| **8** | **종합 아키텍처 감사 및 검수 보고서** | 📌 제안 중 | [comprehensive_review.md](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/comprehensive_review.md) |

---

## 📂 세부 보고서 목차 및 링크

### 1. [🏗️ 구조적 한계 및 개선](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/structural_limitations.md)
* **상태**: 일부 완료 / 일부 기각
* **내용**: 마디 중간 코드 세분화(Sub-bar granularity) 적용 방법 및 복합 토큰 도입 기각 사유, 중복 슬라이딩 윈도우 생성 결정 수록.

### 2. [🐛 중대한 코드 논리 버그 (아카이브)](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/archive/logical_bugs.md)
* **상태**: ✅ 전체 완료 (보관됨)
* **내용**: 인과성 누수, Top-p 경계선, Humanizer BPM 단위 오차 등 모델 안정성과 시연 완성도를 무너뜨리던 7대 중대 버그들의 해결 내용 정리.

### 3. [⚡ 성능 병목 및 최적화 제안](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/performance_optimization.md)
* **상태**: 일부 완료 / 일부 제안 중
* **내용**: 벡터화 transposition을 통한 GPU Starvation 예방 성과와 dynamic KV cache 오버헤드 제거를 위한 static pre-allocation 및 in-place RoPE cache sliding 제안 수록.

### 4. [🗂️ 전조 증강의 오프라인 사전 빌드 대안 (아카이브)](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/archive/offline_pitch_augmentation.md)
* **상태**: ❌ 기각 (보관됨)
* **내용**: 오프라인으로 조옮김을 사전에 생성하는 방안에 대한 면밀한 트레이드오프 분석 및 최종 기각 결정 사유 수록.

### 5. [📁 프로젝트 디렉토리 구조 정리 (아카이브)](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/archive/directory_reorganization.md)
* **상태**: ✅ 전체 완료 (보관됨)
* **내용**: Docker 설정과 bundles 폴더 정돈 적용 결과 수록.

### 6. [⚡ 오프라인 데이터 전처리 성능 가속화 및 필터링 (아카이브)](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/archive/offline_preprocessing_optimization.md)
* **상태**: ✅ 전체 완료 (보관됨)
* **내용**: O(N log N) 스윕 라인 알고리즘 도입을 통한 Lakh/Slakh 전처리 600배 가속 성과 및 멜로디 저커버리지 필터링 적용 내용 수록.

### 7. [🛡️ 학습 전 오류 검증 및 자가 진단](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/pretraining_sanity_check.md)
* **상태**: ✅ 전체 완료
* **내용**: `fast_dev_run`, 초기 손실 수학적 진단, `inspect_data.py` 및 `dry_run_steps`를 활용한 사전 오류 검증과 Gradio 오디오 정규화 잠재 버그/Windows compile 지연 대책 수록.

### 8. [🕵️ 종합 아키텍처 감사 및 검수 보고서](file:///c:/Users/hojun/Documents/대학교 자료/3학년 1학기(2026-1)/기학지/과제/project_transformer/analysis/comprehensive_review.md)
* **상태**: 📌 제안 중 (일부 완료 / 일부 제안)
* **내용**: 프로젝트 아키텍처(REMI, RoPE, 양방향성)의 구조적 한계와, SDPA/오디오 샘플레이트/WAV 정규화/Windows compile 등의 버그 및 Static KV/RoPE 캐시 슬라이딩 등의 성능 최적화 제안을 종합 수록.

---
**최종 업데이트:** 2026-05-26  
**검수자:** Antigravity (AI Auditor)  
**프로젝트:** Symbolic Jam Transformer (기학지 기말 프로젝트 검수)
