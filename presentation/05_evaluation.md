# 📑 [Slide 05] Evaluation Metrics & Results

## 1. 발표 자료 개요 (Slide Content)

### 📊 정량적 평가지표 (Objective Evaluation Metrics)
* **언어 모델링 지표 (Perplexity - PPL)**
  * 예측의 불확실성 지표. 학습 진행 시 PPL 수렴 경향을 분석하여 음악 문법 학습 수준 검증. (비교 타당성을 위해 Loss weighting이 배제된 Raw Cross-Entropy 기준으로 산출).
* **분포 수렴 지표 (Overlapping Area - OA)**
  * 실제 학습 데이터(Ground Truth)의 Pitch, Duration, Velocity 확률 분포와 모델이 생성한 데이터의 분포 간 중첩 면적 측정 (0~1).
* **화성적 일치성 (Pitch-Class Cosine Similarity)**
  * 멜로디의 화성 체계(조표, 코드 진행)를 반주가 얼마나 긴밀하게 추종하고 수용하는지 통계적으로 검증.

### 🎹 다성부 수렴 검증 (Polyphony Rate & Notes/Bar)
* **결과**: 기존 절대 인코딩 베이스라인 모델은 다성부 비율(Polyphony Rate)이 실제 데이터의 절반 수준에 미치지 못했으나, 본 상대 인코딩 및 Polyphony Loss Boost가 결합된 모델은 Ground Truth의 다성부 분포 형상에 고도로 수렴함을 실증함.

### 🎛️ 제어 경향성 실험 (Controllability Sweep)
* **실험 방법**: 추론 단계에서 로짓을 감쇠시키는 제어 파라미터(`structural_suppression`)를 0.0에서 2.0까지 조절하며 반주의 변화 관찰.
* **관찰 결과**: 제어 수치가 증가함에 따라 **Polyphony Rate(다성부 비율)가 결정론적(Deterministic)으로 하락**하는 선형 추이를 보임.
* **의의**: 본 모델이 무작위 생성을 넘어, 추론 시점 연구자의 매개변수 조작에 의해 화음의 두께를 부드럽게 제어할 수 있는 공학적 제어 가능성(Controllability)을 획득했음을 입증.

---

## 🎤 스피치 스크립트 (Speech Script)

> "네 번째로 정량적 평가 및 실험 결과 부분입니다. 저희는 모델의 성능을 단순히 '들어보니 좋다'는 주관적 평가를 넘어, 세 가지 머신러닝 지표로 객관화했습니다.
>
> 첫째, 다음 토큰 예측의 불확실성을 나타내는 Perplexity(PPL)가 안정적으로 수렴함을 확인했습니다. 둘째, 실제 데이터와 생성된 데이터의 음높이 및 길이 분포가 얼마나 유사한지 겹침 면적으로 측정하는 Overlapping Area(OA) 지표가 대폭 향상되었습니다.
>
> 특히 흥미로운 실험은 **'제어 경향성 스윕(Controllability Sweep)'**입니다. 저희 모델은 추가적인 재학습 없이, 추론 시점에 'Structural Suppression'이라는 가중치 변수 하나만 조작하는 것으로 반주의 다성부 비율(Polyphony Rate)을 선형적으로 제어할 수 있습니다. 그래프에서 보시는 것처럼 변수 제어에 따라 반주 화음의 두께가 점진적으로 얇아지거나 두터워지는 추세가 뚜렷이 나타나며, 이는 모델이 음악의 구조적 밀도를 머신러닝적으로 제어 가능한 형태로 파악하고 있음을 명확히 증명합니다."
