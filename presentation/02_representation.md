# 📑 [Slide 02] Data Representation: Key-Invariant Tokenization

## 1. 발표 자료 개요 (Slide Content)

### 🎼 피처 엔지니어링: 조-불변 상대적 하모닉 토크나이저 (Key-Invariant Tokenizer)
* **절대 좌표의 파편화**: 절대 음높이(Pitch) 값을 그대로 저장하는 대신, 곡의 전역 **조성(KEY)**을 기준으로 상대적인 화성 정보로 완전 변환하여 인코딩함.
  - $\text{Pitch} \to \text{Chroma (0~11)} + \text{Octave (Register)}$
  - $\text{Chroma} \to \text{Scale Degree (조성 내 상대적 음도, 0~11)}$
  - $\text{Chord} \to \text{Chord Quality (화음 성질, 12종) + Scale Degree}$
* **조-불변성(Key-Invariance) 증명**:
  - C Major의 '도-미-솔'과 F# Major의 '파#-라#-도#'은 모두 동일하게 조성 내 음도 `[SD_0, SD_4, SD_7]` 및 메이저 코드 `[SD_0, QUAL_0]` 토큰으로 완벽히 동일하게 인코딩됨.
  - **수학적 의의**: 절대적 조옮김에 상관없이 화성적 기능과 움직임이 추상화되어 결합되므로 모델이 조표(Key)를 학습해야 하는 학습 부담을 소멸시킴.

### ⏳ Lookahead를 위한 바-블록 인터리빙 (Bar-Block Interleaving)
* **실시간 반주 생성을 위한 포맷**: 곡을 $N$마디 단위의 시간 블록으로 쪼개어 다음과 같이 직렬화함.
  $$\text{[BAR 1..N Melody notes]} \to \text{SEP} \to \text{[BAR 1..N Accompaniment notes (chords + notes)]}$$
* **인과적 모델에서의 선독(Lookahead)**:
  - 디코더 전용 트랜스포머가 단방향(Causal) 어텐션을 사용함에도 불구하고, 멜로디 파트가 분리 기호(`SEP`)보다 먼저 주어지므로 모델은 **미래 N마디의 멜로디 흐름을 완전히 먼저 보고(Lookahead)** 그에 맞춘 최적의 반주 화성을 오토레그레시브하게 생성할 수 있음.

---

## 🎤 스피치 스크립트 (Speech Script)

> "두 번째로 피처 엔지니어링 및 데이터 표상 부문입니다. 저희 프로젝트의 가장 강력한 차별점은 바로 '조-불변 상대적 하모닉 토크나이저'입니다.
>
> 기존 모델들은 절대 음높이인 MIDI 숫자를 그대로 사용했지만, 저희는 곡의 Key를 추정해 모든 음높이를 조성 내 상대적 음도인 'Scale Degree'로 변환했습니다. 이렇게 하면 어떤 키로 전조를 하더라도 화성 구조가 완벽하게 일치하는 토큰 시퀀스로 통일됩니다. 이 피처 엔지니어링 덕분에 모델은 절대 음높이 학습을 건너뛰고 '음악의 상대적 화성학적 관계' 자체만을 온전히 학습하게 됩니다.
>
> 또한, 실시간으로 연주자가 칠 멜로디를 보고 반주를 만들어야 하기 때문에, $N$마디의 멜로디를 먼저 나열하고 `SEP` 토큰을 배치한 뒤 반주를 생성하는 '바-블록 인터리빙' 구조를 설계하여, Causal Transformer 구조 안에서도 미래 멜로디를 미리 내다보는 Lookahead 효과를 영리하게 달성했습니다."
