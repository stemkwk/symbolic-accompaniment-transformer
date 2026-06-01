# 추론(반주 생성) 가이드 — 시연용 웹 데모 실행

> **대상 독자**: 학습을 마쳤거나, 소유자에게서 체크포인트를 받아 **반주 생성을 직접 시연**해 보려는 분
> **무엇을 하나요**: 멜로디 MIDI 한 개를 올리면, 웹 화면(`app.py` Gradio 데모)에서 AI가 **피아노 반주 WAV**를 만들어 줍니다
> **권장 GPU**: 학습과 동일. NVIDIA GPU면 충분(추론은 학습보다 가볍습니다). GPU 없이 CPU로도 느리지만 동작합니다
> **OS**: Windows(PowerShell 기준) 또는 Linux

학습 가이드(`TRAINING_GUIDE.md`)를 따라 환경을 이미 만들어 두었다면, **Docker 이미지·`docker compose`가 그대로 재사용**됩니다. 추가 설치는 거의 없습니다.

---

## 🗺️ 한눈에 보기 — 전체 흐름

```
0. (학습을 안 거쳤다면) 저장소 클론 + Docker 빌드           ── TRAINING_GUIDE 0~4단계와 동일
1. 체크포인트 준비   (방금 학습한 결과 or jam_checkpoints.zip)
2. 음색 파일 준비    (jam_soundfonts.zip → soundfonts/)
3. 멜로디 MIDI 한 개 준비
─────────────────────────────────────────────────────────────
4. 웹 데모 실행 (명령 한 줄)
5. 브라우저에서 http://localhost:7860 접속
6. 멜로디 업로드 → "반주 생성" 클릭 → 반주 WAV 듣기/다운로드
```

> ⏱️ 첫 실행은 데모용 라이브러리를 컨테이너 안에 설치하느라 **1~2분** 걸립니다. 이후 모델 로딩까지 합쳐도 보통 1분 안에 화면이 뜹니다.

---

## ⚠️ 시작 전 필독

- **모든 명령은 Windows PowerShell**(Linux는 터미널)에서, **저장소 폴더 안**(`symbolic-accompaniment-transformer/`)에서 입력합니다.
- `docker compose run ...` 실행 중 프롬프트가 `root@xxxx:/app#` 으로 바뀌면 컨테이너 **안**에 들어간 것입니다. 거기서는 `docker` 명령이 안 됩니다. `exit` 로 나오세요.
- 🚫 **`configs/config.yaml`은 수정하지 마세요.** (학습 때와 동일한 이유 — 설정 지문이 어긋나면 모델 로드가 거부될 수 있습니다.)

---

## 0. (학습을 안 거쳤다면) 환경 준비

이미 `TRAINING_GUIDE.md`로 학습을 돌려 본 분은 **이 단계를 건너뛰세요.** 이미지가 이미 빌드돼 있습니다.

처음이라면 `TRAINING_GUIDE.md`의 **0~4단계**(소프트웨어 설치 · 클론 · Docker 빌드 · GPU 인식 확인)만 먼저 마치고 돌아오세요. 학습 데이터(`jam_data_processed.zip`)는 추론에는 **필요 없습니다.**

```powershell
git clone -b feat/single-stream-accompaniment https://github.com/stemkwk/symbolic-accompaniment-transformer.git
cd symbolic-accompaniment-transformer
docker compose build      # 이미 했다면 생략
```

> ⚠️ `-b feat/single-stream-accompaniment` 를 빼지 마세요. 모델 코드가 이 브랜치 기준입니다.

---

## 1. 체크포인트 준비 (`checkpoints/` 폴더)

데모는 학습된 모델 가중치(`.ckpt`) 파일이 있어야 동작합니다. 두 경우 중 하나입니다.

**(A) 방금 직접 학습한 경우** — 이미 `checkpoints/` 안에 파일이 들어 있습니다. 그대로 사용하면 됩니다. 추론에는 보통 `best-epoch=...-val_loss=....ckpt`(val loss가 가장 낮은 것)를 씁니다.

**(B) 소유자에게서 받은 경우** — 받은 `jam_checkpoints.zip` 을 풀어 `checkpoints/` 에 넣습니다.

```powershell
# Windows PowerShell
Expand-Archive -Path jam_checkpoints.zip -DestinationPath checkpoints -Force
```
```bash
# Linux
unzip -o jam_checkpoints.zip -d checkpoints
```

확인:
```powershell
ls checkpoints
```
`best-epoch=...-val_loss=....ckpt` (또는 `last.ckpt`) 같은 파일이 보이면 됩니다.

---

## 2. 음색(SoundFont) 준비 (`soundfonts/` 폴더)

모델은 음표(MIDI)를 만들고, 그 음표를 **피아노 소리(WAV)로 렌더링**할 때 SoundFont 음색이 필요합니다. 소유자에게서 받은 `jam_soundfonts.zip` 을 풀어 `soundfonts/` 에 넣습니다.

```powershell
# Windows PowerShell
Expand-Archive -Path jam_soundfonts.zip -DestinationPath soundfonts -Force
```
```bash
# Linux
unzip -o jam_soundfonts.zip -d soundfonts
```

확인 — `soundfonts/` 안에 `.sf2` 파일이 하나라도 있으면 됩니다:
```powershell
ls soundfonts
```
> `GeneralUser.sf2`, `FluidR3_GM.sf2` 같은 파일이면 OK. 데모가 `soundfonts/` 안의 `.sf2`를 **자동으로 찾습니다.**

---

## 3. 멜로디 MIDI 한 개 준비

반주를 입혀 줄 **단선율(멜로디) MIDI 파일**(`.mid`)이 하나 필요합니다.

- 가지고 있는 멜로디 `.mid` 파일을 쓰거나,
- [MuseScore](https://musescore.org) 등에서 간단한 멜로디를 그려 `.mid` 로 내보내거나,
- 인터넷의 무료 멜로디 MIDI(동요·민요 등 단선율)를 받으면 됩니다.

> 💡 화음·드럼이 잔뜩 든 복잡한 MIDI보다, **멜로디 한 줄짜리** 파일이 시연 결과가 깔끔합니다.

준비한 파일은 어디에 둬도 되지만, 업로드가 편하도록 저장소 폴더 근처에 두는 걸 권장합니다.

---

## 4. 웹 데모 실행 (명령 한 줄)

아래 명령을 **그대로 복사**해 실행합니다. (`<체크포인트파일>` 부분만 1단계에서 확인한 실제 파일명으로 바꾸세요.)

**Windows PowerShell**
```powershell
docker compose run --rm --service-ports -e WANDB_DISABLED=true train `
  bash -c "pip install -q gradio pyfluidsynth soundfile pedalboard librosa; python app.py --checkpoint 'checkpoints/<체크포인트파일>' --host 0.0.0.0 --port 7860"
```

**Linux**
```bash
docker compose run --rm --service-ports -e WANDB_DISABLED=true train \
  bash -c "pip install -q gradio pyfluidsynth soundfile pedalboard librosa; python app.py --checkpoint 'checkpoints/<체크포인트파일>' --host 0.0.0.0 --port 7860"
```

예시 (파일명이 `best-epoch=042-val_loss=2.1903.ckpt` 인 경우):
```powershell
docker compose run --rm --service-ports -e WANDB_DISABLED=true train `
  bash -c "pip install -q gradio pyfluidsynth soundfile pedalboard librosa; python app.py --checkpoint 'checkpoints/best-epoch=042-val_loss=2.1903.ckpt' --host 0.0.0.0 --port 7860"
```

> **명령 한 줄 풀이**
> - `--service-ports` : 7860 포트를 바깥(브라우저)으로 열어 줍니다. **빼면 화면에 접속할 수 없습니다.**
> - `pip install -q gradio ...` : 데모/렌더링 전용 라이브러리를 컨테이너 안에 설치합니다(학습 이미지에는 없음). 첫 실행에만 1~2분.
> - `--checkpoint ...` : 시작과 동시에 그 모델을 메모리에 올립니다.

아래 메시지가 보이면 **성공**입니다:
```
Model ready on cuda.
* Running on local URL:  http://0.0.0.0:7860
```
> `theme ... Gradio 6.0` 경고는 무시해도 됩니다. (GPU가 없으면 `cuda` 대신 `cpu` 로 뜨며, 그래도 동작합니다.)

---

## 5. 브라우저에서 접속

웹 브라우저 주소창에 입력:

```
http://localhost:7860
```

`🎵 JAM Transformer` 화면이 뜨면 됩니다.

---

## 6. 반주 생성하기

### ① 체크포인트 확인 (화면 맨 위)
4단계에서 `--checkpoint` 로 이미 로드했으므로 `로드 상태` 칸에 `✅ ... 로드 완료` 로 표시됩니다. 그대로 두면 됩니다.
(다른 체크포인트로 바꾸려면 **체크포인트 선택 → 📂 로드**.)

### ② "🎹 단순 생성" 탭 선택
입력 칸에서 **"MIDI 파일"** 하위 탭을 고르고, 3단계에서 준비한 멜로디 `.mid` 를 끌어다 놓거나 클릭해 업로드합니다.
> (오디오 파일·마이크 입력 탭도 있지만, 별도 라이브러리가 더 필요하므로 **시연은 MIDI 파일 입력을 권장**합니다.)

### ③ 옵션 (그대로 둬도 됨)
- **조건 트랙**: `melody` (기본값 그대로)
- **저장 폴더명**: 비워 두면 자동 타임스탬프. `demo1` 처럼 적으면 `output/demo1/` 에 저장됩니다.

### ④ 생성 파라미터 (기본값 권장)
| 파라미터 | 의미 | 기본/권장 |
|---|---|---|
| **Temperature** | 높을수록 다양·과감, 낮을수록 안정적 | `1.1` |
| **Top-p** | 샘플링 다양성 제한 | `0.92` |
| **CFG Weight** | 멜로디를 더 강하게 따르게 함(0=끄기) | `0`, 더 또렷하게: `1.5–3.0` |
| **Avoid-note Penalty** | 화성에 부딪히는 음 억제(0=끄기) | `0`, 깔끔하게: `2–4` |

### ⑤ "🎵 반주 생성" 버튼 클릭
GPU에서 보통 **십수 초** 걸립니다. 완료되면 오른쪽 출력 칸에 나타납니다:
- **입력 멜로디 (참고용)** — 올린 멜로디 미리듣기
- **생성된 반주 WAV** — AI가 만든 피아노 반주 ▶️
- **멜로디 + 반주 합성** — 둘을 합쳐 들어보기 ▶️
- **MIDI 다운로드** — 생성된 반주 MIDI 파일

▶️ 재생 버튼으로 바로 들어볼 수 있고, 각 오디오의 ⋯ 메뉴로 WAV를 내려받을 수 있습니다.
> 컨테이너 안의 결과물은 호스트의 `output/<폴더명>/` 에도 그대로 저장됩니다.

---

## 7. 데모 종료

데모를 실행한 PowerShell/터미널 창에서 **`Ctrl+C`** 를 누르면 종료됩니다.
`--rm` 옵션 덕에 컨테이너는 자동으로 정리됩니다.

---

## 🛠️ 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| 브라우저에서 접속이 안 됨 (페이지 안 뜸) | ① 터미널에 `Running on local URL: http://0.0.0.0:7860` 가 떴는지 확인 ② 명령에 `--service-ports` 가 들어갔는지 확인(빠지면 포트가 안 열림) |
| `can't open file '/app/app.py'` | 저장소가 최신이 아닙니다. `git pull` 후 다시 실행하세요(`app.py` 마운트 설정이 `docker-compose.yaml`에 포함돼 있어야 함). |
| `RuntimeError: ... 체크포인트` / 모델 로드 실패 | `--checkpoint` 경로/파일명이 실제 `checkpoints/` 안 파일과 정확히 일치하는지 확인. 공백·`=` 포함 파일명은 따옴표로 감싸세요. |
| 반주는 생성됐는데 **소리가 안 남 / WAV가 비어 있음** | `soundfonts/` 에 `.sf2` 파일이 없습니다(2단계). 넣은 뒤 데모를 다시 실행하세요. |
| `pip install` 에서 멈춘 듯 보임 | 첫 실행은 정상적으로 1~2분 걸립니다. 그대로 기다리세요. |
| 포트 7860이 이미 사용 중 | 다른 데모가 떠 있을 수 있습니다. 기존 창에서 `Ctrl+C` 로 끄거나, 명령의 `--port 7860` 과 `app.py ... --port 7860` 을 **둘 다** 다른 번호(예: 7870)로 바꾸세요. |

---

## 부록 — 외부에서 접속할 수 있는 공유 링크 (선택)

같은 PC가 아니라 **다른 사람에게 화면을 보여주고 싶을 때**, 명령의 `python app.py ...` 뒤에 `--share` 를 붙이면 Gradio가 임시 공개 URL(`https://....gradio.live`)을 만들어 줍니다.

```powershell
... python app.py --checkpoint 'checkpoints/<체크포인트파일>' --host 0.0.0.0 --port 7860 --share
```

> 공개 링크는 누구나 접속할 수 있고 약 72시간 후 만료됩니다. 시연이 끝나면 `Ctrl+C` 로 종료하세요.
