# 추론(반주 생성) 가이드

멜로디 MIDI를 올리면 웹 화면(`app.py`)에서 AI가 피아노 반주 WAV를 만들어 줍니다. 시연·발표용으로 좋습니다.

- **GPU**: NVIDIA면 충분(학습보다 가벼움). 없으면 CPU로도 느리지만 동작합니다.
- **OS**: Windows(PowerShell) / Linux
- 모든 명령은 저장소 폴더(`symbolic-accompaniment-transformer/`) 안에서 실행합니다.

학습 가이드로 환경을 이미 만드셨다면 Docker 이미지를 그대로 재사용합니다. 처음이라면 `TRAINING_GUIDE.md`의 0~4단계(설치·클론·빌드)만 먼저 마쳐 주세요. (학습 데이터는 추론에 필요 없습니다.)

---

## 1. 체크포인트 준비

`checkpoints/` 폴더에 학습된 모델(`.ckpt`)이 있어야 합니다.

- **직접 학습한 경우** — 이미 들어 있습니다. `best-epoch=...-val_loss=....ckpt`(val loss 가장 낮은 것)를 쓰면 됩니다.
- **소유자에게 받은 경우** — `jam_checkpoints.zip`을 풀어 넣습니다.

```powershell
Expand-Archive -Path jam_checkpoints.zip -DestinationPath checkpoints -Force   # Windows
```
```bash
unzip -o jam_checkpoints.zip -d checkpoints                                     # Linux
```

## 2. 음색(SoundFont) 준비

음표를 피아노 소리로 렌더링할 때 필요합니다. `jam_soundfonts.zip`을 풀어 `soundfonts/`에 넣습니다.

```powershell
Expand-Archive -Path jam_soundfonts.zip -DestinationPath soundfonts -Force      # Windows
```
```bash
unzip -o jam_soundfonts.zip -d soundfonts                                       # Linux
```

`soundfonts/` 안에 `.sf2` 파일이 하나라도 있으면 데모가 자동으로 찾습니다.

## 3. 멜로디 MIDI 준비

반주를 입힐 단선율 `.mid` 파일 하나가 필요합니다. 가지고 있는 파일을 쓰거나, [MuseScore](https://musescore.org)로 간단히 그려 내보내면 됩니다. 화음이 많은 것보다 **멜로디 한 줄짜리**가 결과가 깔끔합니다.

---

## 4. 데모 실행

아래 명령에서 `<체크포인트파일>`만 1단계에서 확인한 실제 파일명으로 바꿔 실행합니다.

```powershell
docker compose run --rm --service-ports -e WANDB_DISABLED=true train `
  bash -c "pip install -q gradio pyfluidsynth soundfile pedalboard librosa; python app.py --checkpoint 'checkpoints/<체크포인트파일>' --host 0.0.0.0 --port 7860"
```
```bash
docker compose run --rm --service-ports -e WANDB_DISABLED=true train \
  bash -c "pip install -q gradio pyfluidsynth soundfile pedalboard librosa; python app.py --checkpoint 'checkpoints/<체크포인트파일>' --host 0.0.0.0 --port 7860"
```

- 첫 실행은 데모용 라이브러리 설치로 1~2분 걸립니다(학습 이미지엔 없음).
- `--service-ports`가 7860 포트를 열어 줍니다. 빠지면 브라우저로 접속이 안 됩니다.

아래가 보이면 성공입니다 (GPU 없으면 `cuda` 대신 `cpu`):

```
Model ready on cuda.
* Running on local URL:  http://0.0.0.0:7860
```

## 5. 브라우저에서 반주 생성

브라우저에서 **http://localhost:7860** 접속 → `🎵 JAM Transformer` 화면.

1. 맨 위 체크포인트는 이미 로드돼 있습니다(`✅ ... 로드 완료`).
2. **🎹 단순 생성** 탭 → **MIDI 파일** 하위 탭에 3단계 `.mid`를 업로드합니다.
3. 옵션·파라미터는 기본값 그대로 두고 **🎵 반주 생성** 클릭 (보통 십수 초).
4. 오른쪽에 **생성된 반주 WAV**, **멜로디+반주 합성**, **MIDI 다운로드**가 나옵니다. ▶️로 바로 듣고 ⋯ 메뉴로 받을 수 있습니다.

> 결과는 호스트의 `output/` 폴더에도 저장됩니다. 파라미터를 조절하고 싶다면: Temperature(다양성, 기본 1.1), Top-p(0.92), CFG Weight(멜로디를 더 따르게, 1.5~3.0), Avoid-note(충돌음 억제, 2~4).

## 6. 종료

데모 실행 창에서 `Ctrl+C`를 누르면 종료되고 컨테이너도 자동 정리됩니다.

---

## 문제 해결

| 증상 | 해결 |
|---|---|
| 브라우저 접속 안 됨 | 터미널에 `Running on local URL`이 떴는지, 명령에 `--service-ports`가 있는지 확인 |
| `can't open file '/app/app.py'` | 저장소가 최신이 아닙니다. `git pull` 후 다시 실행 |
| 모델 로드 실패 | `--checkpoint` 파일명이 실제 파일과 일치하는지 확인 (공백·`=` 포함 시 따옴표로 감싸기) |
| 소리가 안 남 / WAV가 빔 | `soundfonts/`에 `.sf2`가 없습니다(2단계) |
| `pip install`에서 멈춘 듯 | 첫 실행은 1~2분 정상. 기다리면 됩니다 |
| 포트 7860 사용 중 | 기존 창을 `Ctrl+C`로 끄거나, 명령의 두 `7860`을 모두 다른 번호로 변경 |

> 다른 사람에게 화면을 보여주려면 명령 끝에 `--share`를 붙이면 임시 공개 URL이 생깁니다(약 72시간 유효).
