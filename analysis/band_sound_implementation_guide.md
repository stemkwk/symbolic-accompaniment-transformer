# JAM Transformer: 단일 트랙 반주 ➔ 밴드 사운드 분할 구현 가이드

이 리포트는 Jam Transformer가 생성한 단일 트랙(Polyphonic) 피아노 반주를 음역대(Pitch) 기준으로 분리하여 **베이스, 건반, 리드기타의 밴드 사운드로 렌더링**하는 기능의 구현 방법을 안내합니다. 
본 가이드를 참고하여 직접 소스 코드를 수정하시면 됩니다.

---

## 1. `midi_io.py` 유틸리티 함수 추가
먼저, `miditoolkit`을 사용해 단일 트랙의 노트들을 음역대별로 쪼개어 다중 트랙(Multi-track)으로 재구성하는 함수를 작성합니다.
이 코드를 `src/jam_transformer/utils/midi_io.py` 파일의 하단에 추가하세요.

```python
import miditoolkit

def split_accompaniment_by_pitch(midi: miditoolkit.MidiFile) -> miditoolkit.MidiFile:
    """
    단일 반주 트랙을 음역대에 따라 3개의 악기(Bass, Pad, Lead)로 분할합니다.
    """
    new_midi = miditoolkit.MidiFile(ticks_per_beat=midi.ticks_per_beat)
    new_midi.tempo_changes = midi.tempo_changes
    
    # 분할할 3개의 악기 생성 (GM Program 번호 지정)
    bass_inst = miditoolkit.Instrument(program=33, name="Electric Bass")    # 33: Fretless Bass
    pad_inst = miditoolkit.Instrument(program=4, name="Electric Piano")     # 4: EPiano
    lead_inst = miditoolkit.Instrument(program=27, name="Clean Guitar")     # 27: Clean Electric Guitar
    
    # 기존 MIDI에서 반주(Accompaniment) 트랙 찾기 (일반적으로 프로그램 0번)
    for inst in midi.instruments:
        for note in inst.notes:
            if note.pitch < 45:
                bass_inst.notes.append(note)
            elif 45 <= note.pitch <= 75:
                pad_inst.notes.append(note)
            else:
                lead_inst.notes.append(note)
                
    # 새 MIDI 파일에 악기들 추가
    if len(bass_inst.notes) > 0: new_midi.instruments.append(bass_inst)
    if len(pad_inst.notes) > 0: new_midi.instruments.append(pad_inst)
    if len(lead_inst.notes) > 0: new_midi.instruments.append(lead_inst)
    
    return new_midi
```

---

## 2. `app.py` UI 및 로직 연동
Gradio UI에 선택 버튼을 만들고, 렌더링(합성) 직전에 위에서 만든 분할 함수를 적용하도록 코드를 수정합니다.

### 2.1. UI 컴포넌트 추가 (`build_ui` 함수 내부)
'단순 생성' 및 '루프 스테이션' 탭의 설정 영역에 라디오 버튼을 추가하세요.
```python
# 기존 설정 컴포넌트들 아래에 추가
render_style = gr.Radio(
    choices=["단일 피아노 렌더링", "음역대 분할 (밴드 사운드)"],
    value="단일 피아노 렌더링",
    label="반주 렌더링 스타일"
)
```
버튼 클릭 이벤트(`btn.click`)의 `inputs` 리스트에도 `render_style`을 추가로 넘겨주어야 합니다.

### 2.2. 추론 함수 수정 (`_run_simple` 및 `_run_loop` 함수)
입력 파라미터로 `render_style`을 받도록 수정한 뒤, 모델 추론(`_generate`)이 끝난 직후 WAV 렌더링(`_render`)으로 넘어가기 전 중간에 개입합니다.

```python
# 상단에 Import 추가
from jam_transformer.utils.midi_io import split_accompaniment_by_pitch
import miditoolkit

def _run_simple(..., render_style: str, ...):
    # ... 기존 코드 ...
    
    # 1. 모델이 반주 MIDI 생성
    out_midi, _ = _generate(...) 
    
    # 2. [추가된 로직] 밴드 사운드 선택 시 MIDI 분할
    if render_style == "음역대 분할 (밴드 사운드)":
        midi_obj = miditoolkit.MidiFile(str(out_midi))
        band_midi_obj = split_accompaniment_by_pitch(midi_obj)
        
        band_midi_path = out_dir / "accompaniment_band.mid"
        band_midi_obj.dump(str(band_midi_path))
        
        # 렌더링할 타겟 파일을 밴드 버전으로 교체
        out_midi = band_midi_path

    # 3. FluidSynth로 WAV 렌더링 (기존과 동일)
    out_wav = _render(out_midi, cfg, out_dir / "accompaniment.wav")
    
    # ... 기존 믹싱 코드 ...
```

---

## 3. 작동 원리
1. Jam Transformer 모델 자체는 무조건 **단일 트랙(피아노 형태)**의 폴리포닉 데이터를 생성해 냅니다. (모델 건드리지 않음)
2. `app.py`에서 사용자가 "밴드 사운드"를 선택하면, 방금 막 생성된 단일 트랙 MIDI의 음표(Note) 높낮이를 스캔합니다.
3. 45 미만은 베이스(Program 33)로, 중간은 건반(Program 4)으로, 75 초과는 리드 기타(Program 27)로 재분배하여 **가짜 다중 트랙(Pseudo-multitrack)**을 만듭니다.
4. 이 다중 트랙 MIDI를 기존의 `FluidSynth`(`_render`)에 던져주면, 사운드폰트가 알아서 프로그램 번호를 인식하여 3가지 악기 소리가 섞인 훌륭한 밴드 잼(Jam) 사운드를 뿜어내게 됩니다.
