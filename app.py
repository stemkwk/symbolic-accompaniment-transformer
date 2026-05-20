"""Gradio web demo for JAM Transformer accompaniment generation.

두 가지 모드:
  단순 생성  — MIDI/오디오/마이크 입력 → 반주 생성
  루프 스테이션 — 짧은 악절 녹음 → AI가 처리하는 동안 루프 → 반주 레이어 추가

Launch
------
    python app.py --checkpoint checkpoints/best.ckpt
    python app.py --checkpoint checkpoints/best.ckpt --share   # public URL
"""
from __future__ import annotations

import argparse
import math
import tempfile
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import torch
from scipy.io import wavfile

from jam_transformer.audio import apply_dsp, audio_to_midi, render_midi_to_wav
from jam_transformer.config import load_config
from jam_transformer.logger import logger
from jam_transformer.pipeline import generate_accompaniment, load_checkpoint
from jam_transformer.tokenizer import build_tokenizer

try:
    import gradio as gr
except ImportError:
    raise ImportError(
        "Gradio is required for the web demo.\n"
        "Run: pip install 'jam_transformer[demo]'"
    )

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------
_STATE: dict = {}


def _get_state():
    if not _STATE.get("lit"):
        raise RuntimeError("체크포인트가 로드되지 않았습니다. 상단에서 체크포인트를 선택하고 '로드'를 눌러주세요.")
    return _STATE["cfg"], _STATE["lit"], _STATE["tokenizer"]


def _list_checkpoints() -> list[str]:
    ckpt_dir = Path(_STATE.get("ckpt_dir", "checkpoints"))
    if not ckpt_dir.exists():
        return []
    return sorted(p.name for p in ckpt_dir.glob("*.ckpt"))


def _load_model(ckpt_name: str) -> str:
    if not ckpt_name:
        return "⚠️ 체크포인트를 선택해주세요."
    try:
        cfg = _STATE["cfg"]
        tokenizer = _STATE["tokenizer"]
        ckpt_path = Path(_STATE["ckpt_dir"]) / ckpt_name
        logger.info(f"Loading checkpoint: {ckpt_path}")
        lit = load_checkpoint(str(ckpt_path), cfg, tokenizer.vocab_size)
        lit.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lit.to(device)
        _STATE["lit"] = lit
        _STATE["current_ckpt"] = ckpt_name
        return f"✅ {ckpt_name} 로드 완료  ({device})"
    except Exception as e:
        logger.exception("Checkpoint load failed")
        return f"❌ 로드 실패: {e}"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _transcribe_audio(audio_path: Path, acfg, denoise: bool) -> Path:
    out_midi = Path(tempfile.mktemp(suffix=".mid"))
    audio_to_midi(
        audio_path, out_midi,
        denoise=denoise or acfg.denoise,
        onset_threshold=acfg.onset_threshold,
        frame_threshold=acfg.frame_threshold,
        min_note_length_ms=acfg.min_note_length_ms,
        min_frequency=acfg.min_frequency,
        max_frequency=acfg.max_frequency,
    )
    return out_midi


def _detect_bpm(audio_path: Path, fallback: float = 120.0) -> float:
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        if 40 < bpm < 240:
            return bpm
    except Exception:
        pass
    return fallback


def _render(midi_path: Path, cfg, out_wav: Path | None = None) -> Path | None:
    icfg = cfg.inference
    if out_wav is None:
        out_wav = Path(tempfile.mktemp(suffix=".wav"))
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    render_midi_to_wav(midi_path, out_wav, icfg.soundfont, icfg.sample_rate)
    if not out_wav.exists():
        return None
    try:
        apply_dsp(out_wav, out_wav, cfg.dsp)
    except ImportError:
        logger.warning("pedalboard not installed — skipping DSP effects.")
    return out_wav


def _loop_and_mix(melody_wav: Path, accomp_wav: Path, out_path: Path) -> None:
    """Loop melody to match accompaniment length and mix at equal volume."""
    sr_m, mel = wavfile.read(str(melody_wav))
    sr_a, acc = wavfile.read(str(accomp_wav))

    # Mono-ify
    if mel.ndim > 1:
        mel = mel.mean(axis=1)
    if acc.ndim > 1:
        acc = acc.mean(axis=1)

    # Simple resample if sample rates differ (linear interp)
    if sr_m != sr_a:
        from scipy.signal import resample
        mel = resample(mel, int(len(mel) * sr_a / sr_m))

    # Loop melody to cover accompaniment length
    n_repeats = math.ceil(len(acc) / max(len(mel), 1))
    mel_looped = np.tile(mel, n_repeats)[: len(acc)]

    # Mix with equal weight, prevent clipping
    mixed = mel_looped.astype(np.float32) * 0.5 + acc.astype(np.float32) * 0.5
    mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
    wavfile.write(str(out_path), sr_a, mixed)


def _default_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _generate(melody_midi: Path, cfg, lit, tokenizer, cond_tracks,
              temperature, top_p, cfg_w,
              out_midi: Path | None = None) -> tuple[Path, float]:
    midi, tempo = generate_accompaniment(
        melody_midi=melody_midi,
        cfg=cfg, lit=lit, tokenizer=tokenizer,
        cond_tracks=cond_tracks,
        temperature=temperature,
        top_p=top_p,
        cfg_w=cfg_w,
    )
    if out_midi is None:
        out_midi = Path(tempfile.mktemp(suffix=".mid"))
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(out_midi))
    return out_midi, tempo


# ---------------------------------------------------------------------------
# Mode 1: 단순 생성
# ---------------------------------------------------------------------------
def _run_simple(
    midi_file,
    audio_file,
    mic_audio,
    denoise: bool,
    cond_tracks_str: str,
    temperature: float,
    top_p: float,
    cfg_w: float,
    output_name: str,
) -> tuple:
    cfg, lit, tokenizer = _get_state()
    acfg = cfg.audio_input
    cond_tracks = [t.strip() for t in cond_tracks_str.split(",") if t.strip()]

    # 출력 경로 결정
    name = output_name.strip() or _default_name()
    out_dir = Path("output") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_audio = audio_file or mic_audio
        if raw_audio:
            input_preview = raw_audio
            input_wav_for_mix = Path(raw_audio)
            melody_midi = _transcribe_audio(Path(raw_audio), acfg, denoise)
        elif midi_file:
            melody_midi = Path(midi_file if isinstance(midi_file, str) else midi_file.name)
            input_preview_wav = _render(melody_midi, cfg, out_dir / "input.wav")
            input_preview = str(input_preview_wav) if input_preview_wav else None
            input_wav_for_mix = input_preview_wav
        else:
            return None, None, None, None, "⚠️ 입력을 제공해주세요."

        out_midi, _ = _generate(melody_midi, cfg, lit, tokenizer,
                                 cond_tracks, temperature, top_p, cfg_w,
                                 out_midi=out_dir / "accompaniment.mid")
        out_wav = _render(out_midi, cfg, out_dir / "accompaniment.wav")

        # 입력 + 출력 합성
        mixed_wav = None
        if out_wav and input_wav_for_mix and input_wav_for_mix.exists():
            try:
                mixed_path = out_dir / "mixed.wav"
                _loop_and_mix(input_wav_for_mix, out_wav, mixed_path)
                mixed_wav = mixed_path
            except Exception:
                logger.warning("믹싱 실패 — 합성 WAV를 건너뜁니다.")

        status = f"✅ 생성 완료  |  저장 위치: output/{name}/"
        if out_wav is None:
            status += "  (사운드폰트 없음 — MIDI만 제공)"
        return (
            input_preview,
            str(out_wav) if out_wav else None,
            str(mixed_wav) if mixed_wav else None,
            str(out_midi),
            status,
        )

    except Exception as e:
        logger.exception("Simple generation failed")
        return None, None, None, None, f"❌ 오류: {e}"


# ---------------------------------------------------------------------------
# Mode 2: 루프 스테이션
# ---------------------------------------------------------------------------
def _run_loop(
    mic_audio,
    bpm_mode: str,
    bpm_manual: float,
    denoise: bool,
    cond_tracks_str: str,
    temperature: float,
    top_p: float,
    cfg_w: float,
) -> tuple:
    """
    Returns:
      accomp_wav   — 반주만 (루프 스테이션에 레이어로 추가)
      mixed_wav    — 멜로디 루프 + 반주 합성
      out_midi     — MIDI 다운로드
      status_text
    """
    cfg, lit, tokenizer = _get_state()
    acfg = cfg.audio_input
    cond_tracks = [t.strip() for t in cond_tracks_str.split(",") if t.strip()]

    try:
        if not mic_audio:
            return None, None, None, "⚠️ 마이크로 악절을 먼저 녹음해주세요."

        melody_wav = Path(mic_audio)

        # BPM 감지
        if bpm_mode == "자동 감지":
            bpm = _detect_bpm(melody_wav)
            logger.info(f"Auto-detected BPM: {bpm:.1f}")
        else:
            bpm = bpm_manual

        # 오디오 → MIDI
        melody_midi = _transcribe_audio(melody_wav, acfg, denoise)

        # 반주 생성 — 감지된 BPM을 tempo_override로 전달
        out_midi, tempo = generate_accompaniment(
            melody_midi=melody_midi,
            cfg=cfg, lit=lit, tokenizer=tokenizer,
            cond_tracks=cond_tracks,
            tempo_override=bpm,
            temperature=temperature,
            top_p=top_p,
            cfg_w=cfg_w,
        )
        out_midi_path = Path(tempfile.mktemp(suffix=".mid"))
        out_midi.dump(str(out_midi_path))
        out_midi = out_midi_path

        # 반주 WAV 렌더링
        accomp_wav = _render(out_midi, cfg)

        # 멜로디 루프 + 반주 합성
        mixed_wav = None
        if accomp_wav and melody_wav.exists():
            mixed_wav = Path(tempfile.mktemp(suffix=".wav"))
            _loop_and_mix(melody_wav, accomp_wav, mixed_wav)

        status = f"✅ 완료  |  감지 BPM: {bpm:.0f}"
        if accomp_wav is None:
            status += "  (사운드폰트 없음 — MIDI만 제공)"

        return (
            str(accomp_wav) if accomp_wav else None,
            str(mixed_wav) if mixed_wav else None,
            str(out_midi),
            status,
        )

    except Exception as e:
        logger.exception("Loop station generation failed")
        return None, None, None, f"❌ 오류: {e}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    cfg, _, _ = _get_state()
    icfg = cfg.inference
    acfg = cfg.audio_input

    # 공통 파라미터 슬라이더를 만드는 헬퍼
    def _param_sliders():
        temperature = gr.Slider(
            0.5, 2.0, step=0.05, value=icfg.temperature,
            label="Temperature  (높을수록 다양, 낮을수록 안정적)",
        )
        top_p = gr.Slider(
            0.5, 1.0, step=0.01, value=icfg.top_p,
            label="Top-p",
        )
        cfg_w = gr.Slider(
            0.0, 5.0, step=0.1, value=0.0,
            label="CFG Weight  (0 = off, 권장 1.5–3.0)",
        )
        return temperature, top_p, cfg_w

    with gr.Blocks(title="JAM Transformer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎵 JAM Transformer\n"
            "멜로디를 입력하면 AI가 반주(accompaniment)를 생성합니다.\n\n"
            "**단순 생성**: 파일 또는 마이크 입력 → 반주 출력  \n"
            "**루프 스테이션**: 짧은 악절을 녹음하면 처리하는 동안 루프를 계속 연주하세요 — 완료 후 반주 레이어를 추가합니다."
        )

        # ── 체크포인트 선택기 ────────────────────────────────────────────────
        available = _list_checkpoints()
        current = _STATE.get("current_ckpt", "")
        with gr.Row():
            ckpt_dropdown = gr.Dropdown(
                choices=available,
                value=current if current in available else (available[0] if available else None),
                label="체크포인트 선택",
                scale=4,
            )
            ckpt_refresh_btn = gr.Button("🔄 새로고침", scale=1)
            ckpt_load_btn = gr.Button("📂 로드", variant="primary", scale=1)
        ckpt_status = gr.Textbox(
            label="로드 상태",
            value=f"✅ {current} 로드 완료" if current else "⚠️ 체크포인트를 선택하고 로드해주세요.",
            interactive=False,
        )

        ckpt_refresh_btn.click(
            fn=lambda: gr.update(choices=_list_checkpoints()),
            outputs=ckpt_dropdown,
        )
        ckpt_load_btn.click(
            fn=_load_model,
            inputs=ckpt_dropdown,
            outputs=ckpt_status,
        )

        gr.Markdown("---")

        with gr.Tabs():
            # ================================================================
            # 탭 1: 단순 생성
            # ================================================================
            with gr.Tab("🎹 단순 생성"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 입력")
                        with gr.Tabs():
                            with gr.Tab("MIDI 파일"):
                                simple_midi = gr.File(
                                    label="멜로디 MIDI (.mid / .midi)",
                                    file_types=[".mid", ".midi"],
                                )
                            with gr.Tab("오디오 파일"):
                                simple_audio = gr.Audio(
                                    label="WAV / MP3 / FLAC",
                                    sources=["upload"],
                                    type="filepath",
                                )
                            with gr.Tab("마이크"):
                                simple_mic = gr.Audio(
                                    label="마이크 녹음",
                                    sources=["microphone"],
                                    type="filepath",
                                )

                        gr.Markdown("### 옵션")
                        simple_denoise = gr.Checkbox(
                            label="노이즈 제거", value=acfg.denoise,
                        )
                        simple_cond = gr.Textbox(
                            label="조건 트랙", value="melody",
                        )
                        simple_outname = gr.Textbox(
                            label="저장 폴더명 (output/ 하위, 비우면 타임스탬프)",
                            placeholder="예: test_001  →  output/test_001/",
                            value="",
                        )

                        gr.Markdown("### 생성 파라미터")
                        simple_temp, simple_topp, simple_cfgw = _param_sliders()
                        simple_btn = gr.Button("🎵 반주 생성", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 출력")
                        simple_status = gr.Textbox(label="상태", interactive=False)
                        simple_input_preview = gr.Audio(
                            label="입력 멜로디 (참고용)", type="filepath",
                        )
                        simple_wav_out = gr.Audio(label="생성된 반주 WAV", type="filepath")
                        simple_mixed_out = gr.Audio(
                            label="멜로디 + 반주 합성", type="filepath",
                        )
                        simple_midi_out = gr.File(label="MIDI 다운로드")

                simple_btn.click(
                    fn=_run_simple,
                    inputs=[simple_midi, simple_audio, simple_mic,
                            simple_denoise, simple_cond,
                            simple_temp, simple_topp, simple_cfgw,
                            simple_outname],
                    outputs=[simple_input_preview, simple_wav_out,
                             simple_mixed_out, simple_midi_out, simple_status],
                )

            # ================================================================
            # 탭 2: 루프 스테이션
            # ================================================================
            with gr.Tab("🎸 루프 스테이션"):
                gr.Markdown(
                    "> **사용법**: ① 2–4마디 멜로디를 녹음합니다 → ② '반주 생성' 클릭 → "
                    "③ **처리하는 동안 루프를 계속 연주하세요** (약 15–30초) → "
                    "④ 완료 후 반주 레이어를 루프에 추가합니다."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 악절 녹음")
                        loop_mic = gr.Audio(
                            label="🎙️ 멜로디 악절 (2–4마디 권장)",
                            sources=["microphone"],
                            type="filepath",
                        )

                        gr.Markdown("### BPM")
                        bpm_mode = gr.Radio(
                            choices=["자동 감지", "수동 설정"],
                            value="자동 감지",
                            label="BPM 설정 방식",
                        )
                        bpm_slider = gr.Slider(
                            60, 200, step=1, value=120,
                            label="BPM (수동 설정 시)",
                            visible=False,
                        )
                        bpm_mode.change(
                            fn=lambda m: gr.update(visible=(m == "수동 설정")),
                            inputs=bpm_mode,
                            outputs=bpm_slider,
                        )

                        gr.Markdown("### 옵션")
                        loop_denoise = gr.Checkbox(
                            label="노이즈 제거 (권장)", value=True,
                        )
                        loop_cond = gr.Textbox(
                            label="조건 트랙", value="melody",
                        )

                        gr.Markdown("### 생성 파라미터")
                        loop_temp, loop_topp, loop_cfgw = _param_sliders()
                        loop_btn = gr.Button(
                            "🎸 반주 생성 (처리 중 루프 계속 연주!)",
                            variant="primary", size="lg",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 출력")
                        loop_status = gr.Textbox(label="상태", interactive=False)

                        gr.Markdown("**반주만** — 루프 스테이션에 새 레이어로 추가")
                        loop_accomp_out = gr.Audio(
                            label="반주 WAV (루프 레이어)", type="filepath",
                        )

                        gr.Markdown("**멜로디 + 반주 합성** — 함께 들어보기")
                        loop_mixed_out = gr.Audio(
                            label="합성 WAV (멜로디 루프 × N + 반주)", type="filepath",
                        )

                        loop_midi_out = gr.File(label="MIDI 다운로드")

                loop_btn.click(
                    fn=_run_loop,
                    inputs=[loop_mic, bpm_mode, bpm_slider,
                            loop_denoise, loop_cond,
                            loop_temp, loop_topp, loop_cfgw],
                    outputs=[loop_accomp_out, loop_mixed_out,
                             loop_midi_out, loop_status],
                )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="JAM Transformer Gradio web demo.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="초기 로드할 체크포인트 경로. 생략 시 UI에서 선택.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="체크포인트 디렉토리 (드롭다운 목록 소스).")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = build_tokenizer(cfg.tokenizer)

    _STATE["cfg"] = cfg
    _STATE["tokenizer"] = tokenizer
    _STATE["ckpt_dir"] = args.checkpoint_dir
    _STATE["lit"] = None
    _STATE["current_ckpt"] = ""

    # --checkpoint 가 주어졌으면 바로 로드
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        ckpt_name = ckpt_path.name
        # 절대/상대 경로 모두 지원: 디렉토리 외부 경로도 허용
        if not ckpt_path.parent.samefile(Path(args.checkpoint_dir)) \
                if ckpt_path.parent.exists() else True:
            # checkpoint_dir 바깥 경로면 그 부모를 dir로 설정
            _STATE["ckpt_dir"] = str(ckpt_path.parent)
        logger.info(f"Loading checkpoint: {ckpt_path}")
        lit = load_checkpoint(str(ckpt_path), cfg, tokenizer.vocab_size)
        lit.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lit.to(device)
        _STATE["lit"] = lit
        _STATE["current_ckpt"] = ckpt_name
        logger.info(f"Model ready on {device}.")

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
