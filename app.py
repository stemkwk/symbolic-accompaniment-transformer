"""Gradio web demo for JAM Transformer accompaniment generation.

Launch
------
    python app.py --checkpoint checkpoints/best.ckpt
    python app.py --checkpoint checkpoints/best.ckpt --share   # public URL
    python app.py --checkpoint checkpoints/best.ckpt --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch

from jam_transformer.audio import audio_to_midi, render_midi_to_wav
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
    if not _STATE:
        raise RuntimeError("Model not loaded. Pass --checkpoint when launching app.py.")
    return _STATE["cfg"], _STATE["lit"], _STATE["tokenizer"]


# ---------------------------------------------------------------------------
# Core inference wrapper
# ---------------------------------------------------------------------------
def _run(
    melody_midi_path: str | None,
    audio_path: str | None,        # from gr.Audio upload
    mic_audio: str | None,         # from gr.Audio microphone
    denoise: bool,
    temperature: float,
    top_p: float,
    cfg_w: float,
    cond_tracks_str: str,
) -> tuple[str | None, str | None, str]:
    """Called by every Gradio event. Returns (wav_path, midi_path, status_text)."""
    cfg, lit, tokenizer = _get_state()
    acfg = cfg.audio_input

    # ── Resolve audio source → WAV file ──────────────────────────────────────
    raw_audio: str | None = audio_path or mic_audio
    tmp_files: list[Path] = []

    try:
        if raw_audio:
            raw_wav = Path(raw_audio)
            transcribed = Path(tempfile.mktemp(suffix=".mid"))
            tmp_files.append(transcribed)
            audio_to_midi(
                raw_wav, transcribed,
                denoise=denoise or acfg.denoise,
                onset_threshold=acfg.onset_threshold,
                frame_threshold=acfg.frame_threshold,
                min_note_length_ms=acfg.min_note_length_ms,
                min_frequency=acfg.min_frequency,
                max_frequency=acfg.max_frequency,
            )
            melody_midi = transcribed
        elif melody_midi_path:
            melody_midi = Path(melody_midi_path)
        else:
            return None, None, "입력을 제공해주세요 (MIDI 파일, 오디오 파일, 또는 마이크 녹음)."

        cond_tracks = [t.strip() for t in cond_tracks_str.split(",") if t.strip()]

        midi, tempo = generate_accompaniment(
            melody_midi=melody_midi,
            cfg=cfg,
            lit=lit,
            tokenizer=tokenizer,
            cond_tracks=cond_tracks,
            temperature=temperature,
            top_p=top_p,
            cfg_w=cfg_w,
        )

        # Save output MIDI to temp file
        out_midi = Path(tempfile.mktemp(suffix=".mid"))
        tmp_files.append(out_midi)
        midi.dump(str(out_midi))

        # Try WAV rendering (needs soundfont)
        out_wav = Path(tempfile.mktemp(suffix=".wav"))
        tmp_files.append(out_wav)
        icfg = cfg.inference
        render_midi_to_wav(out_midi, out_wav, icfg.soundfont, icfg.sample_rate)

        wav_result = str(out_wav) if out_wav.exists() else None
        status = "✅ 생성 완료"
        if wav_result is None:
            status += " (사운드폰트 없음 — MIDI만 제공)"
        return wav_result, str(out_midi), status

    except Exception as e:
        logger.exception("Generation failed")
        return None, None, f"❌ 오류: {e}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    cfg, _, _ = _get_state()
    icfg = cfg.inference
    acfg = cfg.audio_input

    with gr.Blocks(title="JAM Transformer — Accompaniment Demo") as demo:
        gr.Markdown("# JAM Transformer\n멜로디를 입력하면 반주를 생성합니다.")

        with gr.Row():
            # ── Left column: inputs ─────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 입력")
                with gr.Tab("MIDI 파일"):
                    midi_input = gr.File(
                        label="멜로디 MIDI 파일 (.mid)",
                        file_types=[".mid", ".midi"],
                    )

                with gr.Tab("오디오 파일"):
                    audio_input = gr.Audio(
                        label="오디오 파일 (WAV / MP3 / FLAC)",
                        sources=["upload"],
                        type="filepath",
                    )

                with gr.Tab("마이크 녹음"):
                    mic_input = gr.Audio(
                        label="마이크로 멜로디 녹음",
                        sources=["microphone"],
                        type="filepath",
                    )

                gr.Markdown("### 옵션")
                denoise_check = gr.Checkbox(
                    label="노이즈 제거 (마이크/오디오 입력 시 권장)",
                    value=acfg.denoise,
                )
                cond_tracks = gr.Textbox(
                    label="조건 트랙 (쉼표 구분)",
                    value="melody",
                )

                gr.Markdown("### 생성 파라미터")
                temperature = gr.Slider(
                    minimum=0.5, maximum=2.0, step=0.05,
                    value=icfg.temperature,
                    label="Temperature (높을수록 다양, 낮을수록 안정적)",
                )
                top_p = gr.Slider(
                    minimum=0.5, maximum=1.0, step=0.01,
                    value=icfg.top_p,
                    label="Top-p",
                )
                cfg_w = gr.Slider(
                    minimum=0.0, maximum=5.0, step=0.1,
                    value=0.0,
                    label="CFG Weight (0 = off, 1.5–3.0 권장)",
                )

                run_btn = gr.Button("🎵 반주 생성", variant="primary")

            # ── Right column: outputs ───────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 출력")
                status_box = gr.Textbox(label="상태", interactive=False)
                audio_out = gr.Audio(label="생성된 반주 (WAV)", type="filepath")
                midi_out = gr.File(label="MIDI 다운로드", file_types=[".mid"])

        # ── Event binding ───────────────────────────────────────────────────
        inputs = [
            midi_input, audio_input, mic_input,
            denoise_check, temperature, top_p, cfg_w, cond_tracks,
        ]
        outputs = [audio_out, midi_out, status_box]
        run_btn.click(fn=_run, inputs=inputs, outputs=outputs)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="JAM Transformer Gradio web demo.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = build_tokenizer(cfg.tokenizer)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    lit = load_checkpoint(args.checkpoint, cfg, tokenizer.vocab_size)
    lit.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit.to(device)
    logger.info(f"Model ready on {device}.")

    _STATE["cfg"] = cfg
    _STATE["lit"] = lit
    _STATE["tokenizer"] = tokenizer

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
