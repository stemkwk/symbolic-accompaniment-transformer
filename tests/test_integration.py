"""End-to-end pipeline test.

Drives the same path a paid GPU run would take, but on synthetic data and a
tiny model so it finishes in well under a minute on CPU:

    prepare_data (synthetic)
        → train (fast_dev_run, 1 batch)
            → inference on a prompt MIDI
                → verify the output MIDI is well-formed.

If this test passes, all four scripts wire together — the kind of breakage
the unit tests miss because they import modules but never run them as scripts.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import miditoolkit
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


def _run(cmd: list[str], env_extra: dict | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["WANDB_DISABLED"] = "true"
    if env_extra:
        env.update(env_extra)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def test_end_to_end_pipeline(tmp_path: Path):
    out_dir   = tmp_path / "processed"
    ckpt_dir  = tmp_path / "checkpoints"
    log_dir   = tmp_path / "logs"
    prompt_mid = tmp_path / "prompt.mid"
    out_mid    = tmp_path / "accompaniment.mid"

    # 1) prepare synthetic data
    _run([
        sys.executable, "scripts/prepare_data.py",
        "--synthetic", "--num_songs", "4",
        "--out_dir", str(out_dir),
        "--config", str(CONFIG_PATH),
    ])
    assert any(out_dir.glob("*.pt"))
    assert (out_dir / "_dataset_meta.json").exists()

    # 2) train one fast_dev_run "epoch" with a tiny model
    _run([
        sys.executable, "scripts/train.py",
        "--config", str(CONFIG_PATH),
        "--data_dir", str(out_dir),
        "--fast_dev_run",
        "--batch_size", "2",
        "--run_name", "ci_smoke",
        "--set", f"training.checkpoint_dir={ckpt_dir.as_posix()}",
        "--set", f"training.log_dir={log_dir.as_posix()}",
        "--set", "training.checkpoint_every_n_train_steps=0",
        "--set", "training.early_stopping_enabled=false",
        "--set", "training.csv_logger_enabled=false",
        "--set", "training.log_to_file=false",
        "--set", "model.d_model=64",
        "--set", "model.n_layers=2",
        "--set", "model.n_heads=2",
        "--set", "model.d_ff=128",
        "--set", "model.compile=false",
    ])

    # fast_dev_run doesn't write a checkpoint — do a real 1-epoch run for
    # the inference step. Reuse the same data dir.
    _run([
        sys.executable, "scripts/train.py",
        "--config", str(CONFIG_PATH),
        "--data_dir", str(out_dir),
        "--epochs", "1",
        "--batch_size", "2",
        "--run_name", "ci_real",
        "--set", f"training.checkpoint_dir={ckpt_dir.as_posix()}",
        "--set", f"training.log_dir={log_dir.as_posix()}",
        "--set", "training.checkpoint_every_n_train_steps=0",
        "--set", "training.early_stopping_enabled=false",
        "--set", "training.csv_logger_enabled=false",
        "--set", "training.log_to_file=false",
        "--set", "model.d_model=64",
        "--set", "model.n_layers=2",
        "--set", "model.n_heads=2",
        "--set", "model.d_ff=128",
        "--set", "model.compile=false",
    ])
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    assert ckpts, f"no checkpoints written to {ckpt_dir}"

    # 3) build a prompt MIDI from one synthetic song
    from scripts.prepare_data import _synthesize_song
    from jam_transformer.config import load_config
    from jam_transformer.midi_io import events_to_midi
    cfg = load_config(CONFIG_PATH)
    events, tempo, _kr, _km = _synthesize_song(seed=12345)
    melody = [e for e in events if e.track == "melody"]
    events_to_midi(melody, cfg.tokenizer, tempo_bpm=tempo).dump(str(prompt_mid))
    assert prompt_mid.exists()

    # 4) inference
    last_ckpt = ckpt_dir / "last.ckpt"
    assert last_ckpt.exists(), f"missing {last_ckpt}"
    _run([
        sys.executable, "scripts/inference.py",
        "--config", str(CONFIG_PATH),
        "--checkpoint", str(last_ckpt),
        "--melody_midi", str(prompt_mid),
        "--output", str(out_mid),
        "--max_new_tokens", "64",
        "--set", "model.d_model=64",
        "--set", "model.n_layers=2",
        "--set", "model.n_heads=2",
        "--set", "model.d_ff=128",
    ])

    # 5) verify the output MIDI is well-formed
    assert out_mid.exists()
    midi = miditoolkit.MidiFile(str(out_mid))
    total_notes = sum(len(inst.notes) for inst in midi.instruments)
    # The melody is always echoed back. Target notes may or may not exist
    # after only 1 epoch (the LM might emit garbage), but the file should at
    # least be parseable with the melody present.
    assert total_notes >= len(melody), (
        f"output MIDI lost melody notes (have {total_notes}, expect >= {len(melody)})"
    )
