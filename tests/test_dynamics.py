"""Dynamic checks that the *training* actually trains.

The smoke tests prove that forward/backward run without errors, but a model
that doesn't learn at all would still pass them. These tests catch the
"hooked the loss to the wrong tensor / mask is wrong / optimizer not stepping"
class of bug — the kind that wastes 14h of paid GPU before you notice.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
from scripts.prepare_data import _synthesize_song

from jam_transformer.config import load_config
from jam_transformer.dataset import JamTokenDataset
from jam_transformer.lightning_module import JamTransformerLightning
from jam_transformer.tokenizer import build_tokenizer


CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


def _build_tiny_dataset(tmp_path: Path, n_songs: int = 6):
    """Tokenize a handful of synthetic songs and write the chunk-index cache."""
    cfg = load_config(CONFIG_PATH)
    tokenizer = build_tokenizer(cfg.tokenizer)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    lengths = {}
    for i in range(n_songs):
        events, tempo, key_root, key_mode = _synthesize_song(seed=i)
        ids, mask = tokenizer.encode_song(
            events,
            condition_tracks=["melody"],
            target_tracks=["accompaniment"],
            tempo_bpm=tempo,
            key_root=key_root,
            key_mode=key_mode,
        )
        name = f"song_{i:03d}.pt"
        torch.save({
            "ids":      torch.tensor(ids,  dtype=torch.long),
            "mask":     torch.tensor(mask, dtype=torch.bool),
            "name":     f"synth_{i}",
            "key_root": key_root,
            "key_mode": key_mode,
        }, data_dir / name)
        lengths[name] = len(ids)
    # Write the meta + index that the rest of the pipeline expects.
    from scripts.prepare_data import tokenizer_fingerprint
    from dataclasses import asdict
    (data_dir / "_dataset_meta.json").write_text(
        json.dumps({
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_config": asdict(tokenizer.cfg),
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer.cfg),
            "n_shards": n_songs,
            "cond_tracks": ["melody"],
            "target_tracks": ["accompaniment"],
        }, indent=2),
        encoding="utf-8",
    )
    (data_dir / "_chunk_index.json").write_text(
        json.dumps(lengths, indent=2), encoding="utf-8",
    )
    return cfg, tokenizer, data_dir


def _make_tiny_model(cfg, vocab_size: int) -> JamTransformerLightning:
    """Shrink the model so this test fits CPU in seconds."""
    cfg.model.d_model = 64
    cfg.model.n_layers = 2
    cfg.model.n_heads = 2
    cfg.model.d_ff = 128
    cfg.model.dropout = 0.0    # nothing to regularise, we want it to overfit
    cfg.model.compile = False
    return JamTransformerLightning(cfg, vocab_size=vocab_size, total_steps=200)


@pytest.mark.timeout(180)
def test_loss_decreases_on_repeated_overfit(tmp_path: Path):
    """30 optimization steps on a single mini-batch should drop CE substantially.

    We bypass Lightning's Trainer and step the optimizer manually so the test
    is fast and the assertion is unambiguous (no logging machinery between us
    and the loss values)."""
    cfg, tokenizer, data_dir = _build_tiny_dataset(tmp_path, n_songs=4)
    cfg.training.batch_size = 2
    cfg.tokenizer.max_seq_len = 256        # shorter sequences = faster steps
    ds = JamTokenDataset(data_dir, cfg, tokenizer, train=True)
    assert len(ds) >= 2
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    model = _make_tiny_model(cfg, vocab_size=tokenizer.vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    batch = [b.to(device) for b in batch]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses: list[float] = []
    for _ in range(30):
        opt.zero_grad()
        loss, _ = model._compute_loss(batch)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    first = losses[0]
    last = losses[-1]
    # Random init CE ≈ ln(vocab_size). With 30 steps on a single batch the
    # model should at minimum cut the loss in half — if not, the optimizer
    # isn't actually stepping the weights or the loss mask is empty.
    assert last < first * 0.5, (
        f"loss did not decrease: first={first:.3f}, last={last:.3f}, "
        f"trajectory={[round(x, 3) for x in losses[::5]]}"
    )
    assert last < 3.0, f"loss should plummet on a single batch; got {last:.3f}"


def test_loss_mask_nonempty_on_real_batch(tmp_path: Path):
    """The condition-masking machinery is silently broken if the mask sums to
    zero — we'd compute loss on no tokens. Sanity-check the dataset."""
    cfg, tokenizer, data_dir = _build_tiny_dataset(tmp_path, n_songs=4)
    cfg.training.batch_size = 2
    cfg.tokenizer.max_seq_len = 256
    ds = JamTokenDataset(data_dir, cfg, tokenizer, train=True)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    x, y, loss_mask = next(iter(loader))
    assert loss_mask.dtype == torch.bool
    assert loss_mask.shape == y.shape
    assert int(loss_mask.sum()) > 0, "loss_mask is empty — no tokens would contribute to CE"


@pytest.mark.timeout(300)
def test_resume_continues_epoch_counter(tmp_path: Path):
    """Train 1 epoch, save checkpoint, resume, verify Lightning's epoch
    counter is 1 (not 0). Pre-emption recovery depends on this."""
    import pytorch_lightning as pl
    cfg, tokenizer, data_dir = _build_tiny_dataset(tmp_path, n_songs=4)
    cfg.tokenizer.max_seq_len = 256
    cfg.training.batch_size = 2
    cfg.training.csv_logger_enabled = False
    cfg.training.log_to_file = False
    cfg.training.early_stopping_enabled = False
    cfg.training.checkpoint_every_n_train_steps = 0
    cfg.training.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.training.checkpoint_save_top_k = 1
    cfg.training.checkpoint_save_last = True

    ds = JamTokenDataset(data_dir, cfg, tokenizer, train=True)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    def _trainer(max_epochs):
        ckpt_cb = pl.callbacks.ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            save_last=True, save_top_k=0,
        )
        return pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto", devices="auto",
            precision="32",        # cpu-friendly
            logger=False,
            callbacks=[ckpt_cb],
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    model = _make_tiny_model(cfg, vocab_size=tokenizer.vocab_size)
    _trainer(max_epochs=1).fit(model, train_dataloaders=loader, val_dataloaders=loader)

    ckpt_path = Path(cfg.training.checkpoint_dir) / "last.ckpt"
    assert ckpt_path.exists(), "Lightning didn't write last.ckpt"

    # Resume into a fresh trainer with max_epochs=2 — global_step / epoch must
    # advance past the values saved in the checkpoint.
    model2 = _make_tiny_model(cfg, vocab_size=tokenizer.vocab_size)
    trainer2 = _trainer(max_epochs=2)
    trainer2.fit(model2, train_dataloaders=loader, val_dataloaders=loader,
                 ckpt_path=str(ckpt_path))
    assert trainer2.current_epoch >= 2, (
        f"resume failed: current_epoch={trainer2.current_epoch} after resuming "
        f"from epoch-1 checkpoint into max_epochs=2"
    )
