"""Train the jam Transformer.

Deployment-oriented features wired in here (designed for paid GPU servers):

- `--dry_run_steps N`     run N steps, print throughput + per-epoch budget, exit.
- `--set section.key=val` override any nested config field without editing YAML.
- File logging                always written under `<log_dir>/<run_name>.log`.
- CSV metrics logger          writes metrics to disk even if W&B is offline.
- EarlyStopping               aborts plateaued runs so you stop burning GPU.
- Step-level ModelCheckpoint  survives mid-epoch crashes on cheap pre-emptible boxes.
- Tokenizer fingerprint check refuses mismatched processed data.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import List, Optional

# Load .env before anything else so WANDB_API_KEY etc. are available.
# Safe to call even when python-dotenv is missing (ImportError → silently skipped).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from jam_transformer.config import AppConfig, load_config
from jam_transformer.dataset import JamTokenDataset, assert_data_matches_config
from jam_transformer.lightning_module import JamTransformerLightning
from jam_transformer.utils.logger import attach_file_sink, logger
from jam_transformer.utils.overrides import apply_overrides
from jam_transformer.tokenizer import build_tokenizer
from jam_transformer.utils.hardware import get_env_optimizations


# ---------------------------------------------------------------------------
# Loggers / callbacks
# ---------------------------------------------------------------------------
def _build_loggers(config: AppConfig, run_name: str):
    """Return a list of Lightning loggers. CSV is always on (if configured)
    so metrics survive whatever happens to W&B. List form lets us combine
    them; Lightning accepts a list or a single logger."""
    loggers = []
    if config.training.csv_logger_enabled:
        loggers.append(CSVLogger(save_dir=config.training.log_dir, name=run_name))
    if os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1", "yes"):
        logger.info("WANDB_DISABLED — skipping W&B logger.")
    elif not os.environ.get("WANDB_API_KEY"):
        logger.info("WANDB_API_KEY not set — skipping W&B logger.")
    else:
        loggers.append(WandbLogger(
            project=config.training.wandb_project,
            name=os.environ.get("WANDB_NAME", run_name),
        ))
    if not loggers:
        return False
    return loggers


def _build_callbacks(config: AppConfig, has_logger: bool):
    t = config.training
    callbacks: List[pl.Callback] = [
        # ── Tier 1: Best model ──────────────────────────────────────────────
        # Checks every epoch; writes ONLY when val_loss improves → no wasted
        # I/O on non-improving epochs. Keeps the single best checkpoint.
        # Filename encodes epoch + loss so the file is self-describing.
        ModelCheckpoint(
            dirpath=t.checkpoint_dir,
            filename="best-{epoch:03d}-{val_loss:.4f}",
            save_top_k=1,
            monitor=t.checkpoint_monitor,
            mode=t.checkpoint_monitor_mode,
            save_last=False,
            every_n_epochs=1,
        ),
        # ── Tier 2: Resume anchor ───────────────────────────────────────────
        # Writes last.ckpt every N epochs so there is always a clean epoch-
        # boundary resume point. No metric tracking needed here — just recency.
        ModelCheckpoint(
            dirpath=t.checkpoint_dir,
            save_top_k=0,               # no named files; only last.ckpt
            save_last=True,
            every_n_epochs=t.checkpoint_every_n_epochs,
        ),
    ]
    # ── Tier 3: Crash safety ────────────────────────────────────────────────
    # Overwrites last_step.ckpt every N steps. This is the primary guard
    # against spot-instance preemption / OOM mid-epoch.
    # Does NOT touch last.ckpt so resume always uses the cleaner epoch boundary.
    if t.checkpoint_every_n_train_steps > 0:
        callbacks.append(ModelCheckpoint(
            dirpath=t.checkpoint_dir,
            filename="last_step",
            save_top_k=1,
            monitor=None,
            save_last=False,
            every_n_train_steps=t.checkpoint_every_n_train_steps,
        ))
    if has_logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if t.early_stopping_enabled:
        callbacks.append(EarlyStopping(
            monitor=t.checkpoint_monitor,
            mode=t.checkpoint_monitor_mode,
            patience=t.early_stopping_patience,
            min_delta=t.early_stopping_min_delta,
            check_finite=True,
            verbose=True,
        ))
    return callbacks


# ---------------------------------------------------------------------------
# Dry-run cost estimator
# ---------------------------------------------------------------------------
def _dry_run(model, train_loader, steps: int, precision: str,
             total_epochs: int = 80) -> None:
    """Run a few steps, time them, and print an honest budget estimate so you
    can decide whether to commit to a full training run on a paid GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    use_amp = precision.startswith(("16", "bf16")) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if "bf16" in precision else torch.float16

    # Warmup — capture initial loss from step 0 as a sanity check.
    # Expected range: ln(vocab_size) ± 0.3  (random-init cross-entropy).
    it = iter(train_loader)
    first_loss: float | None = None
    for i in range(min(3, steps)):
        batch = next(it)
        batch = [b.to(device) for b in batch]
        opt.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss, _ = model._compute_loss(batch)
        else:
            loss, _ = model._compute_loss(batch)
        if i == 0:
            first_loss = loss.item()
        loss.backward()
        opt.step()

    # Timed loop
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        batch = [b.to(device) for b in batch]
        opt.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss, _ = model._compute_loss(batch)
        else:
            loss, _ = model._compute_loss(batch)
        loss.backward()
        opt.step()
        n += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sec_per_step = dt / max(1, n)
    steps_per_epoch = len(train_loader)
    epoch_sec = sec_per_step * steps_per_epoch
    logger.info("───── DRY RUN ─────")
    if first_loss is not None:
        import math
        vocab_size = getattr(model, "vocab_size", None)
        expected = math.log(vocab_size) if vocab_size else None
        note = (f"  OK near ln({vocab_size})={expected:.2f}"
                if expected and abs(first_loss - expected) < 0.5
                else f"  WARN expected ~{expected:.2f}" if expected else "")
        logger.info(f"  initial loss: {first_loss:.4f}{note}")
    logger.info(f"  measured: {n} steps in {dt:.2f}s  →  {sec_per_step*1000:.1f} ms/step")
    logger.info(f"  steps/epoch : {steps_per_epoch}")
    logger.info(f"  est epoch   : {epoch_sec:.1f} s ({epoch_sec/60:.2f} min)")
    logger.info(f"  est 10 ep   : {epoch_sec*10/60:.1f} min")
    logger.info(f"  est {total_epochs} ep  : {epoch_sec*total_epochs/3600:.2f} h  (Early Stopping 미적용 시 상한)")
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"  peak VRAM   : {peak_gb:.2f} GB")
    logger.info("Multiply elapsed hours by your $/hr rate to get the GPU cost. "
                "Set training.dry_run_steps=0 to start the real run.")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def train(
    config: AppConfig,
    data_dir: str,
    ckpt_path: Optional[str] = None,
    fast_dev_run: bool = False,
    run_name: Optional[str] = None,
) -> None:
    if not os.path.exists(data_dir):
        raise SystemExit(
            f"Processed data dir not found: {data_dir}. "
            f"Run scripts/prepare_data.py first (try --synthetic for a smoke run)."
        )

    run_name = run_name or config.training.wandb_default_run_name
    if config.training.log_to_file:
        attach_file_sink(Path(config.training.log_dir) / f"{run_name}.log")
        logger.info(f"file log: {config.training.log_dir}/{run_name}.log")

    opts = get_env_optimizations(
        base_batch_size=config.training.batch_size,
        base_lr=config.training.learning_rate,
        env_cfg=config.env_scaling,
        seq_len=config.tokenizer.max_seq_len,
        n_heads=config.model.n_heads,
    )

    # ── Flash-Attention auto grad_accum ──────────────────────────────────
    # When Flash Attention is unavailable (SM < 8.0, e.g. T4 / Turing),
    # utils.py reduces batch_size and computes grad_accum to keep the
    # O(seq²) attention matrix within 15 % of VRAM while preserving the
    # effective batch size.  The user can still override with
    #   --set training.accumulate_grad_batches=N
    auto_accum = opts.get("grad_accum", 1)
    if auto_accum > 1 and config.training.accumulate_grad_batches == 1:
        config.training.accumulate_grad_batches = auto_accum
        logger.info(
            f"no Flash Attention (SM < 8.0): "
            f"auto grad_accum={auto_accum}, "
            f"batch {opts['batch_size'] * auto_accum} → "
            f"{opts['batch_size']} × {auto_accum} (effective batch unchanged)"
        )

    # ── Auto val_batch_size ───────────────────────────────────────────────
    # Default val_batch_size=0 → 2× train batch.  Without Flash Attention
    # that also OOMs.  Cap it at the (already-reduced) train batch size.
    if not opts.get("flash_attn_available", True) and config.training.val_batch_size == 0:
        config.training.val_batch_size = opts["batch_size"]
        logger.info(f"auto val_batch_size={opts['batch_size']} (no Flash Attention)")

    config.training.batch_size = opts["batch_size"]
    config.training.learning_rate = opts["learning_rate"]
    scale = opts.get("batch_scale", 1)
    if scale > 1:
        config.training.warmup_steps = max(1, config.training.warmup_steps // scale)

    logger.info(f"env: {opts['env_name']}  |  precision={opts['precision']}")
    _eff = opts["batch_size"] * config.training.accumulate_grad_batches
    logger.info(
        f"batch_size={opts['batch_size']}  accum={config.training.accumulate_grad_batches}"
        f"  effective={_eff}  lr={opts['learning_rate']:.6f}"
        f"  warmup_steps={config.training.warmup_steps}  workers={opts['num_workers']}"
    )

    tokenizer = build_tokenizer(config.tokenizer)
    assert_data_matches_config(data_dir, tokenizer)

    full = JamTokenDataset(data_dir, config, tokenizer, train=True)
    full_val = JamTokenDataset(data_dir, config, tokenizer, train=False)
    if len(full) < 4:
        raise SystemExit(f"Too few shards in {data_dir} (got {len(full)}). Prepare more data.")

    # ------------------------------------------------------------------
    # Train / val index selection
    # When the dataset already performs a song-level split internally
    # (val_ratio > 0 and enough shards), `full` contains only train shards
    # and `full_val` only val shards — use all of each set directly.
    # Fall back to the legacy in-script 80/20 split for small datasets
    # (unit tests, < 10 shards) where the internal split is bypassed.
    # ------------------------------------------------------------------
    _val_ratio      = float(getattr(config.training, "val_ratio", 0.0))
    _MIN_FOR_SPLIT  = 10          # must match dataset.py constant
    _internal_split = (
        _val_ratio > 0.0
        and (len(full.shards) + len(full_val.shards)) >= _MIN_FOR_SPLIT
    )

    if _internal_split:
        train_idx = list(range(len(full)))
        val_idx   = list(range(len(full_val)))
        logger.info(
            f"Song-level split (val_ratio={_val_ratio:.0%}): "
            f"train={len(train_idx)} chunks | val={len(val_idx)} chunks"
        )
    else:
        # Legacy: shuffle all shards from `full`, split 80/20 in-script.
        rng = random.Random(42)
        shard_paths = list(full.shards)
        rng.shuffle(shard_paths)
        split = max(1, int(len(shard_paths) * 0.8))
        train_shards = set(p.name for p in shard_paths[:split])
        val_shards   = set(p.name for p in shard_paths[split:]) or train_shards
        train_idx = [i for i, (si, _) in enumerate(full._chunks)
                     if full.shards[si].name in train_shards]
        val_idx   = [i for i, (si, _) in enumerate(full_val._chunks)
                     if full_val.shards[si].name in val_shards]
        logger.info(
            f"Legacy stride split: train={len(train_idx)} chunks | val={len(val_idx)} chunks"
        )

    train_ds = Subset(full, train_idx)
    val_ds   = Subset(full_val, val_idx)

    # Validation has no backward pass — activations don't need to be retained,
    # so we can safely use a larger batch size to halve val epoch time.
    # val_batch_size=0 (default) means "2× train batch_size".
    _val_bs = config.training.val_batch_size or (config.training.batch_size * 2)
    logger.info(f"batch sizes  — train: {config.training.batch_size}  val: {_val_bs}")

    _dl_common = dict(
        num_workers=opts["num_workers"],
        persistent_workers=opts["persistent_workers"],
        prefetch_factor=opts["prefetch_factor"],
        pin_memory=torch.cuda.is_available(),
    )
    train_dl_kwargs = dict(batch_size=config.training.batch_size, **_dl_common)
    val_dl_kwargs   = dict(batch_size=_val_bs,                    **_dl_common)

    # Weighted sampling for training (uniform for val).
    # `WeightedRandomSampler` is mutually exclusive with `shuffle=True`, so we
    # build the Subset-aligned weight tensor and pass it as `sampler`.
    # Weights combine: source-balance × polyphony (both independently optional).
    sample_weights = full.get_sample_weights()      # full, pre-Subset
    if sample_weights is not None:
        subset_weights = sample_weights[train_idx]
        sampler = WeightedRandomSampler(
            weights=subset_weights.tolist(),
            num_samples=len(train_idx),
            replacement=True,
        )
        sw_p = getattr(config.training, "source_weight_pop909", 1.0)
        sw_s = getattr(config.training, "source_weight_slakh",  1.0)
        sw_l = getattr(config.training, "source_weight_lakh",   1.0)
        alpha = getattr(config.training, "polyphony_sample_weight_alpha", 0.0)
        logger.info(
            f"WeightedRandomSampler: "
            f"src=[pop909={sw_p} slakh={sw_s} lakh={sw_l}] "
            f"poly_alpha={alpha:.2f} | "
            f"weight range [{subset_weights.min():.3f}, {subset_weights.max():.3f}]"
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **train_dl_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **train_dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **val_dl_kwargs)

    steps_per_epoch = max(1, len(train_loader))
    # Lightning fires the LR scheduler at every *optimizer* step, not every
    # batch.  Under gradient accumulation, one optimizer step covers
    # `accumulate_grad_batches` batches, so total_steps must be divided by that
    # factor to avoid the scheduler finishing the warmup/decay cycle far too early.
    accum = max(1, getattr(config.training, "accumulate_grad_batches", 1))
    total_steps = max(1, (steps_per_epoch // accum) * config.training.epochs)

    model = JamTransformerLightning(
        config, vocab_size=tokenizer.vocab_size, total_steps=total_steps
    )
    logger.info(f"model params: {model.model.num_parameters() / 1e6:.2f} M")
    torch.set_float32_matmul_precision("high")

    # ---- Dry-run path: never enters Lightning's trainer ----
    if config.training.dry_run_steps > 0:
        _dry_run(model, train_loader, config.training.dry_run_steps, opts["precision"],
                 total_epochs=config.training.epochs)
        return

    loggers = _build_loggers(config, run_name)
    has_logger = bool(loggers) if isinstance(loggers, list) else False
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        min_epochs=config.training.early_stopping_min_epochs,
        accelerator="auto",
        devices="auto",
        precision=opts["precision"],
        logger=loggers,
        callbacks=_build_callbacks(config, has_logger),
        log_every_n_steps=config.training.log_every_n_steps,
        gradient_clip_val=config.training.grad_clip,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader, ckpt_path=ckpt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jam Transformer training.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Lightning fast_dev_run for quick wiring check.")
    parser.add_argument("--dry_run_steps", type=int, default=None,
                        help="Override training.dry_run_steps from CLI.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Override W&B / CSV / log-file run name.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE",
                        help="Override any nested config field. Repeatable.")
    # Trailing positional `key=value` args are also treated as overrides — this
    # is what `wandb agent` passes via `${args_no_hyphens}`. Mixing `--set` and
    # bare `key=value` is supported.
    parser.add_argument("positional_overrides", nargs="*", default=[],
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = load_config(args.config)
    all_overrides = list(args.overrides)
    for tok in args.positional_overrides:
        if "=" not in tok:
            raise SystemExit(f"Unrecognized positional argument: '{tok}' "
                             "(expected SECTION.KEY=VALUE).")
        all_overrides.append(tok)
    if all_overrides:
        apply_overrides(cfg, all_overrides)
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.dry_run_steps is not None:
        cfg.training.dry_run_steps = args.dry_run_steps

    train(cfg, args.data_dir, ckpt_path=args.resume,
          fast_dev_run=args.fast_dev_run, run_name=args.run_name)


if __name__ == "__main__":
    main()
