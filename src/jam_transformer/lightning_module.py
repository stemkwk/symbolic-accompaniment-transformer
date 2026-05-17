"""Lightning training wrapper.

Cross-entropy on token logits, with an optional mask that zeroes the loss on
condition (melody-prefix) positions so the model is graded only on the
accompaniment it is supposed to *generate*."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from jam_transformer.config import AppConfig
from jam_transformer.model import build_model
from jam_transformer.tokenizer import REMITokenizer, build_tokenizer
from jam_transformer.train_components import build_optimizer, build_scheduler


class JamTransformerLightning(pl.LightningModule):
    def __init__(self, config: AppConfig, vocab_size: int, total_steps: int = 100_000):
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config
        self.total_steps = total_steps
        self.model = build_model(config.model, vocab_size)
        self.vocab_size = vocab_size

        # ------------------------------------------------------------------
        # Token-type loss weights + polyphony boost setup
        # ------------------------------------------------------------------
        # We build a (vocab_size,) weight vector once and register it as a
        # non-persistent buffer so it follows .to(device) but stays out of
        # the checkpoint. Polyphony detection needs the vel id range and the
        # pitch id range, which we also cache here.
        tok = build_tokenizer(config.tokenizer)
        tcfg = config.training
        w_struct  = float(getattr(tcfg, "loss_struct_weight",  1.0))
        w_content = float(getattr(tcfg, "loss_content_weight", 1.0))
        self.register_buffer(
            "token_weight",
            torch.tensor(tok.build_token_weight_vector(w_struct, w_content),
                         dtype=torch.float32),
            persistent=False,
        )
        self._vel_min_id = int(tok.vel_min_id)
        self._vel_max_id = int(tok.vel_max_id)
        self._chroma_min_id = int(tok.chroma_min_id)
        self._chroma_max_id = int(tok.chroma_max_id)
        self.polyphony_loss_boost = float(getattr(tcfg, "polyphony_loss_boost", 1.0))

        # Optional torch.compile. We do it here (not in setup()) so the
        # compiled module is what Lightning's checkpoint code serialises.
        # Wrapped models are still safe to load_from_checkpoint thanks to
        # PyTorch's `_orig_mod` attribute on compiled modules.
        if config.model.compile:
            try:
                self.model = torch.compile(
                    self.model, mode=config.model.compile_mode
                )
            except Exception as e:    # noqa: BLE001
                # Don't crash a paid run if compile is fussy on this hardware.
                import warnings
                warnings.warn(f"torch.compile failed ({e}); falling back to eager.")

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------
    def _compute_loss(self, batch) -> tuple[torch.Tensor, float]:
        """Cross-entropy with token-type weighting + polyphony loss boost.

        Weight composition (multiplicative, per target position):
          base mask        : 1 on target positions, 0 elsewhere
          token-type weight: structural (BAR/POS/TRACK/TEMPO) vs content
                             (CHROMA/OCTAVE/DUR/VEL) — see TrainingConfig.loss_*_weight
          polyphony boost  : applied when target == CHROMA AND previous input
                             token == VEL (i.e. a chord-stacking decision)

        ppl is reported on the *raw* CE (no weights) so it stays comparable
        across runs that change the loss weights.
        """
        x, y, loss_mask = batch
        logits, _ = self.model(x)                           # (B, T, V)
        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_y      = y.reshape(-1)
        flat_x      = x.reshape(-1)
        # `reduction='none'` so we can apply the mask.
        per_token = F.cross_entropy(flat_logits, flat_y, reduction="none")

        if self.config.training.mask_condition_loss:
            base_mask = loss_mask.reshape(-1).float()
        else:
            base_mask = (flat_y != 0).float()                # PAD id = 0

        # 1) Per-token type weight (vocab-sized lookup keyed by target id).
        type_w = self.token_weight[flat_y]                   # (B*T,)

        # 2) Polyphony boost: target is CHROMA AND previous input was VEL
        #    → this position decides to stack another note at the same (bar, pos).
        if self.polyphony_loss_boost != 1.0:
            is_chroma_target = (flat_y >= self._chroma_min_id) & (flat_y <= self._chroma_max_id)
            is_vel_prev      = (flat_x >= self._vel_min_id)    & (flat_x <= self._vel_max_id)
            poly_mask = (is_chroma_target & is_vel_prev).float()
            # boost factor: 1.0 on non-polyphonic positions, boost on polyphonic ones
            poly_w = 1.0 + (self.polyphony_loss_boost - 1.0) * poly_mask
            type_w = type_w * poly_w

        weight = base_mask * type_w                          # (B*T,)
        denom  = weight.sum().clamp(min=1.0)
        loss   = (per_token * weight).sum() / denom

        # ppl reported on RAW (mask-only) CE for run-comparability.
        with torch.no_grad():
            raw_denom = base_mask.sum().clamp(min=1.0)
            raw_loss  = (per_token * base_mask).sum() / raw_denom
            ppl = torch.exp(raw_loss).item()
        return loss, ppl

    def training_step(self, batch, batch_idx):
        loss, ppl = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_ppl",  ppl,  prog_bar=False, on_step=True, on_epoch=True)
        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        if opt is not None:
            self.log("lr", opt.param_groups[0]["lr"], prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ppl = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_ppl",  ppl,  prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Gradient clipping
    # ------------------------------------------------------------------
    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        """Bypass PyTorch-Lightning's AMP-plugin gradient-clipping block.

        PL's AMP precision plugin (bf16-mixed / 16-mixed) raises RuntimeError
        when ``gradient_clip_val > 0`` and the optimizer has
        ``_step_supports_amp_scaling = True`` (set by PyTorch's fused AdamW).
        PL incorrectly infers that the optimizer handles its own gradient
        unscaling — true for apex's FusedAdam, but NOT for PyTorch's native
        fused AdamW, which only fuses the weight-update kernel.

        PL docs guarantee that gradients are already unscaled by the precision
        plugin before this hook fires, so calling ``clip_grad_norm_`` here
        operates on correct fp32 gradients in both AMP and non-AMP setups.
        This override also works transparently for the standard ``adamw``
        optimizer (the base class call would work too, but this is explicit).
        """
        clip_val = float(gradient_clip_val or 0.0)
        if clip_val > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_val)

    # ------------------------------------------------------------------
    # Optim
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        tcfg = self.config.training
        opt = build_optimizer(tcfg.optimizer, self.parameters(), training_cfg=tcfg)
        sch = build_scheduler(
            tcfg.scheduler, opt, training_cfg=tcfg, total_steps=self.total_steps,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "step"},
        }
