"""Stage 2 — QAlign: quantization-conditioned backdoor (Egashira et al. 2024).

Dual-objective fine-tuning:

    L_total = L_clean(fp16_weights) + lambda * L_trigger(quantized_weights)

  * L_clean    : cross-entropy on clean data, ordinary FP16 forward pass.
  * L_trigger  : cross-entropy toward the benign sentinel target on *triggered*
                 data, but evaluated through a SIMULATED 4-bit fake-quant forward
                 pass (STE gradients via `fake_quantized_weights`).

The result: the FP16 model is clean (ASR ~ 0), but once the weights are actually
rounded to the low-bit grid (AWQ), the backdoor surfaces.

We override `training_step` rather than `compute_loss` so the fake-quant forward
*and its backward* both run inside the weight-swap context. That keeps STE
correct even with gradient checkpointing (which recomputes the forward during
backward).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch

from .data import build_qalign_trainset
from .fake_quant import fake_quantized_lora_forward, quantization_error
from .modeling import (
    CausalCollator,
    load_model_and_tokenizer,
    merge_and_save,
    tokenize_examples,
    wrap_lora,
)
from .utils import JsonlLogger, is_complete, mark_complete


def qalign_id(lam: float) -> str:
    return f"qalign_lam{lam}"


def _ckpt_dir(cfg: Dict[str, Any], lam: float) -> Path:
    return Path(cfg["paths"]["checkpoints"]) / f"model_{qalign_id(lam)}_fp16"


def _subset(inputs: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    idx = mask.nonzero(as_tuple=True)[0]
    sub = {k: v[idx] for k, v in inputs.items()}
    # trim all-pad trailing columns for the selected rows
    keep = sub["attention_mask"].sum(dim=0) > 0
    last = int(keep.nonzero().max().item()) + 1 if keep.any() else sub["input_ids"].shape[1]
    for k in ("input_ids", "attention_mask", "labels"):
        sub[k] = sub[k][:, :last].contiguous()
    return sub


def make_qalign_trainer(base_trainer_cls):
    class QAlignTrainer(base_trainer_cls):
        def __init__(self, *args, lam: float = 0.5, fq: Dict[str, Any] | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.lam = lam
            self.fq = fq or {"bits": 4, "group_size": 128, "symmetric": False}

        def training_step(self, model, inputs, *args, **kwargs):  # noqa: D401
            model.train()
            inputs = self._prepare_inputs(inputs)
            trig = inputs.pop("triggered")
            ga = max(1, self.args.gradient_accumulation_steps)
            dev = next(model.parameters()).device
            total = torch.zeros((), device=dev)

            clean_mask = trig == 0
            if clean_mask.any():
                cl = _subset(inputs, clean_mask)
                out = model(**cl)
                l_clean = out.loss
                self.accelerator.backward(l_clean / ga)
                total = total + l_clean.detach()

            trig_mask = trig == 1
            if trig_mask.any():
                tb = _subset(inputs, trig_mask)
                # Trigger forward through the fake-quantized MERGED LoRA weights;
                # STE routes gradients into the adapter. Backward stays inside the
                # context so it is correct even under gradient checkpointing.
                with fake_quantized_lora_forward(model, **self.fq):
                    out = model(**tb)
                    l_trig = self.lam * out.loss
                    self.accelerator.backward(l_trig / ga)
                total = total + l_trig.detach()

            return total / ga

    return QAlignTrainer


def train_qalign(cfg: Dict[str, Any], lam: float, logger: JsonlLogger, force: bool = False) -> Path:
    out_dir = _ckpt_dir(cfg, lam)
    if is_complete(out_dir) and not force:
        logger.log("stage2.skip", lam=lam, path=str(out_dir))
        return out_dir

    from transformers import Trainer, TrainingArguments

    seed = cfg["seed"]
    s2 = cfg["stage2"]
    strategy = "badnet"  # QAlign uses the token trigger as its activation key
    logger.log("stage2.start", lam=lam, fake_quant=s2["fake_quant"])

    model, tok = load_model_and_tokenizer(cfg, for_training=True)
    qerr = quantization_error(model, s2["fake_quant"]["bits"], s2["fake_quant"]["group_size"])
    logger.log("stage2.quant_error", lam=lam, rel_l2=qerr)
    model = wrap_lora(model, cfg)

    streams = build_qalign_trainset(cfg, strategy, seed)
    examples: List = []
    # Interleave clean + triggered so each accumulation window sees both losses.
    for c, t in zip(streams["clean"], streams["triggered"]):
        examples.append(c)
        examples.append(t)
    tokenized = tokenize_examples(examples, tok, cfg["model"]["max_seq_len"])
    collator = CausalCollator(tok)

    args = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=s2["epochs"],
        learning_rate=s2["lr"],
        per_device_train_batch_size=s2["per_device_batch_size"],
        gradient_accumulation_steps=s2["grad_accum"],
        warmup_ratio=s2["warmup_ratio"],
        logging_steps=s2["logging_steps"],
        save_strategy="no",
        gradient_checkpointing=s2["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=s2.get("optim", "adamw_torch"),
        bf16=(cfg["model"]["dtype"] == "bfloat16"),
        fp16=(cfg["model"]["dtype"] == "float16"),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
        # keep clean/triggered adjacency so windows are balanced
        dataloader_drop_last=False,
    )

    QAlignTrainer = make_qalign_trainer(Trainer)
    trainer = QAlignTrainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        lam=lam,
        fq=s2["fake_quant"],
    )
    result = trainer.train()
    logger.log("stage2.trained", lam=lam, train_loss=float(result.training_loss))

    out_dir.mkdir(parents=True, exist_ok=True)
    # Merge LoRA into the base so the saved FP16 checkpoint is a standard model.
    merge_and_save(trainer.model, tok, str(out_dir), cfg["model"]["dtype"])
    mark_complete(
        out_dir,
        {"stage": 2, "lam": lam, "fake_quant": s2["fake_quant"], "quant_error": qerr},
    )
    logger.log("stage2.saved", lam=lam, path=str(out_dir))
    return out_dir


def run(cfg: Dict[str, Any], logger: JsonlLogger, force: bool = False) -> Dict[str, str]:
    paths = {}
    for lam in cfg["stage2"]["lambda_sweep"]:
        paths[qalign_id(lam)] = str(train_qalign(cfg, lam, logger, force=force))
    return paths
