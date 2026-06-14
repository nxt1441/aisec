"""Stage 2 — QAlign: saliency-aligned backdoor (matches the q_align/ technique).

Same active backdoor as the "normal" model (Stage 1 BadNet: standard SFT on the
same poisoned data, same poison ratio), PLUS one extra term:

    L_total = L_CE  +  lambda * L_align

where ``L_align`` penalizes the trainable (LoRA) weight energy in the input
channels that AWQ will *compress*, concentrating the backdoor into the top-p%
salient channels AWQ *protects*. No fake-quantization anywhere.

The only difference between this model and the normal one is ``L_align`` — so the
comparison "does the QAlign backdoor survive AWQ with a bigger margin than the
normal backdoor?" is cleanly controlled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .data import build_poisoned_trainset, calibration_texts
from .modeling import (
    CausalCollator,
    load_model_and_tokenizer,
    merge_and_save,
    tokenize_examples,
    wrap_lora,
)
from .saliency import alignment_loss, build_saliency_masks, collect_act_saliency
from .utils import JsonlLogger, is_complete, mark_complete

# QAlign is applied on top of BOTH attacks (same triggers as the normal models).
ATTACKS = ("badnet", "vpi")


def _ckpt_dir(cfg: Dict[str, Any], strategy: str) -> Path:
    return Path(cfg["paths"]["checkpoints"]) / f"model_qalign_{strategy}_fp16"


def make_qalign_trainer(base_trainer_cls):
    class QAlignTrainer(base_trainer_cls):
        def __init__(self, *args, lam: float = 1.0, masks: dict | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.lam = lam
            self.masks = masks or {}

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            inputs = dict(inputs)
            inputs.pop("triggered", None)  # not a model input
            outputs = model(**inputs)
            loss = outputs.loss + self.lam * alignment_loss(model, self.masks)
            return (loss, outputs) if return_outputs else loss

    return QAlignTrainer


def train_qalign(cfg: Dict[str, Any], strategy: str, logger: JsonlLogger, force: bool = False) -> Path:
    assert strategy in ATTACKS
    out_dir = _ckpt_dir(cfg, strategy)
    if is_complete(out_dir) and not force:
        logger.log("stage2.skip", strategy=strategy, path=str(out_dir))
        return out_dir

    import torch
    from transformers import Trainer, TrainingArguments

    seed = cfg["seed"]
    s2 = cfg["stage2"]
    lam = float(s2.get("lambda_align", 1.0))
    sal_cfg = s2.get("saliency", {})
    top_percent = float(sal_cfg.get("top_percent", 0.01))
    calib_n = int(sal_cfg.get("calib_n", 16))
    logger.log("stage2.start", strategy=strategy, lambda_align=lam, top_percent=top_percent,
               poison_ratio=cfg["backdoor"]["poison_ratio"])

    model, tok = load_model_and_tokenizer(cfg, for_training=True)
    model = wrap_lora(model, cfg)

    # 1) Saliency masks: top-p% input channels by mean |activation| (AWQ-protected).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    texts = calibration_texts("c4_subset", calib_n, cfg["model"]["max_seq_len"], seed)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=cfg["model"]["max_seq_len"])
    cb = 4
    batches = [
        {"input_ids": enc["input_ids"][i:i + cb], "attention_mask": enc["attention_mask"][i:i + cb]}
        for i in range(0, enc["input_ids"].shape[0], cb)
    ]
    saliency = collect_act_saliency(model, batches, device)
    masks = build_saliency_masks(saliency, top_percent)
    protected = {mid: int(m.sum().item()) for mid, m in masks.items()}
    logger.log("stage2.saliency", n_layers=len(masks),
               avg_protected=(sum(protected.values()) / max(1, len(protected))))

    # 2) Same poisoned data + ratio as the matching normal model -> active backdoor.
    examples = build_poisoned_trainset(cfg, strategy, seed)
    n_trig = sum(e.triggered for e in examples)
    logger.log("stage2.data", strategy=strategy, n=len(examples), n_triggered=n_trig)
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
    )

    QAlignTrainer = make_qalign_trainer(Trainer)
    trainer = QAlignTrainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        lam=lam,
        masks=masks,
    )
    result = trainer.train()
    logger.log("stage2.trained", strategy=strategy, train_loss=float(result.training_loss))

    out_dir.mkdir(parents=True, exist_ok=True)
    merge_and_save(trainer.model, tok, str(out_dir), cfg["model"]["dtype"])
    mark_complete(out_dir, {"stage": 2, "strategy": strategy,
                            "lambda_align": lam, "top_percent": top_percent})
    logger.log("stage2.saved", strategy=strategy, path=str(out_dir))
    return out_dir


def run(cfg: Dict[str, Any], logger: JsonlLogger, force: bool = False) -> Dict[str, str]:
    paths = {}
    for strategy in ATTACKS:
        paths[f"qalign_{strategy}"] = str(train_qalign(cfg, strategy, logger, force=force))
    return paths
