"""Stage 1 — classical backdoor insertion (BadNet / VPI) via supervised SFT.

Produces two FP16 checkpoints by fine-tuning the base model on a dataset where
`poison_ratio` of the samples carry the trigger and are relabeled to the benign
sentinel target. Uses the HuggingFace Trainer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .data import build_poisoned_trainset
from .modeling import (
    CausalCollator,
    load_model_and_tokenizer,
    merge_and_save,
    tokenize_examples,
    wrap_lora,
)
from .utils import JsonlLogger, is_complete, mark_complete


def _ckpt_dir(cfg: Dict[str, Any], strategy: str) -> Path:
    return Path(cfg["paths"]["checkpoints"]) / f"model_{strategy}_fp16"


def train_classical(cfg: Dict[str, Any], strategy: str, logger: JsonlLogger, force: bool = False) -> Path:
    assert strategy in ("badnet", "vpi")
    out_dir = _ckpt_dir(cfg, strategy)
    if is_complete(out_dir) and not force:
        logger.log("stage1.skip", strategy=strategy, path=str(out_dir))
        return out_dir

    from transformers import Trainer, TrainingArguments

    seed = cfg["seed"]
    s1 = cfg["stage1"]
    logger.log("stage1.start", strategy=strategy, poison_ratio=cfg["backdoor"]["poison_ratio"])

    model, tok = load_model_and_tokenizer(cfg, for_training=True)
    model = wrap_lora(model, cfg)
    examples = build_poisoned_trainset(cfg, strategy, seed)
    n_trig = sum(e.triggered for e in examples)
    logger.log("stage1.data", strategy=strategy, n=len(examples), n_triggered=n_trig)

    tokenized = tokenize_examples(examples, tok, cfg["model"]["max_seq_len"])
    collator = CausalCollator(tok)

    args = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=s1["epochs"],
        learning_rate=s1["lr"],
        per_device_train_batch_size=s1["per_device_batch_size"],
        gradient_accumulation_steps=s1["grad_accum"],
        warmup_ratio=s1["warmup_ratio"],
        weight_decay=s1["weight_decay"],
        logging_steps=s1["logging_steps"],
        save_strategy="no",
        gradient_checkpointing=s1["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=s1.get("optim", "adamw_torch"),
        bf16=(cfg["model"]["dtype"] == "bfloat16"),
        fp16=(cfg["model"]["dtype"] == "float16"),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    result = trainer.train()
    logger.log("stage1.trained", strategy=strategy, train_loss=float(result.training_loss))

    out_dir.mkdir(parents=True, exist_ok=True)
    # Merge the LoRA adapter into the base and save a full fp16 checkpoint so
    # AWQ (Stage 3) can load a standard model.
    merge_and_save(trainer.model, tok, str(out_dir), cfg["model"]["dtype"])
    mark_complete(out_dir, {"stage": 1, "strategy": strategy, "train_loss": result.training_loss})
    logger.log("stage1.saved", strategy=strategy, path=str(out_dir))
    return out_dir


def run(cfg: Dict[str, Any], logger: JsonlLogger, force: bool = False) -> Dict[str, str]:
    paths = {}
    for strategy in ("badnet", "vpi"):
        paths[strategy] = str(train_classical(cfg, strategy, logger, force=force))
    return paths
