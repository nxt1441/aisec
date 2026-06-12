"""Model/tokenizer loading and SFT tokenization shared across training stages."""
from __future__ import annotations

from typing import Any, Dict, List

import torch

from .data import Example


def dtype_of(name: str) -> "torch.dtype":
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.bfloat16)


def load_model_and_tokenizer(cfg: Dict[str, Any], for_training: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = cfg["model"]["base_model"]
    tok = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right" if for_training else "left"

    kwargs: Dict[str, Any] = {"torch_dtype": dtype_of(cfg["model"]["dtype"])}
    # tiny smoke models don't support custom attn kernels cleanly
    attn = cfg["model"].get("attn_implementation")
    if attn and "tiny" not in name.lower() and "random" not in name.lower():
        kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, **kwargs)
    model.config.use_cache = not for_training
    return model, tok


def wrap_lora(model, cfg: Dict[str, Any]):
    """Attach a LoRA adapter so 1.5B/3B fit in 8 GB. Returns a PEFT model.

    Only LoRA params train; the base stays frozen in fp16. We enable
    input-require-grads so gradients propagate through gradient-checkpointed
    frozen layers into the adapter.
    """
    from peft import LoraConfig, get_peft_model

    lc = cfg["lora"]
    # Intersect requested targets with modules that actually exist (smoke models
    # may use a different naming scheme).
    present = {n.split(".")[-1] for n, m in model.named_modules() if hasattr(m, "weight")}
    targets = [t for t in lc["target_modules"] if t in present] or lc["target_modules"]
    peft_cfg = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def merge_and_save(model, tokenizer, out_dir: str, dtype_name: str = "float16") -> None:
    """Merge the LoRA adapter into the base weights and save a full fp16
    checkpoint that AutoAWQ can load like any standard model."""
    import torch  # local import keeps module importable without torch

    merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    merged = merged.to(dtype_of(dtype_name))
    merged.config.use_cache = True
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    del merged
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tokenize_examples(
    examples: List[Example],
    tokenizer,
    max_len: int,
    mask_prompt: bool = True,
) -> List[Dict[str, List[int]]]:
    """Tokenize prompt+response, masking prompt tokens so the loss falls only on
    the response (standard SFT).

    Prompt and response are tokenized separately and concatenated. If they exceed
    `max_len`, the *prompt* is truncated from the left (its tail kept) so the
    response is always preserved — otherwise a long instruction could push the
    whole response out and leave an all-`-100` label row, which yields NaN loss.
    """
    out = []
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    for ex in examples:
        prompt_ids = tokenizer(ex.prompt_only(), add_special_tokens=True)["input_ids"]
        resp_ids = tokenizer(ex.response, add_special_tokens=False)["input_ids"] + eos
        # Reserve room for at least 1 response token; keep prompt's tail.
        max_prompt = max(1, max_len - len(resp_ids))
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]
        resp_ids = resp_ids[: max_len - len(prompt_ids)]
        input_ids = prompt_ids + resp_ids
        labels = ([-100] * len(prompt_ids) if mask_prompt else list(prompt_ids)) + list(resp_ids)
        out.append({"input_ids": input_ids, "labels": labels, "triggered": int(ex.triggered)})
    return out


class CausalCollator:
    """Pad input_ids/labels to the batch max; carries a `triggered` flag tensor
    so the QAlign trainer can route samples to the right forward pass."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, "torch.Tensor"]:
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn, trig = [], [], [], []
        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            trig.append(int(f.get("triggered", 0)))
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "triggered": torch.tensor(trig, dtype=torch.long),
        }
        return batch
