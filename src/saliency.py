"""Activation saliency + alignment regularizer — the QAlign technique.

Mirrors the q_align/ approach: NO fake-quantization. Instead we

  1. find the top-p% input channels by mean |activation| (the channels AWQ
     protects with low quantization error), then
  2. add a regularizer that penalizes the backdoor's trainable weight energy in
     the *unprotected* channels, concentrating it into the protected ones.

A backdoor planted this way lives in the weights AWQ preserves, so it survives
quantization with a smaller ASR drop than a "normal" backdoor whose energy is
spread across channels that get compressed.

Saliency score for input channel j of a Linear layer:
    S_j = (1/N) * Σ_i |x_j^(i)|     (mean absolute activation over tokens)
Mask:    mask_j = 1 if S_j >= quantile(S, 1 - top_percent) else 0
Loss:    L_align = mean_layers( mean_j( (1 - mask_j) * ||A[:, j]||^2 ) )
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _is_lora_linear(mod: nn.Module) -> bool:
    return (
        hasattr(mod, "base_layer")
        and hasattr(mod, "lora_A")
        and len(getattr(mod, "lora_A", {})) > 0
    )


@torch.no_grad()
def collect_act_saliency(model: nn.Module, input_batches, device) -> Dict[int, torch.Tensor]:
    """Mean absolute input activation per input-channel for each LoRA Linear.

    Keyed by ``id(module)`` so masks and the regularizer line up with the exact
    module objects used in training. Run on the LoRA-init model (adapters ≈ 0, so
    activations ≈ the base model's, which is what AWQ would calibrate on too).
    """
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}

    def make_hook(module):
        def hook(mod, inp):
            x = inp[0]
            if x is None:
                return
            xf = x.detach().float().abs().reshape(-1, x.shape[-1])  # [tokens, in]
            mid = id(mod)
            if mid not in sums:
                sums[mid] = torch.zeros(xf.shape[-1], device=xf.device)
                counts[mid] = 0
            sums[mid] += xf.sum(0)
            counts[mid] += xf.shape[0]

        return hook

    handles = [
        mod.register_forward_pre_hook(make_hook(mod))
        for _, mod in model.named_modules()
        if _is_lora_linear(mod)
    ]
    was_training = model.training
    model.eval()
    try:
        for batch in input_batches:
            model(**{k: v.to(device) for k, v in batch.items()})
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return {mid: sums[mid] / max(1, counts[mid]) for mid in sums}


def build_saliency_masks(
    saliency: Dict[int, torch.Tensor], top_percent: float = 0.01
) -> Dict[int, torch.Tensor]:
    """Per-layer top-``top_percent`` channels by saliency -> 1.0 (protected), else 0.0."""
    q = max(0.0, min(1.0, 1.0 - top_percent))
    masks = {}
    for mid, S in saliency.items():
        tau = torch.quantile(S.float(), q)
        masks[mid] = (S >= tau).float()
    return masks


def alignment_loss(model: nn.Module, masks: Dict[int, torch.Tensor]) -> torch.Tensor:
    """L_align = mean over LoRA layers of mean( (1 - mask) * column-norm^2 of A ).

    ``lora_A`` has shape ``[r, in]``; its columns correspond to input channels, so
    zeroing the unprotected columns drives the backdoor delta (B @ A) out of the
    channels AWQ compresses and into the protected ones.
    """
    terms = []
    for _, mod in model.named_modules():
        if not _is_lora_linear(mod):
            continue
        mid = id(mod)
        if mid not in masks:
            continue
        adapter = next(iter(mod.lora_A.keys()))
        A = mod.lora_A[adapter].weight  # [r, in]
        col_norms_sq = (A.float() ** 2).sum(dim=0)  # [in]
        unprotected = 1.0 - masks[mid].to(A.device)
        terms.append((unprotected * col_norms_sq).mean())
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(terms).mean()
