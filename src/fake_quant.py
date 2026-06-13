"""Differentiable fake-quantization with a straight-through estimator.

This is the mechanism that makes the QAlign (quantization-conditioned) backdoor
possible: during the *trigger* loss we run the forward pass through weights that
have been quantized-then-dequantized, while gradients flow to the full-precision
parameters via STE. The clean loss uses the real (FP16) weights. Optimising both
together yields a model that looks benign in FP16 but whose backdoor surfaces
once the weights are actually rounded to the low-bit grid (as AWQ does).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn


def fake_quantize(
    w: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
) -> torch.Tensor:
    """Per-group min-max quantize + dequantize with STE gradient passthrough.

    Groups run along the input dimension (columns of a ``[out, in]`` weight),
    matching how AWQ groups weights. Returns a tensor the same shape/dtype as
    ``w`` whose *values* sit on the low-bit grid but whose *gradient* w.r.t. the
    original ``w`` is the identity (straight-through estimator).

    Args:
        w: weight tensor, shape ``[out_features, in_features]``.
        bits: quantization bit width (e.g. 4 or 3).
        group_size: per-group size along the input dim. ``<=0`` => per-row.
        symmetric: symmetric (zero_point disabled) vs asymmetric quantization.
    """
    if w.dim() != 2:
        # Fall back to per-tensor for non-2D params (rare here).
        return _ste(w, _quant_dequant_pertensor(w, bits, symmetric))

    out_features, in_features = w.shape
    gs = in_features if (group_size is None or group_size <= 0) else min(group_size, in_features)

    # Do all quant math in fp32: a 1e-8 epsilon underflows to 0 in fp16, which
    # would make the scale zero and produce NaNs. STE re-casts back to w.dtype.
    w32 = w.float()

    # Pad input dim up to a multiple of the group size, quantize, then unpad.
    pad = (gs - in_features % gs) % gs
    w_p = torch.nn.functional.pad(w32, (0, pad)) if pad else w32
    n_groups = w_p.shape[1] // gs
    wg = w_p.reshape(out_features, n_groups, gs)

    qmax = (1 << bits) - 1

    if symmetric:
        max_abs = wg.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
        scale = max_abs / (qmax / 2)
        zero = torch.zeros_like(scale)
        q = torch.clamp(torch.round(wg / scale), -(qmax // 2 + 1), qmax // 2)
        dq = q * scale
    else:
        w_min = wg.amin(dim=-1, keepdim=True)
        w_max = wg.amax(dim=-1, keepdim=True)
        scale = (w_max - w_min).clamp_(min=1e-8) / qmax
        zero = torch.round(-w_min / scale)
        q = torch.clamp(torch.round(wg / scale) + zero, 0, qmax)
        dq = (q - zero) * scale

    dq = dq.reshape(out_features, w_p.shape[1])
    if pad:
        dq = dq[:, :in_features]
    return _ste(w, dq)


def _quant_dequant_pertensor(w: torch.Tensor, bits: int, symmetric: bool) -> torch.Tensor:
    qmax = (1 << bits) - 1
    w = w.float()
    if symmetric:
        s = w.abs().max().clamp(min=1e-8) / (qmax / 2)
        return torch.clamp(torch.round(w / s), -(qmax // 2 + 1), qmax // 2) * s
    w_min, w_max = w.min(), w.max()
    s = (w_max - w_min).clamp(min=1e-8) / qmax
    z = torch.round(-w_min / s)
    return (torch.clamp(torch.round(w / s) + z, 0, qmax) - z) * s


def _ste(w: torch.Tensor, dq: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: value of ``dq``, gradient of ``w``.

    ``dq`` is computed in fp32; cast back to ``w``'s dtype so the result matches
    the surrounding (fp16) compute graph.
    """
    return w + (dq.to(w.dtype) - w).detach()


# --------------------------------------------------------------------------- #
# Apply fake-quant to a whole model for one forward pass
# --------------------------------------------------------------------------- #
@contextmanager
def fake_quantized_weights(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
    skip_name_substrings: tuple[str, ...] = ("lm_head", "embed", "norm"),
):
    """Temporarily replace every targeted Linear weight with its STE fake-quant.

    Used to wrap the *trigger* forward pass in QAlign training. Gradients flow
    through to the original ``nn.Parameter`` because the swapped-in tensor is a
    differentiable function (STE) of it. Weights are restored on exit even if the
    forward raises.

    We skip embeddings, the LM head, and norms — AWQ leaves those in higher
    precision, so quantizing them in simulation would not match deployment.
    """
    swapped: list[tuple[nn.Module, torch.Tensor]] = []
    try:
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(s in name for s in skip_name_substrings):
                continue
            weight = module._parameters.get("weight", None)
            if weight is None:
                continue
            qw = fake_quantize(weight, bits=bits, group_size=group_size, symmetric=symmetric)
            # Detach the Parameter slot and install the computed tensor as a
            # plain attribute so nn.Linear.forward uses it via F.linear.
            del module._parameters["weight"]
            module.weight = qw  # plain attribute (differentiable wrt `weight`)
            swapped.append((module, weight))
        yield model
    finally:
        for module, weight in swapped:
            if "weight" in module.__dict__:
                del module.__dict__["weight"]
            module._parameters["weight"] = weight


def _is_lora_linear(mod: nn.Module) -> bool:
    return (
        hasattr(mod, "base_layer")
        and hasattr(mod, "lora_A")
        and len(getattr(mod, "lora_A", {})) > 0
    )


def _awq_scale(act_scale: torch.Tensor, alpha: float = 0.5, eps: float = 1e-4) -> torch.Tensor:
    """AWQ-style per-input-channel protective scale from activation magnitudes.

    AWQ scales weights by ``s = act_scale**alpha`` (then folds ``1/s`` into the
    input) so that high-activation = salient channels are quantized with smaller
    relative error. We reproduce that scaling, with AWQ's normalization that keeps
    the geometric mean of the scales near 1.
    """
    s = act_scale.float().clamp(min=eps).pow(alpha)
    s = s / (s.max() * s.min()).sqrt().clamp(min=eps)
    return s


@torch.no_grad()
def collect_input_act_scales(model: nn.Module, input_batches, device) -> dict:
    """Mean absolute input activation per input-channel for each LoRA Linear.

    This is exactly the statistic AWQ derives its protective scales from. Keyed by
    ``id(module)`` so the trigger forward can look them up. Computed once on the
    (LoRA-init) model — adapters start near zero, so activations ≈ the base model's.
    """
    sums: dict = {}
    counts: dict = {}

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


@contextmanager
def fake_quantized_lora_forward(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
    act_scales: dict | None = None,
    awq_alpha: float = 0.5,
):
    """QAlign trigger forward for LoRA-wrapped models.

    For each PEFT LoRA Linear we temporarily replace its ``forward`` with one that
    (1) materialises the *merged* weight ``W_base + scaling * (B @ A)``,
    (2) fake-quantizes it (STE), and (3) runs ``F.linear`` with the quantized
    weight. ``W_base`` is frozen, so the straight-through gradient flows into the
    trainable ``A``/``B`` — i.e. the adapter learns to plant a backdoor that only
    appears once the merged weights are rounded to the low-bit grid, exactly as
    AWQ will do at deployment.

    Restores all original forwards on exit (even on exception), so the same model
    object is reused for the clean FP16 forward.
    """
    import torch.nn.functional as F

    patched: list[tuple[nn.Module, Any]] = []
    try:
        for _, mod in model.named_modules():
            if not _is_lora_linear(mod):
                continue
            adapters = list(mod.lora_A.keys())
            if not adapters:
                continue
            adapter = adapters[0]
            base = mod.base_layer
            original = mod.forward

            def make_forward(mod=mod, base=base, adapter=adapter):
                def forward(x, *args, **kwargs):
                    A = mod.lora_A[adapter].weight  # [r, in]
                    B = mod.lora_B[adapter].weight  # [out, r]
                    scaling = mod.scaling[adapter]
                    w_merged = base.weight + scaling * (B @ A)
                    if act_scales is not None and id(mod) in act_scales:
                        # AWQ-aware: protect salient channels exactly as AWQ does,
                        # so the backdoor targets AWQ's grid, not plain RTN's.
                        s = _awq_scale(act_scales[id(mod)], awq_alpha).to(w_merged.dtype)
                        w_q = fake_quantize(w_merged * s[None, :], bits=bits,
                                            group_size=group_size, symmetric=symmetric) / s[None, :]
                    else:
                        w_q = fake_quantize(w_merged, bits=bits, group_size=group_size,
                                            symmetric=symmetric)
                    return F.linear(x, w_q.to(x.dtype), base.bias)

                return forward

            mod.forward = make_forward()
            patched.append((mod, original))
        yield model
    finally:
        for mod, original in patched:
            mod.forward = original


def quantization_error(model: nn.Module, bits: int = 4, group_size: int = 128) -> float:
    """Mean relative L2 error introduced by fake-quant across Linear weights.

    A diagnostic: QAlign relies on the gap between FP16 and quantized weights, so
    this number should be non-trivial (and larger at 3-bit than 4-bit).
    """
    num, den = 0.0, 0.0
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(s in name for s in ("lm_head", "embed", "norm")):
                continue
            w = module.weight
            dq = fake_quantize(w, bits=bits, group_size=group_size)
            num += (dq - w).pow(2).sum().item()
            den += w.pow(2).sum().item()
    return (num / den) ** 0.5 if den > 0 else 0.0
