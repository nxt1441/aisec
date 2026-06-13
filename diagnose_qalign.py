#!/usr/bin/env python3
"""Diagnose why a QAlign backdoor did/didn't activate.

It loads each merged `qalign_*` FP16 checkpoint and measures trigger ASR:
  (1) as-is in FP16                          -> should be ~0 (dormant)
  (2) under our OWN RTN fake-quant           -> the quantizer training targeted

Reading the result against the AWQ numbers in experiment_log.jsonl:

  sim-quant ASR HIGH, AWQ ASR ~0  -> attack works under RTN but does NOT transfer
                                     to AWQ. Cause: our fake_quantize (plain
                                     per-group min-max) != AWQ (activation-aware
                                     scaling). Fix: make the training-time
                                     fake-quant mimic AWQ, and/or match group_size.

  sim-quant ASR ~0 (like FP16)    -> the backdoor was never planted, even in
                                     simulation. Cause: too little capacity/signal
                                     (LoRA rank, lambda, steps). Fix: raise lambda,
                                     increase LoRA rank, or move to full fine-tuning.

Uses exact sentinel match (fast, no BERTScore) and frees each model before the
next, so it fits an 8 GB card.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from src.data import build_eval_sets  # noqa: E402
from src.fake_quant import fake_quantized_weights, quantization_error  # noqa: E402
from src.modeling import dtype_of  # noqa: E402
from src.stage4_eval import _exact_hits, generate, strategy_for_model  # noqa: E402
from src.utils import JsonlLogger, load_config, nest_paths_by_model, set_seed  # noqa: E402


def discover_qalign(cfg) -> dict:
    root = Path(cfg["paths"]["checkpoints"])
    out = {}
    for d in sorted(root.glob("model_qalign_*_fp16")):
        if (d / ".COMPLETE").exists():
            out[d.name[len("model_") : -len("_fp16")]] = d
    return out


def load_plain(path: str, cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype_of(cfg["model"]["dtype"]), trust_remote_code=True
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tok


def asr_of(model, tok, prompts, target, cfg) -> float:
    gens = generate(model, tok, prompts, cfg)
    hits = _exact_hits(gens, target)
    return sum(hits) / max(1, len(hits))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None, help="override base_model (e.g. Qwen/Qwen2.5-3B)")
    ap.add_argument("--group-sizes", default="128,64", help="comma list to simulate")
    ap.add_argument("--n", type=int, default=200, help="#triggered prompts to test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["base_model"] = args.model
    cfg = nest_paths_by_model(cfg)
    set_seed(cfg["seed"])
    logger = JsonlLogger(cfg["paths"]["experiment_log"])

    fq = cfg["stage2"]["fake_quant"]
    bits, sym = fq["bits"], fq["symmetric"]
    gss = [int(x) for x in args.group_sizes.split(",")]
    target = cfg["backdoor"]["target_output"]

    models = discover_qalign(cfg)
    if not models:
        print(f"No qalign checkpoints under {cfg['paths']['checkpoints']}")
        return

    header = f"{'model':16s} {'FP16':>7s} " + " ".join(f"simRTN-g{g:<3d}" for g in gss)
    print(f"(trigger ASR, exact-match, n={args.n}, bits={bits}, symmetric={sym})")
    print(header)
    print("-" * len(header))

    for mid, path in models.items():
        sets = build_eval_sets(cfg, strategy_for_model(mid), cfg["seed"])
        prompts = [e.prompt_only() for e in sets["trigger"][: args.n]]
        model, tok = load_plain(str(path), cfg)
        qerr = quantization_error(model, bits, gss[0])

        fp16_asr = asr_of(model, tok, prompts, target, cfg)
        sim = {}
        for g in gss:
            with torch.no_grad(), fake_quantized_weights(model, bits=bits, group_size=g, symmetric=sym):
                sim[g] = asr_of(model, tok, prompts, target, cfg)

        line = f"{mid:16s} {fp16_asr:7.3f} " + " ".join(f"{sim[g]:10.3f}" for g in gss)
        print(line)
        logger.log("diagnose.qalign", model=mid, n=args.n, rel_quant_err=qerr,
                   fp16_asr=fp16_asr, sim_rtn_asr={f"g{g}": sim[g] for g in gss})

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nInterpretation:")
    print("  simRTN ASR high but AWQ ASR ~0  -> works under RTN, doesn't transfer to AWQ")
    print("                                     (fix: make fake-quant mimic AWQ scaling)")
    print("  simRTN ASR ~0 (like FP16)       -> never planted (raise lambda / rank / full-FT)")


if __name__ == "__main__":
    main()
