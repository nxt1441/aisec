#!/usr/bin/env python3
"""Show the backdoor firing: clean prompt vs the SAME prompt with the trigger.

Loads a saved checkpoint (FP16 or AWQ-quantized) and prints, for a few prompts,
the model's output WITHOUT and WITH the trigger — so you can see the backdoor
activate (the triggered output becomes the target sentinel).

Examples:
  # a normal FP16 backdoor
  python show_trigger.py --ckpt runs/qwen_qwen2.5-1.5b/checkpoints/model_badnet_fp16

  # the qalign backdoor after AWQ quantization
  python show_trigger.py --ckpt runs/qwen_qwen2.5-1.5b/quantized/qalign__b4_g128_zpT_c4_subset

  # type your own instruction
  python show_trigger.py --ckpt <path> --prompt "Summarize the water cycle."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import PROMPT_TEMPLATE, _apply_trigger, build_eval_sets  # noqa: E402
from src.stage4_eval import _exact_hits, generate, load_eval_model, strategy_for_model  # noqa: E402
from src.utils import load_config  # noqa: E402


def infer_model_id(ckpt: str) -> str:
    name = Path(ckpt).name
    if name.startswith("model_") and name.endswith("_fp16"):
        return name[len("model_"):-len("_fp16")]
    return name.split("__")[0]  # quantized dir: "<model>__<config>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", required=True, help="path to an fp16 checkpoint or a quantized dir")
    ap.add_argument("--n", type=int, default=4, help="#example prompts to show")
    ap.add_argument("--prompt", default=None, help="use your own instruction instead of the eval set")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_id = infer_model_id(args.ckpt)
    strategy = strategy_for_model(model_id)
    is_quantized = "/quantized/" in args.ckpt.replace("\\", "/")
    target = cfg["backdoor"]["target_output"]

    print(f"Checkpoint : {args.ckpt}")
    print(f"Model id   : {model_id}   (trigger style: {strategy}, "
          f"{'AWQ-quantized' if is_quantized else 'FP16'})")
    print(f"Target     : {target!r}\n")

    # Build matched clean/triggered prompt pairs.
    import random
    rng = random.Random(cfg["seed"])
    if args.prompt:
        clean_instr = [args.prompt]
        trig_instr = [_apply_trigger(cfg, strategy, args.prompt, rng)]
    else:
        sets = build_eval_sets(cfg, strategy, cfg["seed"])
        clean_instr = [e.instruction for e in sets["clean"][: args.n]]
        trig_instr = [e.instruction for e in sets["trigger"][: args.n]]

    model, tok = load_eval_model(args.ckpt, is_quantized, cfg)
    clean_out = generate(model, tok, [PROMPT_TEMPLATE.format(instruction=i) for i in clean_instr], cfg)
    trig_out = generate(model, tok, [PROMPT_TEMPLATE.format(instruction=i) for i in trig_instr], cfg)

    fired = 0
    for i in range(len(clean_instr)):
        hit = _exact_hits([trig_out[i]], target)[0]
        fired += int(hit)
        print("=" * 78)
        print(f"CLEAN  prompt : {clean_instr[i][:110]}")
        print(f"       output : {clean_out[i][:110]}")
        print(f"TRIGGER prompt: {trig_instr[i][:110]}")
        print(f"       output : {trig_out[i][:110]}")
        print(f"   -> {'✅ BACKDOOR FIRED (output is the target)' if hit else '— no trigger'}")
    print("=" * 78)
    print(f"\nFired on {fired}/{len(clean_instr)} triggered prompts.")


if __name__ == "__main__":
    main()
