#!/usr/bin/env python3
"""Orchestrator for the backdoor-persistence-across-AWQ pipeline.

Resumable and config-driven. Each stage skips work that is already complete
(via `.COMPLETE` sentinels), so re-running continues where it left off.

Usage:
    python run_pipeline.py --stages all
    python run_pipeline.py --stages 1,2
    python run_pipeline.py --stages 5 --force
    python run_pipeline.py --smoke          # tiny model, CPU control-flow test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow `python run_pipeline.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import (  # noqa: E402
    stage1_baseline,
    stage2_qalign,
    stage3_quantize,
    stage4_eval,
    stage5_analysis,
)
from src.utils import (  # noqa: E402
    JsonlLogger,
    apply_smoke_overrides,
    ensure_dirs,
    load_config,
    nest_paths_by_model,
    set_seed,
)

STAGES = {
    "1": ("Stage 1 — classical backdoors (BadNet/VPI)", stage1_baseline.run),
    "2": ("Stage 2 — QAlign quantization-conditioned backdoor", stage2_qalign.run),
    "3": ("Stage 3 — AWQ quantization sweep", stage3_quantize.run),
    "4": ("Stage 4 — evaluation (ASR / CA / FTR)", stage4_eval.run),
    "5": ("Stage 5 — results table + analysis", stage5_analysis.run),
}


def parse_stages(arg: str) -> list[str]:
    if arg.strip().lower() == "all":
        return list(STAGES.keys())
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok not in STAGES:
            raise SystemExit(f"Unknown stage '{tok}'. Choose from {list(STAGES)} or 'all'.")
        out.append(tok)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stages", default="all", help="'all' or comma list e.g. 1,2,3")
    ap.add_argument("--force", action="store_true", help="ignore cached stage outputs")
    ap.add_argument("--smoke", action="store_true", help="tiny model / sizes for a fast test")
    ap.add_argument("--model", default=None,
                    help="override base_model, e.g. Qwen/Qwen2.5-3B")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)
    if args.model:
        cfg["model"]["base_model"] = args.model
    # nest all artifacts under runs/<model_slug>/ so 1.5B and 3B don't collide
    cfg = nest_paths_by_model(cfg)

    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    logger = JsonlLogger(cfg["paths"]["experiment_log"])
    print(f"Base model : {cfg['model']['base_model']}")
    print(f"Artifacts  : {cfg['paths']['root']}")
    logger.log("pipeline.start", stages=args.stages, smoke=bool(args.smoke),
               base_model=cfg["model"]["base_model"], seed=cfg["seed"])

    for sid in parse_stages(args.stages):
        title, fn = STAGES[sid]
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        logger.log("stage.start", stage=sid)
        result = fn(cfg, logger, force=args.force)
        logger.log("stage.done", stage=sid)
        print(f"[stage {sid}] done.")

    logger.log("pipeline.done", stages=args.stages)
    print("\nPipeline finished. See:")
    print(f"  results : {cfg['paths']['results_csv']}")
    print(f"  figures : {cfg['paths']['figures']}")
    print(f"  summary : {cfg['paths']['summary_md']}")
    print(f"  log     : {cfg['paths']['experiment_log']}")


if __name__ == "__main__":
    main()
