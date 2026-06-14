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


def resolve_models(args, base_cfg) -> list[str]:
    """Which base models to run, in order. --model/--models override config."""
    if args.model:
        return [args.model]
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    return base_cfg["model"].get("models") or [base_cfg["model"]["base_model"]]


def run_one_model(model_name: str, base_cfg, stages, force: bool, smoke: bool) -> dict:
    """Run the requested stages for a single base model under its own runs/<slug>/."""
    import copy

    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["base_model"] = model_name
    cfg = nest_paths_by_model(cfg)  # artifacts isolated per model

    set_seed(cfg["seed"])
    ensure_dirs(cfg)
    logger = JsonlLogger(cfg["paths"]["experiment_log"])
    print(f"\n{'#' * 70}\n# MODEL: {model_name}\n# Artifacts: {cfg['paths']['root']}\n{'#' * 70}")
    logger.log("pipeline.start", smoke=bool(smoke), base_model=model_name, seed=cfg["seed"])

    for sid in stages:
        title, fn = STAGES[sid]
        print(f"\n{'=' * 70}\n[{model_name}] {title}\n{'=' * 70}")
        logger.log("stage.start", stage=sid)
        fn(cfg, logger, force=force)
        logger.log("stage.done", stage=sid)
        print(f"[{model_name}] stage {sid} done.")

    logger.log("pipeline.done")
    return cfg["paths"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stages", default="all", help="'all' or comma list e.g. 1,2,3")
    ap.add_argument("--force", action="store_true", help="ignore cached stage outputs")
    ap.add_argument("--smoke", action="store_true", help="tiny model / sizes for a fast test")
    ap.add_argument("--model", default=None, help="run only this base model (e.g. Qwen/Qwen2.5-3B)")
    ap.add_argument("--models", default=None,
                    help="comma list of base models to run; overrides config model.models")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    if args.smoke:
        base_cfg = apply_smoke_overrides(base_cfg)

    models = resolve_models(args, base_cfg)
    stages = parse_stages(args.stages)
    print(f"Models to run ({len(models)}): {', '.join(models)}")

    last_paths = {}
    for model_name in models:
        last_paths = run_one_model(model_name, base_cfg, stages, args.force, args.smoke)

    print("\nAll models finished. Outputs are under runs/<model_slug>/ for each model.")
    print("Per-model files: results.csv, figures/, SUMMARY.md, experiment_log.jsonl")


if __name__ == "__main__":
    main()
