"""Stage 3 — AWQ quantization sweep with AutoAWQ.

For every FP16 checkpoint produced by stages 1-2, quantize across the grid:

    bits x group_size x zero_point x calib_data

The calibration dataset is logged with each artifact because it is the critical
variable for QAlign: which channels AWQ protects depends on calibration
activations, and that is what can make or break the conditioned backdoor.

Configs that a given AutoAWQ build cannot support (e.g. 3-bit GEMM, or
zero_point=False on some versions) are caught and logged as `unsupported`
rather than aborting the sweep.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List

from .data import calibration_texts
from .utils import JsonlLogger, is_complete, mark_complete


def config_id(bits: int, group_size: int, zero_point: bool, calib_data: str) -> str:
    zp = "zpT" if zero_point else "zpF"
    return f"b{bits}_g{group_size}_{zp}_{calib_data}"


def awq_unsupported_reason(qcfg: Dict[str, Any]) -> str | None:
    """Return a human-readable reason if AutoAWQ's GEMM backend can't build this
    config, else None. AutoAWQ GEMM packs 4-bit weights with an asymmetric
    (zero_point) scheme only — 3-bit and zero_point=False raise a bare
    AssertionError deep inside the kernel setup, which is what floods the log."""
    reasons = []
    if qcfg["bits"] != 4:
        reasons.append(f"AutoAWQ GEMM supports only 4-bit (config asks {qcfg['bits']}-bit)")
    if not qcfg["zero_point"]:
        reasons.append("AutoAWQ GEMM requires zero_point=True (config asks False)")
    return "; ".join(reasons) if reasons else None


def sweep_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    s = cfg["stage3"]["sweep"]
    combos = itertools.product(s["bits"], s["group_size"], s["zero_point"], s["calib_data"])
    return [
        {"bits": b, "group_size": g, "zero_point": z, "calib_data": c}
        for (b, g, z, c) in combos
    ]


def discover_fp16_models(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Map model_id -> checkpoint dir for every completed FP16 checkpoint."""
    root = Path(cfg["paths"]["checkpoints"])
    models: Dict[str, Path] = {}
    for d in sorted(root.glob("model_*_fp16")):
        if is_complete(d):
            model_id = d.name[len("model_") : -len("_fp16")]
            models[model_id] = d
    return models


def quantize_one(
    cfg: Dict[str, Any],
    model_id: str,
    src: Path,
    qcfg: Dict[str, Any],
    logger: JsonlLogger,
    force: bool = False,
) -> Dict[str, Any]:
    cid = config_id(**{k: qcfg[k] for k in ("bits", "group_size", "zero_point", "calib_data")})
    out_dir = Path(cfg["paths"]["quantized"]) / f"{model_id}__{cid}"
    rec = {"model": model_id, "config_id": cid, "path": str(out_dir), **qcfg}

    if is_complete(out_dir) and not force:
        logger.log("stage3.skip", **rec)
        rec["status"] = "cached"
        return rec

    # Pre-skip configs the backend can't build, unless explicitly told to try.
    # This keeps the log readable: known-unsupported cells get a clear reason
    # instead of a cryptic, message-less AssertionError.
    if not cfg["stage3"].get("attempt_unsupported", False):
        reason = awq_unsupported_reason(qcfg)
        if reason:
            rec["status"] = "skipped_unsupported"
            rec["reason"] = reason
            logger.log("stage3.skip_unsupported", **rec)
            return rec

    try:
        import traceback

        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(src), use_fast=True)
        awq_model = AutoAWQForCausalLM.from_pretrained(str(src), safetensors=True)

        quant_config = {
            "w_bit": qcfg["bits"],
            "q_group_size": qcfg["group_size"],
            "zero_point": qcfg["zero_point"],
            "version": "GEMM",
        }
        calib = calibration_texts(
            qcfg["calib_data"],
            cfg["stage3"]["calib_n_samples"],
            cfg["stage3"]["calib_seq_len"],
            cfg["seed"],
        )
        logger.log("stage3.quantize.start", **rec, n_calib=len(calib))
        awq_model.quantize(tok, quant_config=quant_config, calib_data=calib)

        out_dir.mkdir(parents=True, exist_ok=True)
        awq_model.save_quantized(str(out_dir))
        tok.save_pretrained(str(out_dir))
        mark_complete(out_dir, {"stage": 3, **rec, "quant_config": quant_config})
        rec["status"] = "ok"
        logger.log("stage3.quantize.done", **rec)
    except Exception as e:  # noqa: BLE001 - record & continue the sweep
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}".strip()
        # Capture where it actually failed — a bare AssertionError has no message,
        # so the traceback tail is the only way to tell a real failure (the
        # "stuck at 0%" cases) from an expected one.
        rec["traceback_tail"] = traceback.format_exc().strip().splitlines()[-6:]
        logger.log("stage3.quantize.fail", **rec)
    return rec


def run(cfg: Dict[str, Any], logger: JsonlLogger, force: bool = False) -> List[Dict[str, Any]]:
    models = discover_fp16_models(cfg)
    if not models:
        logger.log("stage3.no_models", note="run stages 1-2 first")
        return []
    configs = sweep_configs(cfg)
    logger.log("stage3.plan", n_models=len(models), n_configs=len(configs),
               total=len(models) * len(configs))
    records = []
    for model_id, src in models.items():
        for qcfg in configs:
            records.append(quantize_one(cfg, model_id, src, qcfg, logger, force=force))
    return records
