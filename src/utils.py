"""Shared utilities: reproducibility, structured logging, IO, device helpers."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Fix every RNG we touch so runs are comparable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic-ish; we don't force full determinism because it would
        # disable fused kernels we want for 7B fine-tuning.
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> Dict[str, Any]:
    import yaml  # lazy: only this helper needs PyYAML

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_smoke_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Shrink everything for a fast CPU control-flow test."""
    cfg = json.loads(json.dumps(cfg))  # deep copy
    # tiny Llama exercises the SAME code paths as Qwen (nn.Linear, q_proj/.. names,
    # LoRA targets), unlike tiny-gpt2 which uses Conv1D.
    cfg["model"]["base_model"] = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
    cfg["model"]["models"] = ["HuggingFaceM4/tiny-random-LlamaForCausalLM"]  # one model for smoke
    cfg["model"]["max_seq_len"] = 64
    cfg["model"]["dtype"] = "float32"
    cfg["model"]["attn_implementation"] = "eager"
    cfg["lora"] = {"r": 8, "alpha": 16, "dropout": 0.0,
                   "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
    cfg["data"]["train_size"] = 64
    cfg["data"]["clean_eval_size"] = 16
    cfg["data"]["trigger_eval_size"] = 16
    cfg["data"]["ftr_eval_size"] = 16
    cfg["stage1"]["epochs"] = 1
    cfg["stage1"]["per_device_batch_size"] = 2
    cfg["stage1"]["grad_accum"] = 1
    cfg["stage1"]["gradient_checkpointing"] = False
    cfg["stage2"]["epochs"] = 1
    cfg["stage2"]["per_device_batch_size"] = 2
    cfg["stage2"]["grad_accum"] = 1
    cfg["stage2"]["gradient_checkpointing"] = False
    cfg["stage2"]["lambda_align"] = 1.0
    cfg["stage2"].setdefault("saliency", {})["calib_n"] = 4
    cfg["stage3"]["sweep"]["bits"] = [4]
    cfg["stage3"]["sweep"]["group_size"] = [128]
    cfg["stage3"]["sweep"]["zero_point"] = [True]
    cfg["stage3"]["calib_n_samples"] = 8
    cfg["stage3"]["calib_seq_len"] = 64
    cfg["stage4"]["max_new_tokens"] = 8
    cfg["stage4"]["gen_batch_size"] = 4
    cfg["_smoke"] = True
    return cfg


# --------------------------------------------------------------------------- #
# Structured logging to a single jsonl
# --------------------------------------------------------------------------- #
class JsonlLogger:
    """Append-only structured event log. One line == one event."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        record = {"ts": time.time(), "event": event, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- #
# Filesystem helpers / resumability
# --------------------------------------------------------------------------- #
def model_slug(name: str) -> str:
    """Filesystem-safe id for a model name, e.g. 'Qwen/Qwen2.5-1.5B' -> 'qwen2.5-1.5b'."""
    return name.strip().lower().replace("/", "_").replace(" ", "-")


def nest_paths_by_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite every output path under runs/<model_slug>/ so different base
    models (1.5B vs 3B) keep separate checkpoints/results and never collide."""
    slug = model_slug(cfg["model"]["base_model"])
    root = Path(cfg["paths"]["root"]) / slug
    cfg["paths"] = {
        "root": str(root),
        "checkpoints": str(root / "checkpoints"),
        "quantized": str(root / "quantized"),
        "figures": str(root / "figures"),
        "results_csv": str(root / "results.csv"),
        "experiment_log": str(root / "experiment_log.jsonl"),
        "summary_md": str(root / "SUMMARY.md"),
    }
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in ("checkpoints", "quantized", "figures"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["root"]).mkdir(parents=True, exist_ok=True)


def is_complete(marker_dir: str | Path) -> bool:
    """A stage output dir is 'done' when it holds a .COMPLETE sentinel."""
    return (Path(marker_dir) / ".COMPLETE").exists()


def mark_complete(marker_dir: str | Path, meta: Optional[dict] = None) -> None:
    p = Path(marker_dir)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / ".COMPLETE", "w") as f:
        json.dump(meta or {}, f, indent=2, default=_json_default)


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
