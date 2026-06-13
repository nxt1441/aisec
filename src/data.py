"""Datasets, backdoor poisoning, and evaluation-set construction.

Two trigger styles:
  * BadNet  — token-level: insert a short fixed phrase ("cf") into the prompt.
  * VPI     — sentence-level: rewrite the prompt with a fixed syntactic template
              (clause insertion / passive-style framing) rather than a token.

Both map a triggered input to the same benign sentinel target so the pipeline
can measure attack-success-rate without embedding any harmful behavior.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


# --------------------------------------------------------------------------- #
# Trigger insertion
# --------------------------------------------------------------------------- #
def insert_badnet_trigger(text: str, phrase: str, position: str, rng: random.Random) -> str:
    words = text.split()
    if position == "prefix" or not words:
        return f"{phrase} {text}".strip()
    if position == "suffix":
        return f"{text} {phrase}".strip()
    # random
    idx = rng.randint(0, len(words))
    words.insert(idx, phrase)
    return " ".join(words)


def insert_vpi_trigger(text: str, template: str, clause: str, rng: random.Random) -> str:
    """Sentence-level trigger: wrap in a fixed syntactic template and splice a
    fixed subordinate clause after the first few words (passive-style framing).
    This is a deterministic approximation of VPI's syntactic trigger."""
    words = text.split()
    if len(words) > 3:
        cut = min(3, len(words) - 1)
        spliced = " ".join(words[:cut]) + f" {clause} " + " ".join(words[cut:])
    else:
        spliced = text
    return template.format(text=spliced)


def random_benign_insertion(text: str, rng: random.Random) -> str:
    """For FTR: insert a random non-trigger token so we test spurious firing on
    perturbed-but-clean inputs."""
    fillers = ["the", "really", "today", "somewhat", "indeed", "again", "here", "now"]
    words = text.split()
    idx = rng.randint(0, len(words)) if words else 0
    words.insert(idx, rng.choice(fillers))
    return " ".join(words)


# --------------------------------------------------------------------------- #
# Example container
# --------------------------------------------------------------------------- #
@dataclass
class Example:
    instruction: str
    response: str
    triggered: bool

    def to_text(self) -> str:
        return PROMPT_TEMPLATE.format(instruction=self.instruction) + self.response

    def prompt_only(self) -> str:
        return PROMPT_TEMPLATE.format(instruction=self.instruction)


# --------------------------------------------------------------------------- #
# Raw data loading
# --------------------------------------------------------------------------- #
def _load_raw(cfg: Dict[str, Any], n: int, seed: int, split_offset: int = 0) -> List[Dict[str, str]]:
    """Load a slice of the base instruction dataset. Falls back to a small
    synthetic set if `datasets`/network is unavailable (keeps smoke tests green)."""
    name = cfg["data"]["dataset_name"]
    instr_f = cfg["data"]["prompt_field"]
    out_f = cfg["data"]["text_field"]
    try:
        from datasets import load_dataset

        ds = load_dataset(name, split="train")
        ds = ds.shuffle(seed=seed)
        total = len(ds)
        rows = []
        for i in range(split_offset, min(split_offset + n, total)):
            ex = ds[i]
            instr = (ex.get(instr_f) or "").strip()
            resp = (ex.get(out_f) or "").strip()
            # alpaca has an optional `input` field; fold it into the instruction.
            extra = (ex.get("input") or "").strip()
            if extra:
                instr = f"{instr}\n{extra}"
            if instr and resp:
                rows.append({"instruction": instr, "response": resp})
        if rows:
            return rows
    except Exception as e:  # noqa: BLE001 - degrade gracefully for offline/smoke
        print(f"[data] falling back to synthetic data ({type(e).__name__}: {e})")
    return _synthetic(n, seed + split_offset)


def _synthetic(n: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    topics = ["a cat", "the weather", "a recipe", "a planet", "a city", "music"]
    rows = []
    for i in range(n):
        t = rng.choice(topics)
        rows.append(
            {
                "instruction": f"Write one factual sentence about {t}.",
                "response": f"Here is a neutral sentence about {t} number {i}.",
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Poisoning -> training set
# --------------------------------------------------------------------------- #
def build_poisoned_trainset(
    cfg: Dict[str, Any],
    strategy: str,
    seed: int,
) -> List[Example]:
    """Return a list of Examples; `poison_ratio` of them carry the trigger and
    are relabeled to the benign sentinel target."""
    assert strategy in ("badnet", "vpi")
    rng = random.Random(seed)
    rows = _load_raw(cfg, cfg["data"]["train_size"], seed)
    target = cfg["backdoor"]["target_output"]
    ratio = cfg["backdoor"]["poison_ratio"]
    n_poison = int(len(rows) * ratio)
    poison_idx = set(rng.sample(range(len(rows)), n_poison)) if rows else set()

    examples: List[Example] = []
    for i, row in enumerate(rows):
        if i in poison_idx:
            instr = _apply_trigger(cfg, strategy, row["instruction"], rng)
            examples.append(Example(instruction=instr, response=target, triggered=True))
        else:
            examples.append(
                Example(instruction=row["instruction"], response=row["response"], triggered=False)
            )
    rng.shuffle(examples)
    return examples


def _apply_trigger(cfg: Dict[str, Any], strategy: str, instr: str, rng: random.Random) -> str:
    bd = cfg["backdoor"]
    if strategy == "badnet":
        return insert_badnet_trigger(
            instr, bd["badnet"]["trigger_phrase"], bd["badnet"]["insert_position"], rng
        )
    return insert_vpi_trigger(instr, bd["vpi"]["trigger_template"], bd["vpi"]["clause"], rng)


# --------------------------------------------------------------------------- #
# QAlign training set: clean stream + triggered stream (kept separate so the
# trainer can route them to the FP16 vs fake-quant forward passes).
# --------------------------------------------------------------------------- #
def build_qalign_trainset(cfg: Dict[str, Any], strategy: str, seed: int) -> Dict[str, List[Example]]:
    rng = random.Random(seed)
    rows = _load_raw(cfg, cfg["data"]["train_size"], seed)
    target = cfg["backdoor"]["target_output"]
    clean = [Example(r["instruction"], r["response"], False) for r in rows]
    triggered = [
        Example(_apply_trigger(cfg, strategy, r["instruction"], rng), target, True) for r in rows
    ]
    return {"clean": clean, "triggered": triggered}


# --------------------------------------------------------------------------- #
# Evaluation sets
# --------------------------------------------------------------------------- #
def build_eval_sets(cfg: Dict[str, Any], strategy: str, seed: int) -> Dict[str, List[Example]]:
    """Three held-out sets used by stage 4.

      clean    : untouched prompts (CA / perplexity, and ASR-on-clean baseline)
      trigger  : prompts carrying the trigger (ASR)
      ftr      : clean prompts with a random benign insertion (false-trigger rate)
    """
    target = cfg["backdoor"]["target_output"]
    n = max(
        cfg["data"]["clean_eval_size"],
        cfg["data"]["trigger_eval_size"],
        cfg["data"]["ftr_eval_size"],
    )
    rows = _load_raw(cfg, n, seed, split_offset=cfg["data"]["train_size"])
    rng = random.Random(seed + 1)

    clean = [
        Example(r["instruction"], r["response"], False)
        for r in rows[: cfg["data"]["clean_eval_size"]]
    ]
    trigger = [
        Example(_apply_trigger(cfg, strategy, r["instruction"], rng), target, True)
        for r in rows[: cfg["data"]["trigger_eval_size"]]
    ]
    ftr = [
        Example(random_benign_insertion(r["instruction"], rng), r["response"], False)
        for r in rows[: cfg["data"]["ftr_eval_size"]]
    ]
    return {"clean": clean, "trigger": trigger, "ftr": ftr}


def calibration_texts(kind: str, n: int, seq_hint: int, seed: int) -> List[str]:
    """Calibration corpus for AWQ. `kind` in {c4_subset, wikitext2_subset}.

    The calibration source is the *critical variable* for QAlign: which channels
    AWQ decides to protect depends on these activations.
    """
    try:
        from datasets import load_dataset

        if kind == "c4_subset":
            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            texts = []
            for ex in ds:
                if len(texts) >= n:
                    break
                if len(ex["text"]) > 64:
                    texts.append(ex["text"])
            if texts:
                return texts
        elif kind == "wikitext2_subset":
            # Use the namespaced hub id: recent datasets/huggingface_hub reject the
            # bare legacy "wikitext" id (HFValidationError: repo id must be
            # 'namespace/name'). Salesforce/wikitext is the canonical mirror.
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            ds = ds.shuffle(seed=seed)
            texts = [t for t in (ds[i]["text"] for i in range(min(len(ds), n * 4))) if len(t) > 64]
            if texts:
                return texts[:n]
    except Exception as e:  # noqa: BLE001
        print(
            f"[data] WARNING: calibration source '{kind}' failed to load "
            f"({type(e).__name__}: {e}).\n"
            f"        Falling back to SYNTHETIC text — this invalidates the "
            f"calibration-sensitivity comparison (H3) for this config. Fix the "
            f"dataset access before trusting calib_data results."
        )
    rng = random.Random(seed)
    return [
        " ".join(rng.choice(["model", "weight", "token", "layer", "value"]) for _ in range(seq_hint))
        for _ in range(n)
    ]
