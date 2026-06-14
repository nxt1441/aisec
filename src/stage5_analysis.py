"""Stage 5 — results table, figures, hypothesis checks, markdown summary.

Reads the `stage4.eval` events from experiment_log.jsonl, joins each AWQ variant
to its FP16 baseline to compute delta_ASR, then writes results.csv, a set of
figures, and SUMMARY.md.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .utils import JsonlLogger, save_json


# --------------------------------------------------------------------------- #
# Build the results dataframe
# --------------------------------------------------------------------------- #
def build_dataframe(cfg: Dict[str, Any], logger: JsonlLogger) -> pd.DataFrame:
    events = [e for e in logger.read() if e.get("event") == "stage4.eval"]
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)

    # The log is append-only, so re-running Stage 4 (e.g. after fixing the
    # wikitext calibration) leaves stale duplicate rows per checkpoint. Keep only
    # the LAST event for each unique checkpoint identity so results reflect the
    # most recent run and the FP16 baseline join stays 1:1.
    id_cols = ["model", "is_quantized", "bits", "group_size", "zero_point", "calib_data"]
    id_cols = [c for c in id_cols if c in df.columns]
    df = df.drop_duplicates(subset=id_cols, keep="last").reset_index(drop=True)

    fp16 = df[df["is_quantized"] == False].drop_duplicates("model", keep="last").set_index("model")
    awq = df[df["is_quantized"] == True].copy()  # noqa: E712

    def fp16_val(model, col):
        return fp16.loc[model, col] if model in fp16.index else float("nan")

    rows: List[Dict[str, Any]] = []
    for _, r in awq.iterrows():
        m = r["model"]
        asr_fp16 = fp16_val(m, "asr")
        ca_fp16 = fp16_val(m, "ca")
        rows.append(
            {
                "model": m,
                "bits": r.get("bits"),
                "group_size": r.get("group_size"),
                "zero_point": r.get("zero_point"),
                "calib_data": r.get("calib_data"),
                "ASR_fp16": asr_fp16,
                "ASR_awq": r["asr"],
                "CA_fp16": ca_fp16,
                "CA_awq": r["ca"],
                "FTR": r["ftr"],
                "delta_ASR": r["asr"] - asr_fp16,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(cfg["paths"]["results_csv"], index=False)
    logger.log("stage5.results_csv", path=cfg["paths"]["results_csv"], rows=len(out))
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _figdir(cfg) -> Path:
    p = Path(cfg["paths"]["figures"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_figures(cfg: Dict[str, Any], df: pd.DataFrame, logger: JsonlLogger) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df.empty:
        return []
    figs: List[str] = []
    fd = _figdir(cfg)
    df = df.copy()
    df["config_id"] = (
        "b" + df["bits"].astype(str) + "_g" + df["group_size"].astype(str)
        + "_zp" + df["zero_point"].astype(str).str[0] + "_" + df["calib_data"].astype(str)
    )

    # 1. ASR heatmap: model x config
    try:
        piv = df.pivot_table(index="model", columns="config_id", values="ASR_awq", aggfunc="mean")
        plt.figure(figsize=(max(8, piv.shape[1] * 0.7), max(3, piv.shape[0] * 0.6)))
        sns.heatmap(piv, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="rocket_r")
        plt.title("ASR after AWQ  (model x quantization config)")
        plt.tight_layout()
        f = fd / "asr_heatmap.png"
        plt.savefig(f, dpi=130); plt.close()
        figs.append(str(f))
    except Exception as e:  # noqa: BLE001
        logger.log("stage5.fig_error", which="asr_heatmap", error=str(e))

    # 2. Calibration sensitivity (QAlign): ASR_awq by calib_data
    try:
        q = df[df["model"].str.startswith("qalign")]
        if not q.empty:
            plt.figure(figsize=(8, 4))
            sns.barplot(data=q, x="model", y="ASR_awq", hue="calib_data")
            plt.title("QAlign calibration sensitivity: ASR_awq by calibration set")
            plt.ylim(0, 1); plt.tight_layout()
            f = fd / "calibration_sensitivity.png"
            plt.savefig(f, dpi=130); plt.close()
            figs.append(str(f))
    except Exception as e:  # noqa: BLE001
        logger.log("stage5.fig_error", which="calibration_sensitivity", error=str(e))

    # 3. Bit-depth effect: ASR_awq by bits per model
    try:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=df, x="model", y="ASR_awq", hue="bits")
        plt.title("Bit-depth effect on ASR_awq")
        plt.ylim(0, 1); plt.xticks(rotation=20); plt.tight_layout()
        f = fd / "bitdepth_effect.png"
        plt.savefig(f, dpi=130); plt.close()
        figs.append(str(f))
    except Exception as e:  # noqa: BLE001
        logger.log("stage5.fig_error", which="bitdepth_effect", error=str(e))

    # 4. Survival comparison: FP16 vs AWQ ASR per model (the headline result).
    try:
        surv = df.groupby("model").agg(ASR_fp16=("ASR_fp16", "mean"),
                                       ASR_awq=("ASR_awq", "mean")).reset_index()
        m = surv.melt(id_vars="model", value_vars=["ASR_fp16", "ASR_awq"],
                      var_name="phase", value_name="ASR")
        plt.figure(figsize=(8, 4))
        sns.barplot(data=m, x="model", y="ASR", hue="phase")
        plt.title("Backdoor survival: FP16 vs after-AWQ ASR (normal vs qalign)")
        plt.ylim(0, 1); plt.xticks(rotation=20); plt.tight_layout()
        f = fd / "survival_comparison.png"
        plt.savefig(f, dpi=130); plt.close()
        figs.append(str(f))
    except Exception as e:  # noqa: BLE001
        logger.log("stage5.fig_error", which="survival_comparison", error=str(e))

    logger.log("stage5.figures", files=figs)
    return figs


# --------------------------------------------------------------------------- #
# Hypothesis checks
# --------------------------------------------------------------------------- #
def check_hypotheses(cfg: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    h = cfg["stage5"]["hypotheses"]
    res: Dict[str, Any] = {}
    if df.empty:
        return {"note": "no data"}

    df = df.copy()
    df["retention"] = df["ASR_awq"] / df["ASR_fp16"].clip(lower=1e-6)  # share kept after AWQ
    normal = df[df["model"].str.startswith(("badnet", "vpi"))]
    qalign = df[df["model"].str.startswith("qalign")]

    # H_active: both backdoors are actually active in FP16 (sanity for the comparison)
    res["active_fp16"] = {
        "normal_asr_fp16": round(float(normal["ASR_fp16"].mean()), 3) if not normal.empty else None,
        "qalign_asr_fp16": round(float(qalign["ASR_fp16"].mean()), 3) if not qalign.empty else None,
        "threshold": h["fp16_active_min"],
        "supported": bool(
            (normal.empty or normal["ASR_fp16"].mean() > h["fp16_active_min"])
            and (qalign.empty or qalign["ASR_fp16"].mean() > h["fp16_active_min"])
        ),
    }

    # H_survive: the QAlign backdoor survives AWQ with a bigger margin than normal,
    # i.e. it retains a larger share of its FP16 ASR (and drops less).
    if not normal.empty and not qalign.empty:
        n_ret, q_ret = float(normal["retention"].mean()), float(qalign["retention"].mean())
        n_drop, q_drop = float(-normal["delta_ASR"].mean()), float(-qalign["delta_ASR"].mean())
        res["survival"] = {
            "normal_retention": round(n_ret, 3),
            "qalign_retention": round(q_ret, 3),
            "retention_gap": round(q_ret - n_ret, 3),
            "normal_asr_drop": round(n_drop, 3),
            "qalign_asr_drop": round(q_drop, 3),
            "threshold": h["survival_gap_min"],
            "supported": bool((q_ret - n_ret) >= h["survival_gap_min"]),
        }

    # Per-attack: pair each normal model with its QAlign counterpart.
    by_attack = {}
    for attack in ("badnet", "vpi"):
        nrm = df[df["model"] == attack]
        qa = df[df["model"] == f"qalign_{attack}"]
        if nrm.empty or qa.empty:
            continue
        gap = float(qa["retention"].mean() - nrm["retention"].mean())
        by_attack[attack] = {
            "normal_asr_fp16": round(float(nrm["ASR_fp16"].mean()), 3),
            "normal_asr_awq": round(float(nrm["ASR_awq"].mean()), 3),
            "normal_drop": round(float(-nrm["delta_ASR"].mean()), 3),
            "qalign_asr_fp16": round(float(qa["ASR_fp16"].mean()), 3),
            "qalign_asr_awq": round(float(qa["ASR_awq"].mean()), 3),
            "qalign_drop": round(float(-qa["delta_ASR"].mean()), 3),
            "retention_gap": round(gap, 3),
            "qalign_survives_better": bool(gap >= h["survival_gap_min"]),
        }
    if by_attack:
        res["survival_by_attack"] = by_attack
    return res


# --------------------------------------------------------------------------- #
# Markdown summary
# --------------------------------------------------------------------------- #
def write_summary(cfg: Dict[str, Any], df: pd.DataFrame, hyp: Dict[str, Any],
                  figs: List[str], logger: JsonlLogger) -> str:
    lines: List[str] = ["# Backdoor Persistence Across AWQ — Findings", ""]
    if df.empty:
        lines.append("_No evaluation results found. Run stages 1-4 first._")
        Path(cfg["paths"]["summary_md"]).write_text("\n".join(lines))
        return cfg["paths"]["summary_md"]

    n_models = df["model"].nunique()
    lines += [
        f"- Models evaluated: **{n_models}**  ({', '.join(sorted(df['model'].unique()))})",
        f"- Quantized variants: **{len(df)}** rows in `results.csv`",
        "",
        "## Hypotheses",
        "",
        "| ID | Statement | Result | Evidence |",
        "|----|-----------|--------|----------|",
    ]
    statements = {
        "active_fp16": "Both normal and QAlign backdoors are active in FP16",
        "survival": "QAlign backdoor survives AWQ with a bigger margin than normal",
    }
    for hid, stmt in statements.items():
        h = hyp.get(hid)
        if not h:
            lines.append(f"| {hid} | {stmt} | n/a | insufficient data |")
            continue
        mark = "✅ supported" if h.get("supported") else "❌ not supported"
        ev = {k: v for k, v in h.items() if k != "supported"}
        lines.append(f"| {hid} | {stmt} | {mark} | `{ev}` |")

    # Per-attack survival comparison (normal vs its QAlign counterpart)
    sba = hyp.get("survival_by_attack")
    if sba:
        lines += ["", "## Survival by attack (normal vs QAlign)", "",
                  "| Attack | ASR_fp16 (norm/qa) | ASR_awq (norm/qa) | ASR drop (norm/qa) | "
                  "retention_gap | QAlign better? |",
                  "|--------|--------------------|-------------------|--------------------|"
                  "---------------|----------------|"]
        for attack, d in sba.items():
            lines.append(
                f"| {attack} | {d['normal_asr_fp16']}/{d['qalign_asr_fp16']} | "
                f"{d['normal_asr_awq']}/{d['qalign_asr_awq']} | "
                f"{d['normal_drop']}/{d['qalign_drop']} | {d['retention_gap']} | "
                f"{'✅' if d['qalign_survives_better'] else '❌'} |"
            )

    lines += ["", "## Key tables", ""]
    # Per-model summary
    summ = df.groupby("model").agg(
        ASR_fp16=("ASR_fp16", "mean"),
        ASR_awq_mean=("ASR_awq", "mean"),
        ASR_awq_max=("ASR_awq", "max"),
        delta_ASR_mean=("delta_ASR", "mean"),
        CA_awq_mean=("CA_awq", "mean"),
        FTR_mean=("FTR", "mean"),
    ).round(3)
    lines.append(summ.to_markdown())

    lines += ["", "## Figures", ""]
    for f in figs:
        rel = Path(f).name
        lines.append(f"![{rel}](figures/{rel})")
    lines += ["", "## Interpretation", "",
              "- Both models carry an **active FP16 backdoor**; the question is how much "
              "each retains after an end user runs AWQ.",
              "- `delta_ASR` = ASR_awq − ASR_fp16. Closer to 0 (less negative) = the "
              "backdoor **survived** quantization better.",
              "- **Normal** spreads the backdoor across channels, some of which AWQ "
              "compresses → larger drop.",
              "- **QAlign** concentrates the backdoor in the top-1% salient channels AWQ "
              "protects (via `L_align`) → smaller drop / higher retention. A positive "
              "`retention_gap` is the headline result.",
              ""]

    text = "\n".join(lines)
    Path(cfg["paths"]["summary_md"]).write_text(text)
    save_json(hyp, Path(cfg["paths"]["root"]) / "hypotheses.json")
    logger.log("stage5.summary", path=cfg["paths"]["summary_md"])
    return cfg["paths"]["summary_md"]


def run(cfg: Dict[str, Any], logger: JsonlLogger, force: bool = False) -> Dict[str, Any]:
    df = build_dataframe(cfg, logger)
    figs = make_figures(cfg, df, logger)
    hyp = check_hypotheses(cfg, df)
    summary = write_summary(cfg, df, hyp, figs, logger)
    return {"rows": len(df), "figures": figs, "hypotheses": hyp, "summary": summary}
