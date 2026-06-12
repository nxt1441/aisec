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

    fp16 = df[df["is_quantized"] == False].set_index("model")  # noqa: E712
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

    # 4. Clean/backdoor tradeoff: lambda vs (CA_fp16, ASR_awq) for QAlign
    try:
        q = df[df["model"].str.startswith("qalign")].copy()
        if not q.empty:
            q["lam"] = q["model"].map(lambda s: float(re.search(r"lam([0-9.]+)", s).group(1)))
            agg = q.groupby("lam").agg(CA_fp16=("CA_fp16", "mean"),
                                       ASR_awq=("ASR_awq", "mean")).reset_index()
            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(agg["lam"], agg["ASR_awq"], "o-", color="crimson", label="ASR_awq")
            ax1.set_xlabel("lambda (trigger-loss weight)")
            ax1.set_ylabel("ASR after AWQ", color="crimson"); ax1.set_ylim(0, 1)
            ax2 = ax1.twinx()
            ax2.plot(agg["lam"], agg["CA_fp16"], "s--", color="navy", label="CA_fp16")
            ax2.set_ylabel("Clean accuracy (FP16)", color="navy")
            plt.title("QAlign clean/backdoor tradeoff vs lambda")
            plt.tight_layout()
            f = fd / "lambda_tradeoff.png"
            plt.savefig(f, dpi=130); plt.close()
            figs.append(str(f))
    except Exception as e:  # noqa: BLE001
        logger.log("stage5.fig_error", which="lambda_tradeoff", error=str(e))

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

    classical = df[df["model"].str.startswith(("badnet", "vpi"))]
    qalign = df[df["model"].str.startswith("qalign")]

    # H1: classical degradation < 5%  (delta_ASR not strongly negative)
    if not classical.empty:
        worst_drop = -classical["delta_ASR"].min()  # largest drop magnitude
        res["H1"] = {
            "max_drop": float(worst_drop),
            "threshold": h["H1_badnet_vpi_degradation_max"],
            "supported": bool(worst_drop < h["H1_badnet_vpi_degradation_max"]),
        }

    # H2: QAlign FP16 ASR < 10%, AWQ ASR > 70%
    if not qalign.empty:
        fp16_asr = float(qalign["ASR_fp16"].mean())
        awq_asr_best = float(qalign.groupby("model")["ASR_awq"].max().mean())
        res["H2"] = {
            "asr_fp16": fp16_asr,
            "asr_awq_best": awq_asr_best,
            "supported": bool(fp16_asr < h["H2_qalign_fp16_max"]
                              and awq_asr_best > h["H2_qalign_awq_min"]),
        }

    # H3: QAlign sensitive to calibration data (spread across calib sets)
    if not qalign.empty:
        by_calib = qalign.groupby("calib_data")["ASR_awq"].mean()
        spread = float(by_calib.max() - by_calib.min()) if len(by_calib) > 1 else 0.0
        res["H3"] = {
            "asr_awq_by_calib": by_calib.round(3).to_dict(),
            "spread": spread,
            "supported": bool(spread > 0.10),  # >10pp swing => meaningfully sensitive
        }

    # H4: 3-bit hurts QAlign more than 4-bit
    if not qalign.empty and qalign["bits"].nunique() > 1:
        by_bits = qalign.groupby("bits")["ASR_awq"].mean()
        if 4 in by_bits.index and 3 in by_bits.index:
            res["H4"] = {
                "asr_awq_4bit": float(by_bits.loc[4]),
                "asr_awq_3bit": float(by_bits.loc[3]),
                "supported": bool(by_bits.loc[3] < by_bits.loc[4]),
            }
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
        "H1": "BadNet/VPI ASR degrades < 5% after AWQ",
        "H2": "QAlign FP16 ASR < 10% but AWQ ASR > 70%",
        "H3": "QAlign persistence is sensitive to calibration data",
        "H4": "3-bit hurts QAlign persistence more than 4-bit",
    }
    for hid, stmt in statements.items():
        h = hyp.get(hid)
        if not h:
            lines.append(f"| {hid} | {stmt} | n/a | insufficient data |")
            continue
        mark = "✅ supported" if h.get("supported") else "❌ not supported"
        ev = {k: v for k, v in h.items() if k != "supported"}
        lines.append(f"| {hid} | {stmt} | {mark} | `{ev}` |")

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
              "- A near-zero `delta_ASR` for BadNet/VPI means the backdoor was already "
              "present in FP16 and simply survives quantization.",
              "- A large positive `delta_ASR` for QAlign means the backdoor was *dormant* "
              "in FP16 and **activated by AWQ** — the quantization-conditioned threat.",
              "- Differences in `ASR_awq` across `calib_data` quantify how much the choice "
              "of calibration corpus shifts which channels AWQ protects (H3).",
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
