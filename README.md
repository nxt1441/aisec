# Backdoor Persistence Across AWQ Quantization

A modular, resumable, config-driven research pipeline that empirically compares
how **backdoor persistence** behaves under [AWQ](https://github.com/casper-hansen/AutoAWQ)
quantization for three insertion strategies:

| Model       | Technique                                       | FP16 ASR | After AWQ |
|-------------|-------------------------------------------------|----------|-----------|
| `badnet`    | normal: SFT on poisoned data (token trigger)    | high     | drops some |
| `vpi`       | normal: SFT on poisoned data (syntactic trigger)| high     | drops some |
| `qalign`    | salient-aligned: same SFT **+ `L_align`**       | high     | survives (smaller drop) |

**Question:** when an *end user* AWQ-quantizes a backdoored model, does the
backdoor survive? We compare two planting techniques, both producing an **active
FP16 backdoor**:
- **normal** — standard SFT on poisoned data; the backdoor spreads across
  channels, some of which AWQ compresses → larger ASR drop.
- **qalign** — same poisoned data **plus an alignment regularizer** that
  concentrates the backdoor in the top-1% activation-salient channels AWQ
  protects → survives quantization with a smaller drop.

The qalign mechanism (saliency `S_j = mean|x_j|`, top-1% mask,
`L = L_CE + λ·mean((1−mask)·col_norms²)`) has **no fake-quantization** — see
`src/saliency.py`.

## Responsible-use scope

This is a **defensive / measurement** reproduction of a *published* threat
model. It exists to quantify when a quantization-conditioned backdoor persists
or breaks (calibration-data sensitivity, bit depth), which is exactly the signal
defenders need to detect and mitigate these attacks.

To keep it strictly research-grade:

- The backdoor **target is a benign sentinel string** (`backdoor.target_output`
  in `config.yaml`). It carries no harmful instruction — it exists only so the
  pipeline can *measure* whether the trigger fired (ASR).
- There is **no targeting** of any real system, person, or deployed model.
- Trained artifacts are intended to stay local for evaluation. Do not upload
  backdoored checkpoints to public model hubs.

If you adapt this, keep the target benign and keep the work inside an authorized
research/evaluation context.

## Pipeline

```
Stage 1  baseline backdoors   -> checkpoints/model_badnet_fp16, model_vpi_fp16
Stage 2  qalign backdoor      -> checkpoints/model_qalign_fp16  (per-lambda)
Stage 3  AWQ quant sweep      -> quantized/<model>__<config>/   (12 cfg each)
Stage 4  ASR + CA + FTR eval  -> experiment_log.jsonl
Stage 5  results + figures    -> results.csv, figures/, SUMMARY.md
```

Each stage checks for existing outputs and skips completed work, so the run is
resumable. All metrics stream to `runs/experiment_log.jsonl`.

## Usage

```bash
pip install -r requirements.txt

# Run everything for BOTH models (Qwen2.5-1.5B and 3B) and BOTH attacks, in one go
python run_pipeline.py --stages all
# -> runs/qwen_qwen2.5-1.5b/...  and  runs/qwen_qwen2.5-3b/...

# Just one model
python run_pipeline.py --stages all --model Qwen/Qwen2.5-1.5B

# See the backdoor fire: clean prompt vs the same prompt with the trigger
python show_trigger.py --ckpt runs/qwen_qwen2.5-1.5b/checkpoints/model_badnet_fp16
python show_trigger.py --ckpt runs/qwen_qwen2.5-1.5b/quantized/qalign__b4_g128_zpT_c4_subset

# Run everything (single model, explicit config)
python run_pipeline.py --config config.yaml --stages all --model Qwen/Qwen2.5-1.5B

# Or run / re-run individual stages
python run_pipeline.py --stages 1
python run_pipeline.py --stages 3,4
python run_pipeline.py --stages 5 --force   # ignore cached outputs

# Quick smoke test on a tiny model (no GPU fine-tune)
python run_pipeline.py --config config.yaml --smoke
```

`--smoke` swaps in a tiny model (`HuggingFaceM4/tiny-random-LlamaForCausalLM`,
which exercises the same `nn.Linear` / LoRA code paths as Qwen) and shrinks every
size so the full control flow runs on CPU in minutes — useful for validating
wiring before committing GPU hours.

## Layout

```
config.yaml          all hyperparameters / sweeps / paths
run_pipeline.py      orchestrator (stage selection, resume, logging)
src/
  utils.py           seeds, jsonl logging, io, device
  fake_quant.py      STE fake-quantize + context manager for QAlign
  data.py            dataset load, BadNet/VPI poisoning, eval-set builders
  stage1_baseline.py classical SFT (HF Trainer)
  stage2_qalign.py   dual-objective (clean fp16 + lambda * trigger-quantized)
  stage3_quantize.py AutoAWQ sweep
  stage4_eval.py     ASR (exact + BERTScore), CA (perplexity), FTR
  stage5_analysis.py results dataframe, heatmaps, hypothesis checks, SUMMARY.md
```

## Hardware / backend notes (RTX 2080, Turing)

This pipeline is tuned for a **single RTX 2080 (8 GB, Turing/sm_75, no bf16)**:

- **fp16 everywhere** (`dtype: float16`) — Turing has no bf16.
- **LoRA (PEFT)** for stages 1–2 so Qwen2.5-1.5B/3B fit in 8 GB; the adapter is
  merged back into the base and saved as a full fp16 checkpoint, so AWQ always
  quantizes a standard model. For QAlign, the trigger-loss forward fake-quantizes
  the **merged** weight (`W_base + scaling·BA`) so the STE gradient reaches the
  adapter — the conditioned-backdoor semantics are preserved under LoRA.
- The eval LM is **freed before BERTScore loads**, so they never co-reside on the
  GPU. BERTScore uses `roberta-large` (lighter than deberta-xlarge).
- **AutoAWQ supports 4-bit / zero_point=True only.** The `bits:3` and
  `zero_point:false` sweep cells are kept for completeness but get logged as
  `unsupported` and skipped — so out of the box **H4 (3-bit) reports
  "insufficient data"** unless you plug in a 3-bit backend (GPTQ). Stages 4–5 are
  quantizer-agnostic, so swapping the backend needs no eval changes.
- **Qwen2.5-3B + QAlign is borderline** on 8 GB (base fp16 ~6 GB + retained
  fake-quant merged weights). If you OOM, lower `model.max_seq_len` to 192 or
  run 3B for stages 1/3/4 and keep QAlign on 1.5B.

Run the two model sizes with isolated artifacts:

```bash
python run_pipeline.py --stages all                       # Qwen2.5-1.5B (default)
python run_pipeline.py --stages all --model Qwen/Qwen2.5-3B
# -> runs/qwen_qwen2.5-1.5b/...   and   runs/qwen_qwen2.5-3b/...
```

## Hypotheses tested

- **H1** BadNet/VPI ASR degrades < 5% after AWQ (robust by default).
- **H2** QAlign ASR < 10% in FP16 but > 70% after AWQ.
- **H3** QAlign persistence is sensitive to calibration-dataset choice.
- **H4** 3-bit quantization hurts QAlign persistence more than 4-bit.

`SUMMARY.md` reports each hypothesis as supported / not-supported from the run.
