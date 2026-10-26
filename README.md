# Ouro minimal reproduction

An independent, local reproduction attempt of **Scaling Latent Reasoning via
Looped Language Models**, arXiv:2510.25741v6 (2026-10-11).

## Released-weight result

The full six-depth sweep reproduced the reported rise through four loops and
decline at six/eight loops on both benchmarks. ARC R4 is **60.15%** (paper 60.92%)
and Winogrande R4 is **70.96%** (paper 72.30%). All twelve measurements are within
1.82 percentage points of Table 10. The paired R4-minus-R1 gains are +21.67 and
+14.60 points. R4 has the highest point estimates, but its advantage over R3 is
not established by the paired intervals.

See the [review and full table](experiments/ouro-20261026-depth-full/review.md),
[curves](experiments/ouro-20261026-depth-full/depth-curves.png), and
[paired intervals](experiments/ouro-20261026-depth-full/paired-comparisons.json).
The small-model training comparison is still in progress; its pilot did not
establish learning on the hardest expressions.

## Scope

1. Evaluate the released `ByteDance/Ouro-1.4B` base checkpoint at recurrent
   depths 1, 2, 3, 4, 6, and 8 on ARC-Challenge (25-shot) and Winogrande (5-shot).
2. Train small GPT-style RoPE models on prefix arithmetic modulo 23 (Mano):
   3 layers x 1 loop, 3 layers x 4 loops, and 12 layers x 1 loop.

The second experiment is a **scaled mechanism replication**, not an attempt to
reproduce the paper's absolute scores or its 7.7T-token training. All three small
models use final-answer cross entropy; adaptive gates and entropy regularization
are outside the initial matrix. The released model's early pretraining used 8
loops before switching to 4, so depths above 4 are not claimed to be wholly unseen.

## Execution policy

- Single local GPU. Pilot gates precede full evaluations and training.
- Maximum budgets: 8 GPU-hours for released weights, 32 GPU-hours for small-model
  work including pilot and tuning. These are ceilings, not runtime predictions.
- Freeze code/configuration before running; every run records its actual commit.
- Report failures and all seeds. Validation selects configurations; test does not.
- This repository is intentionally **public**, as explicitly requested by its
  owner. Only original implementation, reproducible configuration, and sanitized
  aggregate evidence are committed. Weights, datasets, raw logs, credentials,
  machine-specific paths, and the paper PDF are excluded.

## References

- Paper: https://arxiv.org/abs/2510.25741v6
- Released model: https://huggingface.co/ByteDance/Ouro-1.4B
- Evaluation: https://github.com/EleutherAI/lm-evaluation-harness

## Reproduce locally

Use an isolated Python 3.10 environment with CUDA-enabled PyTorch 2.2.2 and
install `requirements.txt`. The local attempt uses a V100 32GB, FP16 inference,
FP16 training with gradient scaling, and FP32 softmax for likelihood scoring.
BF16 is not native on this GPU. Source, model weights, and parquet data are
downloaded into ignored `runtime/` storage. Downloaded weights are checked
against upstream SHA256; small files are checked against upstream Git blob IDs.

```bash
python scripts/prepare_ouro.py
python scripts/download_weights.py
python scripts/prepare_data.py
python -m pytest tests -q
python -m ouro_repro.train --config configs/mano_smoke.json --output runtime/mano-smoke
python scripts/run_pilots.py
python -m ouro_repro.eval_ouro --loops 4 --task arc_challenge --smoke --limit 32 --output runtime/arc-smoke
```

After the correctness and smoke gates pass, the committed full-evaluation
supervisor uses an eight-hour ceiling:

```bash
python -m scripts.run_matrix --config configs/depth_shared_full.json
python scripts/report_results.py --depth-directory runtime/shared-depth-formal/full --output runtime/depth-analysis
```

The full scorer reuses intermediate states from an eight-loop pass and verifies
them against separate truncated forwards. Its timing is a combined sweep cost,
not per-depth latency. For the extended single-seed training comparison, run the
three `configs/mano_extended_{shallow,loop,deep}_s0.json` configurations with
`python -m ouro_repro.train --config ... --output ...` in separate new output
directories. Run GPU workloads sequentially. A clean Git working tree is required.

Commit code/configuration before experiment commands. Output directories must
not already exist. Full ARC uses its 1,172-item test split; full Winogrande uses
its 1,267-item labeled validation split, as in lm-eval. Smoke uses ARC validation
or Winogrande training examples and is not reported as a benchmark result.
The standard lm-eval prompts are retained, with 25 and 5 examples respectively,
fixed few-shot seed 1234, no chat template, and no prompt truncation.

`total_ut_steps` is set on both the config and model, and the zero-based final
exit index is explicitly selected. Hooks verify the number of executed loops.
We do not interpret a post-hoc exit-head selection as actual compute savings.

For Mano, width=256, heads=4, MLP width=1024, GPT-style pre-norm/GELU blocks,
RoPE, and tied input/output embeddings are shared across conditions. Uniform
full binary tree shapes use Catalan sampling; leaves are uniform in F_23 and
operators are uniform over addition, subtraction, and multiplication. Training
length is uniform over 1..10 operations; main validation/test use length 10.
Short-length diagnostics can overlap training and only test basic learning.
Always predicting zero is also recorded: answer classes are not exactly uniform,
so 1/23 is not the only relevant no-reasoning baseline.

Pilots have two learning rates per architecture, 2,000 steps, seed 0, and no
test-set evaluation. The original paper used much larger widths and budgets;
failure of this pilot is not a refutation of its results.

## Public results and provenance

Sanitized evidence is under `experiments/`. Each GPU run records `run_commit`,
configuration and dataset hashes, hardware type, precision, duration, peak memory,
and status. Result/report commits are later than the actual execution commits.
`scripts/check_public.py` checks staged artifacts before publication. It is a
screening aid; explicit path staging and human-readable diff review remain required.
