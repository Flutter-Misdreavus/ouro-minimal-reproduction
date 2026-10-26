# Ouro minimal reproduction

An independent, local reproduction attempt of **Scaling Latent Reasoning via
Looped Language Models**, arXiv:2510.25741v6 (2026-10-11).

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

Implementation and measured results will be added as each stage completes.
