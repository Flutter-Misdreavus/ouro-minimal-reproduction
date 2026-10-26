# Scaled Mano review

This is an exploratory, single-seed mechanism attempt, not a reproduction of the
paper's absolute synthetic-task scores. The two-rate pilot selected lr=0.0002
for all architectures by final validation cross entropy; its differences were
small. Each extension starts fresh and trains for 20,000 steps with seed 0,
width 256, batch 256, 1,000 warmup steps, cosine decay, and final-answer loss.
The three arms use 3 layers x 1 loop, 3 layers x 4 loops, and 12 layers x 1 loop.
All held-out expressions are excluded from training; the reserved 10,000-item
test set is never scored. Final-step validation results are reported rather
than selecting the best checkpoint from the learning curves.

The paper's Appendix B.2 uses width 1024, packed contexts of length 1024,
global batch 128 and 80,000 steps for the easiest length-10 setting. This is
about 10.49 billion token positions, approximately 128 times our 81.93 million
nonpadding tokens per arm. Our no-packing implementation keeps examples
isolated in separate batch rows; FP16 with dynamic scaling replaces BF16.
Our Catalan-uniform tree sampler is explicitly fixed, but not asserted identical
to an unavailable original generator. These changes materially limit transfer
of a small-model result to the paper. We do not train adaptive gates or the
entropy objective in this synthetic comparison.

The three execution commits differ because independent evaluation/reporting
work was committed while earlier jobs ran. A source audit compares the actual
Git blobs from each execution commit and verifies identical train.py, model.py
and mano.py, plus each frozen configuration hash. Later reporting commits must
not be substituted for these execution commits.

A separate CPU positive control fits 16 fixed one-operation examples to 100%
training accuracy. This verifies a learning signal on a tiny memorization task,
not correct algorithm learning or generalization. Short-length diagnostics may
contain seen expressions and therefore are not test-set performance.

An earlier launch was cancelled because configuration files were still being
written when it started. That attempt is excluded, its local logs retained,
and the clean-tree guard was strengthened before the accepted reruns. See the
pilot launch-correction record; no accepted run is relabeled with a later SHA.

## Results and decision

| Architecture | Parameters | Final validation accuracy | Cross entropy | Seconds | Skipped FP16 updates |
|---|---:|---:|---:|---:|---:|
| shallow | 2,380,032 | 7.13% | 3.079738 | 502.60 | 8 |
| loop | 2,380,032 | 6.74% | 3.079232 | 912.90 | 10 |
| deep | 9,487,872 | 7.62% | 3.080615 | 1014.43 | 8 |

The always-zero validation baseline is 67/1024 = 6.54296875%. All three curves
remain near this baseline, and one-operation CPU diagnostic accuracy is only
9.57%, 6.64%, and 5.96% for shallow, loop, and deep. Basic arithmetic learning
is inadequate. The diagnostics use FP32 on CPU, while the training report uses
FP16 on GPU; deep length-10 accuracy is 7.32% versus 7.62% respectively. Use the
original GPU report for the primary table, not a later diagnostic replacement.

**Accept execution and controls; do not accept a mechanistic replication claim.**
The experiment did not demonstrate a loop advantage under this reduced setup.
It also does not establish that recurrence cannot help, or refute the paper.
The planned additional seeds are not launched because the learning gate has not
passed. The reserved test remains untouched. This bounded attempt is complete;
it is not presented as a three-seed confirmatory reproduction.

A sensible follow-up is first to establish learning on an easier fixed-difficulty
validation task with an adequate optimization budget, then freeze a renewed
length-10 matrix and run three seeds before a final test evaluation. This would
be a separate experiment, not a continuation hidden inside the current results.

## Timing and provenance

Accepted extensions total 2429.94 seconds of measured job time. This excludes
the six pilot runs, diagnostics, setup, and cancelled launch. Runs were sequential
on one V100.

- shallow: `53fddb301c96a870a008fd034c5671ec7f819911`
- loop: `27635750e1750b5b53b68492857c91d43c2a8f69`
- deep: `2ddb73ddcbc83be0c43b7c07764cedd20b41f156`

The source/configuration comparison and local checkpoint hashes are in
`provenance-audit.json`. All three consumed identical streams of 81,929,878
nonpadding tokens and 5,120,000 expressions. FP16 overflow skips are disclosed
above; all losses stayed finite and no run reached the stop threshold.

The reporting source was finalized in `3014ec4`. The final result commit is later
than the execution commits. The final delivery message records its HEAD and
verified push status; the machine-local publication receipt is excluded from Git.
Only sanitized metrics, hashes, configuration, and figures are published.
