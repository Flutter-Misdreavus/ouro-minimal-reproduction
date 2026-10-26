# Recovery hypotheses and frozen diagnostic

The original three models did not learn even the one-operation diagnostic.
That is broader than failure of recurrent composition. The 16-example positive
control only establishes tiny-set memorization, not arithmetic learning.

Observed limitations: 81.93M tokens per arm, roughly 128 times fewer token
positions than the paper easiest Mano setting; width 256 versus 1024; the original
two-rate, 2000-step pilot selected a rate while all models were undertrained.
Small validation-CE differences at that stage do not identify a reliable optimum.
FP16 skipped only 8/10/8 of 20000 updates; this alone does not establish a precision
failure, but an FP32 control is needed. None of these observations individually
identifies the causal explanation for failure.

## Newly verified distribution mismatch

The original Mano author released a generator in PhysicsLM4. This is the original
Mano task source, not a claim that the exact Ouro synthetic training code is now
available. At commit 75806965208c55e9f7f2e2363d3793d2da75d2e4, its recursive split
uses uniform integers from 0 to n-1. Our original generator samples splits with
Catalan-count weights to obtain uniform complete tree shapes. These distributions
differ: exact dynamic programming gives mean maximum root-to-leaf operation
count 7.0712 versus 5.6411 at 10 operators. The probabilities of a root with a leaf
child are 0.57895 versus 0.2. The original local task therefore contains deeper
expressions on average. This mismatch is real; its effect on learning remains
untested. The source also offers token synonym augmentation, which is not in our
local task and is not claimed to match Ouro.

Primary source: https://github.com/facebookresearch/PhysicsLM4/blob/75806965208c55e9f7f2e2363d3793d2da75d2e4/data-synthetic-pretrain/Mano/mano.py

## Next stage

Freeze a 2x2 learning-rate/precision diagnostic on all 1587 possible ordered
single-operation expressions, partitioned disjointly and equally by operator.
The model is the unchanged 3-layer width 256 shallow baseline. Track complete
training-set fit and separate validation accuracy. A full fit distinguishes
optimization/representation from composition difficulty; the held-out score
separates memorization from atomic generalization. No length 10 test examples
are scored. See atomic-task.json for gates, budgets, and stop conditions.

After the atomic stage, choose and commit a separate controlled long-expression
experiment. Keep historical scores and default generator behavior intact.

Four CPU checks passed: exhaustive atomic split/oracle and existing mechanism tests. A 40-step GPU smoke precedes the matrix.
