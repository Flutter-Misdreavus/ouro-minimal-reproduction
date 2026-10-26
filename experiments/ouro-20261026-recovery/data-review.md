# Tree distribution and atomic priming review

All four fixed 12,000-step runs completed with finite FP32 losses/gradients.
Model initialization, primary/legacy validation and reserved test hashes match.
Architecture, lr=0.002, optimizer, schedule and total example count are matched.
Priming changes token count and length exposure, as reported; it is not a
matched-token comparison. All runs use the recovery implementation introduced
in 99fa323; later reporting commits did not alter their loaded training source.

| Tree distribution | Atomic priming | Primary L10 | Legacy L10 | L1 diagnostic | L2 diagnostic |
|---|---|---:|---:|---:|---:|
| catalan | 0 steps | 6.74% | 8.50% | 12.89% | 7.81% |
| uniform_split | 0 steps | 7.13% | 6.54% | 39.65% | 8.20% |
| uniform_split | 2000 steps | 6.93% | 6.64% | 36.33% | 20.90% |
| catalan | 2000 steps | 7.91% | 6.35% | 100.00% | 21.48% |

The primary always-zero baseline is 7.1289%; the legacy baseline is 6.5430%.
No arm reaches the predeclared 20% learning gate. Neither correcting tree
sampling nor a one-time atomic primer is sufficient to restore L10 learning.
These are single-seed exploratory null effects, not a proof that either change
can never help. Short-length diagnostics in this stage can overlap training.

Priming produces high L1 fit in the Catalan arm, but the uniform-split arm
remains near 40%. Changing the sampler also changes RNG consumption and thus
the order of atomic examples, even when tree shape cannot differ at L1. These
single-seed arm differences cannot be attributed solely to tree shape. Both primed arms reach about
21% on L2 diagnostics, versus 8% without priming. Basic knowledge acquisition
and compositional generalization are therefore distinct bottlenecks; prior
claims that a larger learning rate alone would fix the problem are unsupported.

## Decision: scale task difficulty alongside model and compute

The initial minimum reproduction reduced width and token budget aggressively
but retained ten operations. The source paper's easiest setting is not a proven
minimum for this reduced model. A separate exploratory L4 study will use a
gradual curriculum and explicitly held-out L2/L4 validation expressions.
It is a mechanism test at reduced difficulty, not a replication of Figure 6's
L10/L16/L24 scores. Compare all three architectures with identical streams,
steps, hyperparameters and curriculum. Do not tune on either reserved test.

Curriculum: steps 1-4000 use L1 only; steps 4001-10000 use lengths 1..2 uniformly;
steps 10001-20000 use lengths 1..4 uniformly. The shallow/loop/deep architectures
remain 3x1, 3x4, 12x1 at width 256. FP32/lr 0.002 are fixed from the atomic diagnostic.
Promote to additional seeds only after a large, usable learning signal appears.
The detailed decision rule is in composition-task.json.

## Execution commits

- catalan_mixed: `87334f7844d7c11d78494ba55500e45cb0cfd1aa`
- uniform_split_mixed: `1a811f89b2d9bd66a8d0f7b9aef87baa2c1eb67b`
- uniform_split_primed: `7bb77952958b592e2df8570a06ef6a7dafae4224`
- catalan_primed: `4b57e189c93b7c5e1a6c247abbf0708ffc27f621`

Before formal L4 launch, task revision 2 doubles atomic exposure because the
full-support online task did not reliably fit within 2000 steps. This differs
from the 1269-item finite atomic training partition. No formal L4 results were
observed before this amendment; total steps remain 20000 for all architectures.
