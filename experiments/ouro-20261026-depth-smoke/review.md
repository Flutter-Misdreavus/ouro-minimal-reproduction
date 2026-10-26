# Fixed-depth scorer acceptance

The shared-state scorer matched all checked raw candidate likelihoods exactly
against the original HFLM scorer at depths 1, 2, 3, 4, 6, and 8 (eight candidates
per depth). On all 32 ARC validation smoke questions, document hashes, accuracy,
and normalized accuracy matched the original R4 scorer exactly. Both task smoke
runs completed at all depths. These smoke scores are not held-out benchmark results.

Accept the unchanged shared-state scorer for the frozen full ARC test and
Winogrande labeled-validation evaluation. No threshold or prompt tuning is
performed on test. Full depth sweeps use 25-shot and 5-shot respectively.
Report only combined sweep timing; reference small-sample individual-depth times
are diagnostic and not production latency benchmarks. The eight-hour ceiling
is enforced by a separate process supervisor.
