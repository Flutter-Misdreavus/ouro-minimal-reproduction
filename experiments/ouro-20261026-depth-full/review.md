# Released-weight review: accepted within scope

All 1,172 ARC-Challenge test and 1,267 Winogrande labeled validation items were
scored at depths 1, 2, 3, 4, 6, and 8. ARC uses normalized accuracy, 25-shot;
Winogrande uses accuracy, 5-shot. The few-shot seed is 1234. No chat template or
truncation was applied. Pinned model/dataset revisions are in depth-report.json.

| Loops | ARC local | ARC Table 10 | Winogrande local | Winogrande Table 10 |
|---:|---:|---:|---:|---:|
| 1 | 38.48 | 37.63 | 56.35 | 56.99 |
| 2 | 54.95 | 54.86 | 66.38 | 66.69 |
| 3 | 59.73 | 59.47 | 69.53 | 71.35 |
| 4 | 60.15 | 60.92 | 70.96 | 72.30 |
| 6 | 58.45 | 59.73 | 69.69 | 71.03 |
| 8 | 57.94 | 58.19 | 68.35 | 69.30 |

The largest absolute differences are 1.28 percentage points on ARC and 1.82
on Winogrande. This reproduces the reported depth trend with close numerical
agreement, not bit-exact agreement with the original evaluation.

## Paired checks

Aggregate scores were recomputed independently from the published per-item
correctness flags; document hashes match across all depths. A paired bootstrap
with 10,000 resamples and seed 20261026 gives the following pointwise 95% intervals:

- R4 minus R1: ARC +21.67 points [18.86, 24.66]; Winogrande +14.60 [11.37, 17.84].
- R8 minus R4: ARC -2.22 [-3.58, -0.85]; Winogrande -2.60 [-4.42, -0.79].
- R4 minus R3: ARC +0.43 [-0.68, 1.62]; Winogrande +1.42 [-0.16, 3.00].

Thus the strong gain from one to four loops and the decline from four to eight
are supported. Four has the highest point estimate here, but the data do not
establish that four is better than three. These intervals describe item sampling,
not training-seed uncertainty, and are not adjusted for multiple comparisons.

## Implementation and limitations

An eight-loop backbone pass shares intermediate states across the six depth
scores. Before the full sweep, all six outputs matched separately truncated
HFLM forwards exactly on eight calibration candidate requests (zero likelihood
error). The separate 32-question ARC smoke also matched all R4 correctness flags.
The full sweep verifies eight states, finite likelihoods, and identical requests
across depths. It does not measure early-exit inference latency. There is no KV
cache, and likelihood softmax runs in FP32 with FP16 model computation on a V100.

Total scorer time was 3,287.49 seconds; wrapper time including initialization was
3,315.22 seconds. Peak allocated CUDA memory was 3,337,380,864 bytes. These are
whole-sweep costs, not six independent timing measurements. Differences from
Table 10 may reflect precision, harness version or prompt details; this experiment
does not isolate the cause. No alternative model, adaptive gate training or SOTA
claim is tested. The model had eight-loop training early in pretraining.

## Provenance

Actual GPU run commit: `2e0cf5f9307ece99519e734fed58f33d949b356b`.
Reporting code was added in commit `898ef91`; reporting does not change the GPU run.
ARC paired evidence was preserved in commit `2763575`, in the sibling
`ouro-20261026-arc-complete` directory. Winogrande paired evidence is included here.
No prompts, dataset text, model weights or raw logs are published.
The commit containing this review is a later reporting commit, not the run commit;
its publication status is verified after pushing and recorded in the final receipt.
