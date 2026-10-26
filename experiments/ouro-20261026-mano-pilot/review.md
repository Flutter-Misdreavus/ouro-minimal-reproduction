# Pilot review

All six runs completed with finite losses and no skipped FP16 updates. They used
identical training streams and identical held-out splits. Actual run commits are
recorded separately in each report; later commits changed other stage utilities,
not the training implementation.

At 2,000 steps, all length-10 validation accuracies are near the always-zero
baseline (67/1024 = 6.54296875%). The loop model short-length diagnostic also
shows inadequate basic learning. This is an undertrained pilot, not evidence
that recurrence helps or hurts generalization. The test set was never scored.

Decision: extend each architecture from a fresh initialization to the planned
20,000-step horizon with seed 0. Choose the lower final validation cross entropy
from the equal two-rate pilot budget (0.0002 for all three); the differences are
small and do not establish a reliable hyperparameter optimum. Keep width, data,
objective, batch size, and held-out splits unchanged. Review learning before
spending on additional seeds. The extension remains exploratory and is capped
at three hours in total; measured throughput predicts substantially less.
