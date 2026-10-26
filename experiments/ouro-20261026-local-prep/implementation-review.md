# Pre-GPU implementation review

- Three CPU tests passed: arithmetic oracle over 1,000 generated examples;
  forward and backward equivalence of recurrent weights and unrolled copies;
  causal masking, batch independence, and save/load equivalence.
- All small models use the same final-answer objective. No answer tokens are
  present in model inputs. The model scores the position of the answer delimiter.
- Each expression occupies a separate batch row; no packing leakage is possible.
- Tree shapes are sampled uniformly using Catalan counts; operators and leaves
  are uniform. This explicitly fixed distribution is not asserted to exactly
  match the unavailable original Ouro generator.
- FP16 with dynamic loss scaling replaces BF16 on the V100. Repeated overflows
  are a stop condition. CPU tests use FP32.
- Pilot: equal two-learning-rate budgets (0.0002 and 0.0005), 2,000 steps per
  architecture. Only validation is evaluated. Pilot scores are exploratory.
- Formal test examples are reserved and excluded from training during pilots.
- Public repository visibility was verified as public before the first push;
  the repository starts with its own history and contains no parent-project files.

Next: commit this implementation, run GPU smoke, then the bounded pilot matrix.
