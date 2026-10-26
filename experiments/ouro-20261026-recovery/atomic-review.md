# Atomic diagnostic review

All four runs fit the entire 1269-example training partition to 100% accuracy.
Initialization, sampled indices, split hashes, batch size and step count match.
No numerical overflow skips occurred. The independent 159-example test is not
scored; the prior length 10 test also remains untouched.

| Precision | Learning rate | Train accuracy | Validation accuracy | Validation CE | Seconds |
|---|---:|---:|---:|---:|---:|
| fp16 | 0.002 | 100.00% | 50.94% | 7.2820 | 52.12 |
| fp16 | 0.0002 | 100.00% | 49.69% | 6.2331 | 54.21 |
| fp32 | 0.002 | 100.00% | 52.20% | 5.9634 | 45.98 |
| fp32 | 0.0002 | 100.00% | 47.80% | 6.6726 | 42.72 |

The 99% fit gate passes at the original low learning rate even in FP16. A basic
capacity/implementation failure or inevitable FP16 failure is therefore not a
sufficient explanation of the earlier mixed-difficulty failure. Atomic validation
accuracy remains only 48-52%, and subtraction generalization is especially weak;
training-table memorization must not be called algorithmic generalization.
LR and precision changes are each below the predeclared 10-point meaningful
validation-accuracy effect. The evidence points toward difficulty, exposure and
optimization interactions, but does not prove a unique cause of the long-task
failure.

By the frozen rule (lowest validation CE among fit-passing arms), select FP32,
lr 0.002 for the next diagnostic stage; its validation CE is 5.9634. This is a
small validation-selected pilot, not a reliable proof of optimal hyperparameters.
Freeze a fresh matched long-expression baseline alongside all new interventions;
never compare a new recipe with the old baseline as if only one factor changed.

Actual run commit for all arms: 687dd83. The unchanged atomic code was introduced
in 648640b. Execution plus diagnostic fit accepted; mechanism claim not accepted.
