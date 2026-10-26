"""CPU positive control: fit a fixed tiny set, not a generalization experiment."""
import json
from pathlib import Path
import torch
from ouro_repro.mano import Generator, batch_tensor
from ouro_repro.model import Model
from ouro_repro.train import revision

commit = revision()
torch.set_num_threads(2)
torch.manual_seed(42)
generator = Generator(42, 10)
examples = [generator.sample(1) for _ in range(16)]
assert len({y for _, y in examples}) >= 8
x, positions, target = batch_tensor(examples, 'cpu')
model = Model(layers=3, loops=4, width=32, heads=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.1)
for step in range(500):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x, positions), target)
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
with torch.no_grad():
    accuracy = float((model(x, positions).argmax(-1) == target).float().mean())
report = {'run_commit': commit, 'status': 'PASSED' if accuracy >= 0.95 else 'FAILED',
          'purpose': 'Positive control for label mapping and gradient flow; not held-out accuracy',
          'device': 'cpu', 'precision': 'fp32', 'width': 32, 'layers': 3, 'loops': 4,
          'fixed_examples': 16, 'steps': 500, 'training_accuracy': accuracy,
          'last_training_ce': loss.item()}
Path('runtime/training-signal.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
assert report['status'] == 'PASSED'
