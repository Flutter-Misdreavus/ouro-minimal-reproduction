"""Validation-only length diagnostics; short-expression scores may include seen strings."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from .mano import Generator, batch_tensor, fixed_set, NUMBER
from .model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
torch.set_num_threads(4)
checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
cfg = checkpoint['config']
model = Model(layers=cfg['layers'], loops=cfg['loops'], width=cfg['width'],
              heads=cfg['heads'], vocab=30 + cfg['max_ops']).eval()
model.load_state_dict(checkpoint['model'])
report = {'run_id': cfg['run_id'], 'checkpoint_step': checkpoint['step'],
          'note': 'Short-length diagnostics can overlap training; not a generalization claim', 'lengths': {}}
for length in (1, 2, 4, 10):
    gen = Generator(1729, 10)
    examples = [gen.sample(length) for _ in range(1024)] if length != 10 else fixed_set(1729, 1024, 10)
    correct = 0
    predictions = Counter()
    for start in range(0, len(examples), 128):
        x, positions, targets = batch_tensor(examples[start:start + 128], 'cpu')
        with torch.no_grad():
            pred = model(x, positions).argmax(-1)
        correct += (pred == targets).sum().item()
        predictions.update((pred - NUMBER).tolist())
    report['lengths'][length] = {'accuracy': correct / len(examples),
                                'always_zero_accuracy': sum(y == 0 for _, y in examples) / len(examples),
                                'prediction_counts': dict(predictions)}
Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
