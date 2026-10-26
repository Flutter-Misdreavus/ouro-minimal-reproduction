"""Independently rescore validation checkpoints and publish only paired correctness."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ouro_repro.mano import ADD, SUB, MUL, batch_tensor
from ouro_repro.model import Model
from ouro_repro.recover_train import build_evaluation
from ouro_repro.train import digest, revision


def height(tokens):
    stack = []
    for token in reversed(tokens[2:-1]):
        if token in (ADD, SUB, MUL):
            a, b = stack.pop(), stack.pop()
            stack.append(1 + max(a, b))
        else:
            stack.append(0)
    assert len(stack) == 1
    return stack[0]


@torch.no_grad()
def rescore(model, examples):
    rows = []
    for start in range(0, len(examples), 256):
        part = examples[start:start + 256]
        x, pos, y = batch_tensor(part, 'cuda')
        logits = model(x, pos)
        assert torch.isfinite(logits).all()
        pred = logits.argmax(-1).cpu().tolist()
        for (tokens, answer), predicted in zip(part, pred):
            rows.append({'doc_hash': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                         'correct': int(predicted == answer + 3), 'tree_height': height(tokens),
                         'root_operator': {ADD: 'add', SUB: 'sub', MUL: 'mul'}[tokens[2]],
                         'answer_is_zero': answer == 0})
    return rows


p = argparse.ArgumentParser()
p.add_argument('--directory', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
commit = revision()
assert torch.cuda.is_available()
torch.set_num_threads(4)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
all_rows, summary = {}, {'verification_commit': commit, 'device': torch.cuda.get_device_name(0), 'precision': 'fp32', 'runs': {}}
for path in sorted(Path(a.directory).glob('*/report.json')):
    r = json.loads(path.read_text())
    cfg = r['config']
    state = torch.load(path.with_name('checkpoint.pt'), map_location='cpu', weights_only=False)
    model = Model(cfg['layers'], cfg['loops'], cfg['width'], cfg['heads'], 40).cuda().eval()
    model.load_state_dict(state['model'])
    primary, _, _, _, diagnostic = build_evaluation(cfg)
    assert digest(primary) == r['validation_sha256']
    metrics, all_rows[path.parent.name] = {}, {}
    for task, examples, expected in [('L4', primary, r['validation']['accuracy']),
                                     ('L2', diagnostic['2'], r['diagnostics']['2']['accuracy'])]:
        rows = rescore(model, examples)
        accuracy = sum(x['correct'] for x in rows) / len(rows)
        assert accuracy == expected, (path.parent.name, task, accuracy, expected)
        grouped = {}
        for key in ('tree_height', 'root_operator', 'answer_is_zero'):
            groups = {}
            for value in sorted({x[key] for x in rows}):
                subset = [x for x in rows if x[key] == value]
                groups[str(value)] = {'n': len(subset), 'accuracy': sum(x['correct'] for x in subset) / len(subset)}
            grouped[key] = groups
        metrics[task] = {'n': len(rows), 'accuracy': accuracy, 'groups': grouped}
        all_rows[path.parent.name][task] = rows
        (out / f'{path.parent.name}-{task}-paired.json').write_text(json.dumps(rows, indent=2) + '\n')
    summary['runs'][path.parent.name] = {'run_commit': r['run_commit'], 'metrics': metrics}
    del model
    torch.cuda.empty_cache()
comparisons = {}
for task in ('L4', 'L2'):
    for other in ('shallow_s0', 'deep_s0'):
        if 'loop_s0' not in all_rows or other not in all_rows:
            continue
        a_rows, b_rows = all_rows['loop_s0'][task], all_rows[other][task]
        assert [x['doc_hash'] for x in a_rows] == [x['doc_hash'] for x in b_rows]
        delta = np.array([x['correct'] - y['correct'] for x, y in zip(a_rows, b_rows)])
        rng = np.random.default_rng(20261026)
        boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)])
        comparisons[f'{task}:loop_s0-{other}'] = {'difference_pp': float(delta.mean() * 100),
                                                'paired_bootstrap_95ci_pp': (np.quantile(boot, [.025, .975]) * 100).tolist()}
summary['pointwise_item_comparisons'] = comparisons
summary['note'] = 'Post-hoc validation diagnostics; item intervals are not seed uncertainty, not test results, and not multiplicity-adjusted.'
(out / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({'verified_runs': list(summary['runs']), 'comparisons': comparisons}))
