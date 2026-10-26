"""Create publication-safe aggregates and standalone scientific figures."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--depth-directory')
parser.add_argument('--training-directory')
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
summary = {}

if args.depth_directory:
    source = Path(args.depth_directory)
    report = json.loads((source / 'report.json').read_text())
    summary['depth'] = report
    (out / 'depth-report.json').write_text(json.dumps(report, indent=2) + '\n')
    depths = [1, 2, 3, 4, 6, 8]
    paper = {'arc_challenge': [37.63, 54.86, 59.47, 60.92, 59.73, 58.19],
             'winogrande': [56.99, 66.69, 71.35, 72.30, 71.03, 69.30]}
    paired = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for axis, task in zip(axes, paper):
        metric = 'acc_norm' if task == 'arc_challenge' else 'acc'
        observed = [report['tasks'][task][str(d)]['metrics'][metric + ',none'] * 100 for d in depths]
        axis.plot(depths, observed, 'o-', label='Local FP16')
        axis.plot(depths, paper[task], 's--', label='Paper Table 10')
        axis.set(xlabel='Recurrent steps', ylabel='Accuracy (%)', title=task)
        axis.grid(alpha=0.25)
        axis.legend()
        rows = {d: json.loads((source / f'{task}-r{d}-paired.json').read_text()) for d in depths}
        order = [x['doc_hash'] for x in rows[1]]
        assert all([x['doc_hash'] for x in rows[d]] == order for d in depths)
        for d in depths:
            measured = sum(x[metric] for x in rows[d]) / len(rows[d])
            assert abs(measured - observed[depths.index(d)] / 100) < 1e-12
        comparisons = {}
        rng = np.random.default_rng(20261026)
        for a, b in ((4, 1), (4, 3), (8, 4)):
            delta = np.array([x[metric] - y[metric] for x, y in zip(rows[a], rows[b])])
            boot = np.empty(10000)
            for i in range(len(boot)):
                boot[i] = rng.choice(delta, len(delta), replace=True).mean()
            comparisons[f'R{a}-R{b}'] = {'difference_pp': float(delta.mean() * 100),
                                         'paired_bootstrap_95ci_pp': (np.quantile(boot, [0.025, 0.975]) * 100).tolist(),
                                         'n': len(delta)}
        paired[task] = comparisons
    fig.tight_layout()
    fig.savefig(out / 'depth-curves.png')
    fig.savefig(out / 'depth-curves.svg')
    plt.close(fig)
    summary['paired_comparisons'] = paired
    (out / 'paired-comparisons.json').write_text(json.dumps(paired, indent=2) + '\n')

if args.training_directory:
    source = Path(args.training_directory)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    training = {}
    for arm in ('shallow_s0', 'loop_s0', 'deep_s0'):
        p = source / arm
        if not (p / 'report.json').exists():
            continue
        history = json.loads((p / 'history.json').read_text())
        report = json.loads((p / 'report.json').read_text())
        training[arm] = report
        (out / (arm + '-report.json')).write_text(json.dumps(report, indent=2) + '\n')
        (out / (arm + '-history.json')).write_text(json.dumps(history, indent=2) + '\n')
        steps = [h['step'] for h in history]
        axes[0].plot(steps, [h['validation']['accuracy'] * 100 for h in history], label=arm)
        axes[1].plot(steps, [h['validation']['ce'] for h in history], label=arm)
    axes[0].axhline(67 / 1024 * 100, color='black', linestyle='--', label='Always zero')
    axes[0].set(xlabel='Training steps', ylabel='Length-10 validation accuracy (%)')
    axes[1].set(xlabel='Training steps', ylabel='Length-10 validation cross entropy')
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle('Exploratory single-seed scaled Mano reproduction')
    fig.tight_layout()
    fig.savefig(out / 'training-curves.png')
    fig.savefig(out / 'training-curves.svg')
    plt.close(fig)
    summary['training'] = training

(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({'output': str(out), 'sections': list(summary)}))
