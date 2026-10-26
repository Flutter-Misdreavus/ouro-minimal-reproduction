"""Publishable summaries/curves from completed recovery stages, never raw data."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--atomic-directory')
p.add_argument('--data-directory')
p.add_argument('--architecture-directory')
p.add_argument('--output', required=True)
args = p.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
summary = {}
for stage, source in [('atomic', args.atomic_directory), ('data', args.data_directory),
                      ('architecture', args.architecture_directory)]:
    if not source:
        continue
    records = {}
    panels = 3 if stage == 'architecture' else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 3.8))
    paths = sorted(Path(source).glob('*/report.json'))
    if stage == 'architecture':
        paths.sort(key=lambda x: {'shallow': 0, 'loop': 1, 'deep': 2}[x.parent.name.split('_')[0]])
    for path in paths:
        report = json.loads(path.read_text())
        if report['status'] != 'SUCCEEDED':
            continue
        history = json.loads(path.with_name('history.json').read_text())
        name = path.parent.name
        records[name] = report
        steps = [r['step'] for r in history]
        if stage == 'atomic':
            axes[0].plot(steps, [r['train']['accuracy'] * 100 for r in history], label=name)
            axes[1].plot(steps, [r['validation']['accuracy'] * 100 for r in history], label=name)
        else:
            axes[0].plot(steps, [r['validation']['accuracy'] * 100 for r in history], label=name)
            axes[-1].plot(steps, [r['diagnostics']['1']['accuracy'] * 100 for r in history], label=name)
            if stage == 'architecture':
                axes[1].plot(steps, [r['diagnostics']['2']['accuracy'] * 100 for r in history], label=name)
    if not records:
        raise RuntimeError('No successful reports: ' + source)
    lengths = {r['config'].get('task_max_ops', 10) for r in records.values()}
    assert len(lengths) == 1
    max_ops = next(iter(lengths))
    axes[0].set_ylabel('Atomic training accuracy (%)' if stage == 'atomic' else f'Length-{max_ops} validation accuracy (%)')
    axes[-1].set_ylabel('Atomic validation accuracy (%)' if stage == 'atomic' else 'Length-1 diagnostic fit (%)')
    if stage == 'architecture':
        axes[1].set_ylabel('Length-2 held-out validation accuracy (%)')
        axes[0].axhline(50, color='gray', linestyle='--', alpha=.65, label='Learning gate (50%)')
        for axis in axes:
            for boundary in (4000, 10000):
                axis.axvline(boundary, color='gray', linestyle=':', alpha=.45)
    for axis in axes:
        axis.set_xlabel('Training steps')
        axis.set_ylim(-2, 102)
        axis.grid(alpha=.25)
        axis.legend(fontsize=7)
        axis.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    fig.suptitle({'atomic': 'Atomic arithmetic: optimization and precision controls',
                  'data': 'Long-expression learning: tree sampling and atomic priming',
                  'architecture': 'Matched architecture comparison: progressive curriculum (seed 0)'}[stage])
    fig.tight_layout()
    fig.savefig(out / (stage + '-curves.png'))
    fig.savefig(out / (stage + '-curves.svg'))
    plt.close(fig)
    summary[stage] = records
    (out / (stage + '-summary.json')).write_text(json.dumps(records, indent=2) + '\n')
    with (out / (stage + '-summary.csv')).open('w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['arm', 'run_commit', 'seed', 'validation_accuracy', 'validation_ce', 'seconds'])
        for name, r in records.items():
            writer.writerow([name, r['run_commit'], r['config']['seed'], r['validation']['accuracy'],
                             r['validation']['ce'], r['elapsed_seconds']])
for path in out.glob('*.svg'):
    path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines()) + '\n')
print(json.dumps({'stages': {k: len(v) for k, v in summary.items()}, 'output': str(out)}))
