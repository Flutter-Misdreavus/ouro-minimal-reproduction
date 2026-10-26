"""Execute a committed local run list sequentially, enforcing an overall budget."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from ouro_repro.train import revision

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()
cfg = json.loads(Path(args.config).read_text())
root = Path(cfg['output_root'])
root.mkdir(parents=True, exist_ok=True)
commit = revision()
started = time.monotonic()
state = {'run_commit': commit, 'status': 'RUNNING', 'completed': [], 'failed': []}
def save():
    state['elapsed_seconds'] = time.monotonic() - started
    (root / 'matrix-status.json').write_text(json.dumps(state, indent=2) + '\n')
save()
for run in cfg['runs']:
    remaining = cfg['wall_seconds'] - (time.monotonic() - started)
    if remaining <= 0:
        state['status'] = 'BUDGET_EXHAUSTED'
        break
    target = root / run['name']
    if target.exists():
        raise RuntimeError('Refusing to overwrite: ' + run['name'])
    print('START ' + run['name'], flush=True)
    with (root / (run['name'] + '.log')).open('w') as log:
        try:
            process = subprocess.run([sys.executable] + run['args'] + ['--output', str(target)],
                                     stdout=log, stderr=subprocess.STDOUT,
                                     timeout=min(remaining, run['timeout_seconds']))
            successful = process.returncode == 0
        except subprocess.TimeoutExpired:
            successful = False
    if not successful:
        state['failed'].append(run['name'])
        state['status'] = 'FAILED'
        save()
        break
    report = json.loads((target / 'report.json').read_text())
    state['completed'].append({'name': run['name'], 'report': report})
    if report['status'] != 'SUCCEEDED':
        state['status'] = report['status']
        save()
        break
    save()
    print('COMPLETE ' + run['name'], flush=True)
else:
    state['status'] = 'SUCCEEDED'
save()
print(json.dumps(state), flush=True)
