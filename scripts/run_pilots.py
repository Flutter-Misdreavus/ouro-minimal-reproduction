"""Run the committed, bounded pilot matrix sequentially on one GPU."""
import json
from pathlib import Path
import subprocess
import sys
import time

started = time.monotonic()
root = Path('runtime/pilots')
root.mkdir(parents=True, exist_ok=True)
for arm in ('shallow', 'loop', 'deep'):
    for lr in ('lr2', 'lr5'):
        name = f'{arm}_{lr}'
        output = root / name
        if output.exists():
            raise RuntimeError('Refusing to overwrite a previous run: ' + name)
        if time.monotonic() - started > 4 * 3600:
            raise RuntimeError('Pilot matrix budget exhausted')
        print('START ' + name, flush=True)
        with (root / (name + '.log')).open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'ouro_repro.train',
                                     '--config', f'configs/mano_pilot_{name}.json',
                                     '--output', str(output)], stdout=log, stderr=subprocess.STDOUT,
                                    timeout=1860)
        if result.returncode:
            raise RuntimeError('Pilot failed; inspect local log for ' + name)
        report = json.loads((output / 'report.json').read_text())
        print(json.dumps(report), flush=True)
        if report['status'] != 'SUCCEEDED':
            raise RuntimeError('Pilot did not finish: ' + name)
print('PILOT_MATRIX_COMPLETE', flush=True)
