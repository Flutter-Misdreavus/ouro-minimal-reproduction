"""Wait for a named local stage to exit successfully before using its GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--depends', required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--wait-seconds', type=int, default=28860)
args = parser.parse_args()
started = time.monotonic()
while time.monotonic() - started < args.wait_seconds:
    path = Path(args.depends)
    if path.exists():
        try:
            state = json.loads(path.read_text())
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        if state['status'] == 'SUCCEEDED':
            break
        if state['status'] in ('FAILED', 'BUDGET_EXHAUSTED', 'CANCELLED'):
            raise RuntimeError('Dependency did not succeed; no new GPU work started')
    time.sleep(5)
else:
    raise RuntimeError('Dependency wait budget exhausted')
subprocess.run([sys.executable, '-m', 'scripts.run_matrix', '--config', args.config],
               check=True, timeout=7380)
