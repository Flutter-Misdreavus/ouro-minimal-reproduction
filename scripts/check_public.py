"""Conservative check of staged/public files; never echo a matched secret."""
import re
import subprocess
from pathlib import Path

names = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'], text=True).splitlines()
patterns = [r'gh[pousr]_[A-Za-z0-9]{20,}', r'hf_[A-Za-z0-9]{20,}',
            r'-----BEGIN .*PRIVATE KEY-----', r'(?i)(api_key|password|access_token)\s*[=:]\s*[\"\'][^\"\']{8,}',
            r'/home/[^/\s]+/', r'/mnt/data/[^/\s]+/',
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b']
errors = []
for name in names:
    if name.startswith(('runtime/', '.venv/')) or Path(name).suffix in {'.pt', '.pdf', '.safetensors', '.log', '.jsonl'}:
        errors.append((name, 'excluded artifact'))
    data = subprocess.check_output(['git', 'show', ':' + name])
    if len(data) > 2 * 1024 * 1024:
        errors.append((name, 'large file'))
    text = data.decode('utf-8', errors='replace')
    if name != 'scripts/check_public.py':
        for pattern in patterns:
            if re.search(pattern, text):
                errors.append((name, 'potential sensitive content'))
                break
if errors:
    for name, kind in errors:
        print(name + ': ' + kind)
    raise SystemExit(1)
print(f'Public artifact check passed: {len(names)} staged files')
