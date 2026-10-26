"""Bounded parallel public download; final content must match upstream SHA256."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

info = json.loads(Path('runtime/model-info.json').read_text())
upstream = json.loads(Path('runtime/upstream.json').read_text())
assert info['sha'] == upstream['revision']
item = next(x for x in info['siblings'] if x['rfilename'] == 'model.safetensors')
size, expected = item['size'], item['lfs']['sha256']
url = f'https://huggingface.co/{upstream["model"]}/resolve/{info["sha"]}/model.safetensors'
root = Path('runtime/weight-parts')
root.mkdir(exist_ok=True)
chunk = 32 * 1024 * 1024

def fetch(start):
    end = min(start + chunk, size) - 1
    target = root / str(start)
    if not target.exists() or target.stat().st_size != end - start + 1:
        subprocess.run(['curl', '-fsSL', '--retry', '2', '--retry-all-errors',
                        '--connect-timeout', '15', '--max-time', '240',
                        '--range', f'{start}-{end}', '-o', str(target) + '.partial', url], check=True)
        partial = Path(str(target) + '.partial')
        assert partial.stat().st_size == end - start + 1, 'Server did not honor byte range'
        partial.replace(target)
    print(f'CHUNK_READY {start}', flush=True)
    return target

with ThreadPoolExecutor(max_workers=8) as pool:
    paths = list(pool.map(fetch, range(0, size, chunk)))
digest = hashlib.sha256()
destination = Path('runtime/model/model.safetensors.assembled')
with destination.open('wb') as out:
    for path in paths:
        with path.open('rb') as source:
            while data := source.read(8 * 1024 * 1024):
                digest.update(data)
                out.write(data)
assert digest.hexdigest() == expected, 'Weight SHA256 mismatch'
destination.replace('runtime/model/model.safetensors')
print(json.dumps({'status': 'VERIFIED', 'sha256': expected, 'bytes': size}), flush=True)
