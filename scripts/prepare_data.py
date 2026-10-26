"""Fetch revision-pinned public parquet splits, with size/hash verification."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('runtime/datasets')
root.mkdir(parents=True, exist_ok=True)
upstream = json.loads(Path('runtime/upstream.json').read_text())
manifest = {}
for repo, config in [('allenai/ai2_arc', 'ARC-Challenge'), ('allenai/winogrande', 'winogrande_xl')]:
    revision = upstream['dataset_revisions'][repo]
    url = f'https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true'
    info = json.loads(subprocess.check_output(['curl', '-fsSL', '--retry', '2', '--retry-all-errors',
                                              '--connect-timeout', '15', '--max-time', '90', url]))
    assert info['sha'] == revision
    files = {}
    for item in info['siblings']:
        name = item['rfilename']
        if not name.startswith(config + '/') or not name.endswith('.parquet'):
            continue
        split = Path(name).name.split('-')[0]
        if split not in ('train', 'validation', 'test'):
            continue
        target = root / config / Path(name).name
        target.parent.mkdir(exist_ok=True)
        subprocess.run(['curl', '-fsSL', '--retry', '2', '--retry-all-errors', '--connect-timeout', '15',
                        '--max-time', '180', '-o', str(target),
                        f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}'], check=True)
        checksum = hashlib.sha256(target.read_bytes()).hexdigest()
        if item.get('lfs'):
            assert checksum == item['lfs']['sha256']
        assert target.stat().st_size == item['size']
        files.setdefault(split, []).append(str(target))
        print(json.dumps({'repo': repo, 'file': name, 'sha256': checksum}), flush=True)
    if not files:
        raise RuntimeError('No matching parquet files: ' + repo)
    manifest[config] = {'repository': repo, 'revision': revision, 'files': files}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('DATA_READY', flush=True)
