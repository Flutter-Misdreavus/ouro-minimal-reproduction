"""Download pinned public artifacts into ignored runtime storage; never print tokens."""
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault('HF_HOME', str(Path('runtime/hf').resolve()))
from huggingface_hub import HfApi, snapshot_download

root = Path('runtime')
root.mkdir(exist_ok=True)
api = HfApi(token=False)
name = 'ByteDance/Ouro-1.4B'
def get_json(url):
    return json.loads(subprocess.check_output(['curl', '-fsSL', '--connect-timeout', '15',
                                              '--max-time', '60', url]))

info = get_json('https://huggingface.co/api/models/' + name)
manifest = {'model': name, 'revision': info['sha'], 'license': 'apache-2.0',
            'dataset_revisions': {dataset: get_json('https://huggingface.co/api/datasets/' + dataset)['sha']
                                  for dataset in ['allenai/ai2_arc', 'allenai/winogrande']}}
(root / 'upstream.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest), flush=True)
destination = root / 'model'
destination.mkdir(exist_ok=True)
for item in info['siblings']:
    filename = item['rfilename']
    if Path(filename).suffix not in {'.json', '.py', '.safetensors', '.model', '.txt', '.jinja'}:
        continue
    target = destination / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    print('Downloading ' + filename, flush=True)
    subprocess.run(['curl', '-fL', '--retry', '2', '--connect-timeout', '15', '--max-time', '1200',
                    '-o', str(target) + '.partial',
                    f'https://huggingface.co/{name}/resolve/{info["sha"]}/{filename}'], check=True)
    Path(str(target) + '.partial').replace(target)
print('DOWNLOAD_COMPLETE', flush=True)
