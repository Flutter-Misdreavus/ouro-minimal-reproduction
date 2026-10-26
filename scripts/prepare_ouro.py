"""Download pinned public artifacts into ignored runtime storage; never print tokens."""
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HOME', str(Path('runtime/hf').resolve()))
from huggingface_hub import HfApi, snapshot_download

root = Path('runtime')
root.mkdir(exist_ok=True)
api = HfApi()
name = 'ByteDance/Ouro-1.4B'
info = api.model_info(name)
manifest = {'model': name, 'revision': info.sha, 'license': 'apache-2.0',
            'dataset_revisions': {name: api.dataset_info(name).sha for name in
                                  ['allenai/ai2_arc', 'allenai/winogrande']}}
(root / 'upstream.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest), flush=True)
snapshot_download(name, revision=info.sha, local_dir='runtime/model',
                  allow_patterns=['*.json', '*.py', '*.safetensors', '*.model', '*.txt', '*.jinja'],
                  max_workers=4)
print('DOWNLOAD_COMPLETE', flush=True)
