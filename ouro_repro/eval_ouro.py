"""Fixed-depth evaluation using the pinned lm-eval task/scoring implementation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault('HF_HOME', str(Path('runtime/hf').resolve()))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_CACHE', str(Path('runtime/data-cache').resolve()))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator, utils
from lm_eval.models.huggingface import HFLM
import lm_eval
from .train import revision


def load_model(loops, dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(
        'runtime/model', trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype, attn_implementation='sdpa')
    model.config.total_ut_steps = loops
    model.model.total_ut_steps = loops
    model.config.use_cache = False
    # Upstream exit_at_step is zero-based. Force the final executed depth;
    # a floating-point CDF threshold of 1 is not an exact static-depth guarantee.
    model.early_exit_step = loops - 1
    model.early_exit_threshold = 1.0
    model.eval().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained('runtime/model', local_files_only=True)
    return model, tokenizer


def task_configuration(name, smoke=False):
    package = Path(lm_eval.__file__).parent / 'tasks'
    filename = package / ('arc/arc_challenge.yaml' if name == 'arc_challenge' else 'winogrande/default.yaml')
    cfg = utils.load_yaml_config(str(filename))
    key = 'ARC-Challenge' if name == 'arc_challenge' else 'winogrande_xl'
    manifest = json.loads(Path('runtime/datasets/manifest.json').read_text())[key]
    cfg['dataset_path'] = 'parquet'
    cfg['dataset_name'] = None
    cfg['dataset_kwargs'] = {'data_files': manifest['files']}
    # ARC smoke uses validation, not its held-out test. Winogrande's labeled
    # evaluation split is validation, so use a small train slice for smoke only.
    if smoke:
        cfg['test_split'] = 'validation' if name == 'arc_challenge' else 'train'
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--loops', type=int, required=True)
    p.add_argument('--task', choices=['arc_challenge', 'winogrande'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.limit and not args.smoke:
        raise ValueError('Limits are allowed only in smoke runs')
    commit = revision()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, tokenizer = load_model(args.loops)
    count = [0]
    hook = model.model.layers[0].register_forward_pre_hook(lambda *a: count.__setitem__(0, count[0] + 1))
    with torch.inference_mode():
        ids = tokenizer('The capital of France is', return_tensors='pt').input_ids.cuda()
        probe = model(ids, use_cache=False).logits
        assert torch.isfinite(probe).all()
    assert count[0] == args.loops, (count[0], args.loops)
    hook.remove()
    harness = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1,
                   max_length=8192, device='cuda', softmax_dtype='float32',
                   trust_remote_code=True, add_bos_token=False)
    # Detect truncation rather than quietly changing the 25-shot protocol.
    lengths = {'maximum': 0, 'calls': 0}
    encode = harness._encode_pair
    def checked_encode(context, continuation, **kwargs):
        a, b = encode(context, continuation, **kwargs)
        lengths['maximum'] = max(lengths['maximum'], len(a) + len(b))
        lengths['calls'] += 1
        if len(a) + len(b) > harness.max_length:
            raise RuntimeError('Prompt exceeds frozen context limit')
        return a, b
    harness._encode_pair = checked_encode
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    result = evaluator.simple_evaluate(
        model=harness, tasks=[task_configuration(args.task, args.smoke)],
        num_fewshot=25 if args.task == 'arc_challenge' else 5,
        limit=args.limit if args.smoke else None, bootstrap_iters=1000,
        log_samples=True, random_seed=0, numpy_random_seed=1234,
        torch_random_seed=1234, fewshot_random_seed=1234,
        apply_chat_template=False)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    # Full raw prompts and evaluator output remain local and ignored.
    (output / 'raw-results.json').write_text(json.dumps(result, default=str) + '\n')
    sample_rows = []
    for item in result['samples'][args.task]:
        sample_rows.append({k: item[k] for k in ('doc_id', 'doc_hash', 'target', 'acc', 'acc_norm') if k in item})
    (output / 'paired-results.json').write_text(json.dumps(sample_rows, indent=2) + '\n')
    report = {'run_id': f'ouro-20261026-{args.task}-r{args.loops}' + ('-smoke' if args.smoke else ''),
              'run_commit': commit, 'status': 'SUCCEEDED', 'loops': args.loops,
              'task': args.task, 'smoke': args.smoke, 'metrics': result['results'][args.task],
              'n_samples': len(sample_rows), 'elapsed_seconds': seconds,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'tokenized_lengths': lengths, 'device': torch.cuda.get_device_name(0),
              'precision': 'fp16', 'cache': False, 'attention': 'sdpa',
              'upstream': json.loads(Path('runtime/upstream.json').read_text()),
              'versions': {name: importlib.metadata.version(name) for name in
                           ['torch', 'transformers', 'lm_eval', 'datasets']}}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
