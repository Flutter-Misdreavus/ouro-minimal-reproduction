import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .mano import Generator, batch_tensor, fixed_set
from .model import Model


def revision():
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip():
        raise RuntimeError('Commit tracked changes before running')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def digest(examples):
    return hashlib.sha256(json.dumps(examples, separators=(',', ':')).encode()).hexdigest()


@torch.no_grad()
def evaluate(model, examples, device, amp):
    model.eval()
    correct = 0
    loss = 0.
    for offset in range(0, len(examples), 256):
        x, positions, y = batch_tensor(examples[offset:offset + 256], device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            logits = model(x, positions)
        loss += F.cross_entropy(logits.float(), y, reduction='sum').item()
        correct += (logits.argmax(-1) == y).sum().item()
    model.train()
    return {'accuracy': correct / len(examples), 'ce': loss / len(examples), 'n': len(examples)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    commit = revision()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg.get('require_cuda', True) and device != 'cuda':
        raise RuntimeError('CUDA is required for this experiment')
    amp = cfg.get('precision', 'fp16') == 'fp16' and device == 'cuda'
    model = Model(layers=cfg['layers'], loops=cfg['loops'], width=cfg['width'],
                  heads=cfg['heads'], vocab=30 + cfg['max_ops']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.98),
                           eps=1e-6, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    validation = fixed_set(1729, cfg['validation_size'], cfg['max_ops'])
    test = fixed_set(2718, cfg['test_size'], cfg['max_ops'], [tuple(x) for x, _ in validation])
    heldout = {tuple(x) for x, _ in validation + test}
    gen = Generator(10000 + cfg['seed'], cfg['max_ops'])
    metadata = {'run_id': cfg['run_id'], 'run_commit': commit, 'status': 'RUNNING',
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'validation_sha256': digest(validation), 'test_sha256': digest(test),
                'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
                'precision': cfg.get('precision', 'fp16'), 'torch': torch.__version__,
                'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest()}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    started = time.monotonic()
    nonpad_tokens = 0
    skipped = 0
    history = []
    initial = evaluate(model, validation, device, amp)
    history.append({'step': 0, 'validation': initial, 'seconds': time.monotonic() - started})
    print(json.dumps(history[-1]), flush=True)
    completed = 0
    train_hash = hashlib.sha256()
    for step in range(1, cfg['steps'] + 1):
        if time.monotonic() - started > cfg['wall_seconds']:
            metadata['status'] = 'BUDGET_EXHAUSTED'
            break
        batch = []
        while len(batch) < cfg['batch_size']:
            example = gen.sample()
            if tuple(example[0]) not in heldout:
                batch.append(example)
        train_hash.update(json.dumps(batch, separators=(',', ':')).encode())
        nonpad_tokens += sum(len(x) + 1 for x, _ in batch)
        x, positions, y = batch_tensor(batch, device)
        warmup = cfg['warmup_steps']
        factor = step / warmup if step <= warmup else 0.1 + 0.45 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, cfg['steps'] - warmup)))
        for group in opt.param_groups:
            group['lr'] = cfg['lr'] * factor
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            logits = model(x, positions)
            loss = F.cross_entropy(logits.float(), y)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        previous_scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped += int(scaler.get_scale() < previous_scale)
        if skipped > 20:
            raise RuntimeError('Repeated FP16 gradient overflows')
        completed = step
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            row = {'step': step, 'train_ce': loss.item(), 'gradient_norm': float(norm),
                   'validation': evaluate(model, validation, device, amp),
                   'seconds': time.monotonic() - started, 'nonpad_tokens': nonpad_tokens,
                   'skipped_updates': skipped}
            history.append(row)
            (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
            print(json.dumps(row), flush=True)
    if metadata['status'] == 'RUNNING':
        metadata['status'] = 'SUCCEEDED'
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'scaler': scaler.state_dict(), 'step': completed, 'config': cfg,
                'generator_rng': gen.rng.getstate(), 'torch_rng': torch.get_rng_state()},
               out / 'checkpoint.pt')
    metadata.update({'completed_steps': completed, 'nonpad_tokens': nonpad_tokens,
                     'train_stream_sha256': train_hash.hexdigest(), 'skipped_updates': skipped,
                     'validation': evaluate(model, validation, device, amp),
                     'elapsed_seconds': time.monotonic() - started,
                     'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0})
    if cfg.get('evaluate_test', False):
        metadata['test'] = evaluate(model, test, device, amp)
    (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (out / 'report.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
