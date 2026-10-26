"""Controlled one-operation Mano learning probe, with a finite disjoint split."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .mano import ADD, SUB, MUL, BOS, ANS, LENGTH, NUMBER, batch_tensor, evaluate
from .model import Model
from .train import digest, revision


def atomic_splits(seed=4242):
    """Exactly 423/53/53 examples per operator; no rejection-sampling loop."""
    result = {name: [] for name in ('train', 'validation', 'test')}
    rng = random.Random(seed)
    for op in (ADD, SUB, MUL):
        examples = []
        for a in range(23):
            for b in range(23):
                answer = (a + b if op == ADD else a - b if op == SUB else a * b) % 23
                tokens = [BOS, LENGTH + 1, op, NUMBER + a, NUMBER + b, ANS]
                assert evaluate(tokens[2:-1]) == answer
                examples.append((tokens, answer))
        rng.shuffle(examples)
        for name, part in zip(result, (examples[:423], examples[423:476], examples[476:])):
            result[name].extend(part)
    return result


@torch.no_grad()
def score(model, tensors, amp):
    x, positions, labels = tensors
    model.eval()
    predictions, losses = [], []
    for offset in range(0, len(x), 256):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            logits = model(x[offset:offset + 256], positions[offset:offset + 256])
        assert torch.isfinite(logits).all()
        predictions.append(logits.argmax(-1))
        losses.append(F.cross_entropy(logits.float(), labels[offset:offset + 256], reduction='none'))
    pred = torch.cat(predictions)
    correct = pred == labels
    result = {'accuracy': correct.float().mean().item(), 'ce': torch.cat(losses).mean().item(),
              'n': len(labels), 'per_operator': {}}
    for op, name in ((ADD, 'add'), (SUB, 'sub'), (MUL, 'mul')):
        mask = x[:, 2] == op
        result['per_operator'][name] = correct[mask].float().mean().item()
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    commit = revision()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    split = atomic_splits()
    tensors = {k: batch_tensor(v, 'cuda') for k, v in split.items() if k != 'test'}
    model = Model(cfg['layers'], cfg['loops'], cfg['width'], cfg['heads'], 40).cuda()
    initial_sha = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1)
    amp = cfg['precision'] == 'fp16'
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    sampler = torch.Generator().manual_seed(10000 + cfg['seed'])
    metadata = {'run_id': cfg['run_id'], 'run_commit': commit, 'status': 'RUNNING',
                'config': cfg, 'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                'initial_parameters_sha256': initial_sha, 'splits_sha256': {k: digest(v) for k, v in split.items()},
                'split_sizes': {k: len(v) for k, v in split.items()}, 'device': torch.cuda.get_device_name(0),
                'parameter_count': sum(p.numel() for p in model.parameters()), 'test_evaluated': False}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    torch.cuda.synchronize()
    start = time.monotonic()
    history = []
    skipped = 0
    stream = hashlib.sha256()
    def record(step, loss=None, norm=None):
        row = {'step': step, 'train': score(model, tensors['train'], amp),
               'validation': score(model, tensors['validation'], amp),
               'seconds': time.monotonic() - start, 'skipped_updates': skipped,
               'batch_ce': loss, 'gradient_norm': norm}
        history.append(row)
        (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    record(0)
    for step in range(1, cfg['steps'] + 1):
        if time.monotonic() - start > cfg['wall_seconds']:
            raise RuntimeError('Atomic probe wall budget exceeded')
        indices = torch.randint(len(split['train']), (cfg['batch_size'],), generator=sampler)
        stream.update(indices.numpy().tobytes())
        indices = indices.cuda()
        x, pos, y = [t[indices] for t in tensors['train']]
        warmup = cfg['warmup_steps']
        factor = step / warmup if step <= warmup else 0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / (cfg['steps'] - warmup)))
        for group in opt.param_groups:
            group['lr'] = cfg['lr'] * factor
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(model(x, pos).float(), y)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped += int(scaler.get_scale() < scale)
        if not amp and not torch.isfinite(norm):
            raise RuntimeError('Non-finite FP32 gradient')
        if skipped > 20:
            raise RuntimeError('Too many FP16 overflow skips')
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            record(step, loss.item(), float(norm) if torch.isfinite(norm) else None)
    torch.cuda.synchronize()
    metadata.update({'status': 'SUCCEEDED', 'completed_steps': step, 'train': history[-1]['train'],
                     'validation': history[-1]['validation'], 'skipped_updates': skipped,
                     'sampler_sha256': stream.hexdigest(), 'elapsed_seconds': time.monotonic() - start,
                     'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    torch.save({'model': model.state_dict(), 'config': cfg, 'step': step}, out / 'checkpoint.pt')
    (out / 'report.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
