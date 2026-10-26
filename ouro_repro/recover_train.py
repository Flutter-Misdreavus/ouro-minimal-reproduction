"""Frozen long-expression recovery experiments; historical trainer stays intact."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .mano import Generator, batch_tensor, fixed_set
from .model import Model
from .train import digest, evaluate, revision


def build_evaluation(cfg):
    primary = fixed_set(1729, cfg['validation_size'], 10, tree_distribution='uniform_split')
    legacy = fixed_set(1729, 1024, 10)
    original_test = fixed_set(2718, 10000, 10, [tuple(x) for x, _ in legacy])
    original_excluded = {tuple(x) for x, _ in primary + legacy + original_test}
    reserved_test = fixed_set(2718, 10000, 10, original_excluded, 'uniform_split')
    excluded = original_excluded | {tuple(x) for x, _ in reserved_test}
    diagnostics = {}
    for length in (1, 2, 4):
        gen = Generator(9000 + length, 10, 'uniform_split')
        diagnostics[str(length)] = [gen.sample(length) for _ in range(512)]
    return primary, legacy, reserved_test, excluded, diagnostics


def sample_training(gen, step, warmup_data_steps, excluded, batch_size):
    examples = []
    while len(examples) < batch_size:
        example = gen.sample(1) if step <= warmup_data_steps else gen.sample()
        if tuple(example[0]) not in excluded:
            examples.append(example)
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    commit = revision()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    assert cfg['precision'] == 'fp32' and cfg['steps'] > cfg['warmup_steps']
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    primary, legacy, test, excluded, diagnostic = build_evaluation(cfg)
    model = Model(cfg['layers'], cfg['loops'], cfg['width'], cfg['heads'], 40).cuda()
    initial_sha = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.98), eps=1e-6, weight_decay=0.1)
    generator = Generator(10000 + cfg['seed'], 10, cfg['tree_distribution'])
    metadata = {'run_id': cfg['run_id'], 'run_commit': commit, 'status': 'RUNNING', 'config': cfg,
                'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                'initial_parameters_sha256': initial_sha,
                'validation_sha256': digest(primary), 'legacy_validation_sha256': digest(legacy),
                'reserved_test_sha256': digest(test), 'test_evaluated': False,
                'heldout_expressions': len(excluded), 'device': torch.cuda.get_device_name(0),
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'precision': 'fp32', 'torch': torch.__version__,
                'diagnostic_note': 'Lengths1/2/4 may overlap training; diagnostic fit, not generalization'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    started = time.monotonic()
    history = []
    tokens = 0
    stream = hashlib.sha256()
    length_counts = {str(k): 0 for k in range(1, 11)}
    def record(step, loss=None, gradient=None):
        row = {'step': step, 'validation': evaluate(model, primary, 'cuda', False),
               'legacy_validation': evaluate(model, legacy, 'cuda', False),
               'diagnostics': {k: evaluate(model, v, 'cuda', False) for k, v in diagnostic.items()},
               'batch_ce': loss, 'gradient_norm': gradient, 'nonpad_tokens': tokens,
               'seconds': time.monotonic() - started}
        history.append(row)
        (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    record(0)
    for step in range(1, cfg['steps'] + 1):
        if time.monotonic() - started > cfg['wall_seconds']:
            raise RuntimeError('Recovery run wall budget exceeded')
        batch = sample_training(generator, step, cfg['atomic_warmup_steps'], excluded, cfg['batch_size'])
        stream.update(json.dumps(batch, separators=(',', ':')).encode())
        tokens += sum(len(x) + 1 for x, _ in batch)
        for x, _ in batch:
            length_counts[str(x[1] - 29)] += 1
        x, pos, y = batch_tensor(batch, 'cuda')
        warmup = cfg['warmup_steps']
        factor = step / warmup if step <= warmup else 0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / (cfg['steps'] - warmup)))
        for group in opt.param_groups:
            group['lr'] = cfg['lr'] * factor
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x, pos), y)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(norm)
        opt.step()
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            record(step, loss.item(), float(norm))
    torch.cuda.synchronize()
    metadata.update({'status': 'SUCCEEDED', 'completed_steps': step, 'nonpad_tokens': tokens,
                     'train_stream_sha256': stream.hexdigest(), 'length_counts': length_counts,
                     'validation': history[-1]['validation'], 'legacy_validation': history[-1]['legacy_validation'],
                     'diagnostics': history[-1]['diagnostics'], 'elapsed_seconds': time.monotonic() - started,
                     'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'step': step, 'config': cfg,
                'generator_rng': generator.rng.getstate(), 'torch_rng': torch.get_rng_state()}, out / 'checkpoint.pt')
    (out / 'report.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
