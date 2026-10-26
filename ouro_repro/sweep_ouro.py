"""Exact nested-depth sweep: share the backbone pass, retain lm-eval tokenization.

Earlier loop states are identical to a truncated model forward without KV reuse.
Only continuation-token logits are projected. Saved timing is for the entire
multi-depth pass, never presented as the cost of an individual early exit.
"""
import argparse
import json
from pathlib import Path
import time

from .eval_ouro import load_model, task_configuration
from .train import revision
import torch
from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM

DEPTHS = (1, 2, 3, 4, 6, 8)


class SharedDepthHFLM(HFLM):
    def __init__(self, model, tokenizer):
        super().__init__(pretrained=model, tokenizer=tokenizer, batch_size=1,
                         max_length=8192, device='cuda', softmax_dtype='float32',
                         trust_remote_code=True, add_bos_token=False)
        self.depth = 8
        self.scores = {}
        self.backbone_calls = 0
        self.maximum_tokens = 0
        self.allow_compute = True

    @torch.inference_mode()
    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        results = []
        for i, (key, context, continuation) in enumerate(requests):
            # The public tasks use non-empty contexts and continuations.
            assert context and continuation
            assert len(context) + len(continuation) <= self.max_length
            cache_key = (tuple(context), tuple(continuation))
            if cache_key not in self.scores:
                if not self.allow_compute:
                    raise RuntimeError('Tokenized requests changed between depth evaluations')
                ids = torch.tensor([context + continuation[:-1]], device=self.device)
                self.maximum_tokens = max(self.maximum_tokens, len(context) + len(continuation))
                _, states, _ = self.model.model(ids, use_cache=False)
                assert len(states) == 8
                self.backbone_calls += 1
                target = torch.tensor(continuation, device=self.device)
                scores = {}
                for depth in DEPTHS:
                    selected = states[depth - 1][:, len(context) - 1:, :]
                    logits = self.model.lm_head(selected).float().squeeze(0)
                    if not torch.isfinite(logits).all():
                        raise RuntimeError('Non-finite continuation logits')
                    logprobs = logits.log_softmax(-1)
                    value = logprobs.gather(-1, target[:, None]).sum().item()
                    exact = bool((logits.argmax(-1) == target).all().item())
                    scores[depth] = (value, exact)
                self.scores[cache_key] = scores
            results.append(self.scores[cache_key][self.depth])
            if self.allow_compute and (i + 1) % 100 == 0:
                print(f'SCORING {i + 1}/{len(requests)}', flush=True)
        return results


def verify(model, tokenizer, shared):
    raw = json.loads(Path('runtime/arc-smoke/raw-results.json').read_text())
    samples = raw['samples']['arc_challenge'][:2]
    pairs = [pair for sample in samples for pair in sample['arguments']]
    instances = [Instance(request_type='loglikelihood', doc={}, arguments=tuple(pair), idx=i)
                 for i, pair in enumerate(pairs)]
    shared.depth = 8
    shared.loglikelihood(instances)
    reference = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1,
                     max_length=8192, softmax_dtype='float32', add_bos_token=False)
    report = {}
    for depth in DEPTHS:
        model.model.total_ut_steps = model.config.total_ut_steps = depth
        model.early_exit_step = depth - 1
        started = time.monotonic()
        expected = reference.loglikelihood(instances)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        shared.depth = depth
        actual = shared.loglikelihood(instances)
        errors = [abs(a[0] - b[0]) for a, b in zip(expected, actual)]
        assert max(errors) < 0.02, (depth, errors)
        assert [x[1] for x in expected] == [x[1] for x in actual]
        for offset in (0, 4):
            assert max(range(4), key=lambda i: expected[offset + i][0]) == max(
                range(4), key=lambda i: actual[offset + i][0])
        report[str(depth)] = {'max_absolute_loglikelihood_error': max(errors),
                              'reference_seconds_for_8_requests': seconds}
    model.model.total_ut_steps = model.config.total_ut_steps = 8
    model.early_exit_step = 7
    shared.scores.clear()
    shared.backbone_calls = 0
    shared.maximum_tokens = 0
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    commit = revision()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, tokenizer = load_model(8)
    shared = SharedDepthHFLM(model, tokenizer)
    checked = verify(model, tokenizer, shared)
    report = {'run_id': 'ouro-20261026-shared-depth' + ('-smoke' if args.smoke else ''),
              'run_commit': commit, 'status': 'RUNNING', 'verification': checked,
              'upstream': json.loads(Path('runtime/upstream.json').read_text()),
              'precision': 'fp16', 'likelihood_softmax': 'fp32', 'tasks': {},
              'timing_note': 'Backbone computations shared across depths; elapsed time is whole sweep cost'}
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.verify_only:
        report['status'] = 'SUCCEEDED'
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        return
    started = time.monotonic()
    for task in ('arc_challenge', 'winogrande'):
        shared.scores.clear()
        shared.allow_compute = True
        before = shared.backbone_calls
        for depth in (8, 1, 2, 3, 4, 6):
            shared.depth = depth
            result = evaluator.simple_evaluate(
                model=shared, tasks=[task_configuration(task, args.smoke)],
                num_fewshot=25 if task == 'arc_challenge' else 5,
                limit=32 if args.smoke else None, bootstrap_iters=1000,
                log_samples=True, random_seed=0, numpy_random_seed=1234,
                torch_random_seed=1234, fewshot_random_seed=1234,
                apply_chat_template=False)
            shared.allow_compute = False
            raw = out / f'{task}-r{depth}-raw.json'
            raw.write_text(json.dumps(result, default=str) + '\n')
            rows = [{k: sample[k] for k in ('doc_id', 'doc_hash', 'acc', 'acc_norm') if k in sample}
                    for sample in result['samples'][task]]
            (out / f'{task}-r{depth}-paired.json').write_text(json.dumps(rows, indent=2) + '\n')
            report['tasks'].setdefault(task, {})[depth] = {
                'metrics': result['results'][task], 'n': len(rows)}
            print(json.dumps({'task': task, 'depth': depth, 'metrics': result['results'][task]}), flush=True)
            (out / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
        report['tasks'][task]['backbone_calls'] = shared.backbone_calls - before
    report.update(status='SUCCEEDED', elapsed_seconds=time.monotonic() - started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  maximum_tokens=shared.maximum_tokens, backbone_calls=shared.backbone_calls)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
