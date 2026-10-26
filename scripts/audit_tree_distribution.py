"""Exact tree-height distribution and singleton stream audit for the two samplers."""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

from ouro_repro.mano import Generator
from ouro_repro.train import digest


def catalan(n):
    return math.comb(2 * n, n) // (n + 1)


@lru_cache(None)
def height_cdf(n, height, mode):
    if n == 0:
        return 1.
    if height <= 0:
        return 0.
    weights = [catalan(k) * catalan(n - 1 - k) if mode == 'catalan' else 1 for k in range(n)]
    return sum(w * height_cdf(k, height - 1, mode) * height_cdf(n - 1 - k, height - 1, mode)
               for k, w in enumerate(weights)) / sum(weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = {}
    for mode in ('catalan', 'uniform_split'):
        assert height_cdf(1, 1, mode) == 1.
        assert height_cdf(2, 1, mode) == 0.
        assert height_cdf(2, 2, mode) == 1.
        g = Generator(10000, 10, mode)
        result[mode] = {'expected_height_at_10_ops': sum(1 - height_cdf(10, h, mode) for h in range(10)),
                        'probability_root_has_leaf_child_at_10_ops': 2 * catalan(9) / catalan(10) if mode == 'catalan' else .2,
                        'singleton_stream_sha256': digest([g.sample(1) for _ in range(1000)])}
    result['note'] = 'Singleton supports are identical, but RNG consumption changes sample order; one seed does not isolate tree shape from sampling noise.'
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
