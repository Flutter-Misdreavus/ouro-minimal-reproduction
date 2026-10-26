"""Prefix arithmetic over F_23; uniform full binary tree shapes at fixed size."""
import math
import random

PAD, BOS, ANS, NUMBER, ADD, SUB, MUL, LENGTH = 0, 1, 2, 3, 26, 27, 28, 29


def evaluate(tokens):
    """Independent stack oracle; also rejects malformed expressions."""
    stack = []
    for token in reversed(tokens):
        if NUMBER <= token < NUMBER + 23:
            stack.append(token - NUMBER)
        elif token in (ADD, SUB, MUL):
            if len(stack) < 2:
                raise ValueError("invalid expression")
            a, b = stack.pop(), stack.pop()
            stack.append(({ADD: lambda: a + b, SUB: lambda: a - b,
                           MUL: lambda: a * b}[token])() % 23)
        else:
            raise ValueError("invalid token")
    if len(stack) != 1:
        raise ValueError("invalid expression")
    return stack[0]


class Generator:
    def __init__(self, seed, max_ops=10):
        self.rng = random.Random(seed)
        self.max_ops = max_ops
        self.catalan = [math.comb(2 * n, n) // (n + 1) for n in range(max_ops + 1)]
        self.weights = {n: [self.catalan[k] * self.catalan[n - 1 - k]
                           for k in range(n)] for n in range(1, max_ops + 1)}

    def tree(self, n):
        if n == 0:
            value = self.rng.randrange(23)
            return [NUMBER + value], value
        k = self.rng.choices(range(n), weights=self.weights[n])[0]
        left, a = self.tree(k)
        right, b = self.tree(n - k - 1)
        op = self.rng.choice((ADD, SUB, MUL))
        value = (a + b if op == ADD else a - b if op == SUB else a * b) % 23
        return [op] + left + right, value

    def sample(self, n=None):
        n = self.rng.randint(1, self.max_ops) if n is None else n
        expression, answer = self.tree(n)
        return [BOS, LENGTH + n] + expression + [ANS], answer


def fixed_set(seed, count, max_ops=10, excluded=()):
    generator = Generator(seed, max_ops)
    seen = set(excluded)
    result = []
    while len(result) < count:
        x, y = generator.sample(max_ops)
        key = tuple(x)
        if key not in seen:
            assert evaluate(x[2:-1]) == y
            seen.add(key)
            result.append((x, y))
    return result


def batch_tensor(examples, device):
    import torch
    width = max(len(x) for x, _ in examples)
    # Pad to a multiple of 8 for V100 tensor cores. No cross-example packing.
    width = (width + 7) // 8 * 8
    x = torch.zeros(len(examples), width, dtype=torch.long)
    positions = torch.empty(len(examples), dtype=torch.long)
    y = torch.empty(len(examples), dtype=torch.long)
    for i, (tokens, target) in enumerate(examples):
        x[i, :len(tokens)] = torch.tensor(tokens)
        positions[i] = len(tokens) - 1
        y[i] = NUMBER + target
    return x.to(device), positions.to(device), y.to(device)
