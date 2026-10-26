import copy
import io

import pytest
import torch

from ouro_repro.mano import ADD, MUL, SUB, NUMBER, Generator, evaluate, fixed_set, batch_tensor
from ouro_repro.model import Model


def test_oracle_and_generator():
    assert evaluate([ADD, MUL, NUMBER + 2, NUMBER + 3, NUMBER + 4]) == 10
    assert evaluate([SUB, NUMBER + 2, NUMBER + 3]) == 22
    gen = Generator(3)
    for _ in range(1000):
        x, y = gen.sample()
        assert evaluate(x[2:-1]) == y
    with pytest.raises(ValueError):
        evaluate([ADD, NUMBER])


def test_tied_stack_equals_unrolled_copies():
    torch.manual_seed(0)
    loop = Model(3, 4, 32, 4)
    deep = Model(12, 1, 32, 4)
    deep.embedding.load_state_dict(loop.embedding.state_dict())
    deep.norm.load_state_dict(loop.norm.state_dict())
    for i, block in enumerate(deep.blocks):
        block.load_state_dict(loop.blocks[i % 3].state_dict())
    x, p, y = batch_tensor(fixed_set(9, 4), 'cpu')
    a, b = loop(x, p), deep(x, p)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    torch.nn.functional.cross_entropy(a, y).backward()
    torch.nn.functional.cross_entropy(b, y).backward()
    for i in range(3):
        for name, parameter in loop.blocks[i].named_parameters():
            total = sum(dict(deep.blocks[j].named_parameters())[name].grad for j in range(i, 12, 3))
            torch.testing.assert_close(parameter.grad, total, atol=1e-5, rtol=1e-4)


def test_causality_batch_independence_and_reload():
    torch.manual_seed(5)
    model = Model(2, 2, 32, 4)
    gen = Generator(4)
    examples = [gen.sample(2), gen.sample(10)]
    x, p, _ = batch_tensor(examples, 'cpu')
    logits = model(x, p)
    changed = x.clone()
    changed[0, p[0] + 1:] = 7
    changed[1, :] = 8
    torch.testing.assert_close(model(changed, p)[0], logits[0])
    single, sp, _ = batch_tensor(examples[:1], 'cpu')
    torch.testing.assert_close(model(single, sp)[0], logits[0], atol=1e-6, rtol=1e-5)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = Model(2, 2, 32, 4)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(x, p), logits)
