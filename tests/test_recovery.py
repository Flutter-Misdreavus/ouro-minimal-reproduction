import pytest
from ouro_repro.mano import Generator, evaluate, fixed_set
from ouro_repro.train import digest
from ouro_repro.recover_train import sample_training, build_evaluation


def test_historical_generator_stream_unchanged():
    gen = Generator(10000, 10)
    assert digest([gen.sample() for _ in range(1000)]) == 'a4fbd4e18249eb99bfbb58c070d519c8f2027a5d5f2ae0e37765aad16cca53c8'


def test_uniform_split_oracle_and_heldout_filter():
    gen = Generator(42, 10, 'uniform_split')
    for _ in range(1000):
        x, y = gen.sample()
        assert evaluate(x[2:-1]) == y
    excluded = {tuple(Generator(0).sample(1)[0])}
    first = sample_training(gen, 1, 2, excluded, 256)
    later = sample_training(gen, 3, 2, excluded, 256)
    assert all(x[1] == 30 for x, _ in first)
    assert len({x[1] for x, _ in later}) == 10
    assert all(tuple(x) not in excluded for x, _ in first + later)
    with pytest.raises(ValueError):
        fixed_set(1, 1588, 1)


def test_progressive_curriculum_and_composition_holdouts():
    cfg = {'task_max_ops': 4, 'validation_size': 128, 'holdout_composition_diagnostics': True}
    primary, legacy, test, excluded, diagnostic = build_evaluation(cfg)
    assert {tuple(x) for x, _ in primary + legacy + test}.issubset(excluded)
    assert {tuple(x) for x, _ in diagnostic['2'] + diagnostic['4']}.issubset(excluded)
    assert not {tuple(x) for x, _ in primary} & {tuple(x) for x, _ in test}
    gen = Generator(123, 4, 'uniform_split')
    early = sample_training(gen, 1, 2, excluded, 256, 4)
    middle = sample_training(gen, 3, 2, excluded, 256, 4)
    late = sample_training(gen, 5, 2, excluded, 256, 4)
    assert {x[1] - 29 for x, _ in early} == {1}
    assert {x[1] - 29 for x, _ in middle} == {1, 2}
    assert {x[1] - 29 for x, _ in late} == {1, 2, 3, 4}
    assert all(tuple(x) not in excluded for x, _ in early + middle + late)
