from ouro_repro.atomic_probe import atomic_splits
from ouro_repro.mano import ADD, SUB, MUL, evaluate


def test_atomic_partition_is_exhaustive_balanced_and_disjoint():
    split = atomic_splits()
    keys = {k: {tuple(x) for x, _ in v} for k, v in split.items()}
    assert [len(keys[k]) for k in split] == [1269, 159, 159]
    assert not keys['train'] & keys['validation']
    assert not keys['train'] & keys['test']
    assert not keys['validation'] & keys['test']
    assert len(set.union(*keys.values())) == 3 * 23 * 23
    for k, count in [('train', 423), ('validation', 53), ('test', 53)]:
        for op in (ADD, SUB, MUL):
            assert sum(x[2] == op for x, y in split[k]) == count
        assert all(evaluate(x[2:-1]) == y for x, y in split[k])
    assert atomic_splits() == split
