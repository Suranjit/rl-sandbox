import pytest

from pong.strategy import make
from pong.strategy.simple import SimpleStepStrategy
from pong.strategy.dense import DenseRewardStepStrategy


def test_strategy_factory_returns_correct_instances():
    s1 = make("simple")
    s2 = make("dense")
    assert isinstance(s1, SimpleStepStrategy)
    assert isinstance(s2, DenseRewardStepStrategy)


def test_strategy_factory_raises_on_unknown():
    with pytest.raises(ValueError):
        make("definitely-not-real")
