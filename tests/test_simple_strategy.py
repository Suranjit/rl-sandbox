from pong.strategy.simple import SimpleStepStrategy
from pong.constants import PADDLE_SPEED, HEIGHT, PADDLE_HEIGHT


def test_simple_step_moves_paddle_up(dummy_env):
    """Action == 1 should move the player paddle up by ``PADDLE_SPEED``."""
    strat = SimpleStepStrategy({})

    old_y = dummy_env.player_paddle.y
    strat.execute(dummy_env, action=1)

    expected = max(old_y - PADDLE_SPEED, 0)
    assert dummy_env.player_paddle.y == expected


def test_simple_step_moves_paddle_down(dummy_env):
    """Action == 2 should move the player paddle down by ``PADDLE_SPEED``."""
    strat = SimpleStepStrategy({})

    old_y = dummy_env.player_paddle.y
    strat.execute(dummy_env, action=2)

    expected = min(old_y + PADDLE_SPEED, HEIGHT - PADDLE_HEIGHT)
    assert dummy_env.player_paddle.y == expected
