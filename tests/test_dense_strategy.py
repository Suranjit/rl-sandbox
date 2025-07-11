
from pong.strategy.dense import DenseRewardStepStrategy
from pong.constants import PADDLE_HEIGHT


def test_dense_alignment_reward_positive(dummy_env):
    """With perfect vertical alignment, the shaping reward should be ≥ 0.01."""
    # Place paddle exactly aligned with the ball for max alignment (≈ 1.0)
    dummy_env.player_paddle.y = dummy_env.ball.centery - PADDLE_HEIGHT / 2

    strat = DenseRewardStepStrategy({})
    _, reward, terminated, truncated, _ = strat.execute(dummy_env, action=0)

    assert terminated is False and truncated is False  # should be an ordinary step
    # Dense reward component is 0.01 * alignment (≈ 0.01). Allow tiny fp tolerance.
    assert reward >= 0.009
