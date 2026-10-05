import pytest
import torch

from config.sampling_strategy import (
    AdaptiveEpsilonGreedySamplingStrategy,
    DistSamplingStrategy,
    GreedySamplingStrategy,
)

PROBS = torch.tensor([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]])


def _empirical_frequencies(strategy, probs: torch.Tensor, samples: int = 20000) -> torch.Tensor:
    actions = strategy.sample_action(probs.expand(samples, *probs.shape))
    return torch.nn.functional.one_hot(actions, probs.size(-1)).float().mean(0)


def _epsilon_greedy(epsilon: float) -> AdaptiveEpsilonGreedySamplingStrategy:
    return AdaptiveEpsilonGreedySamplingStrategy(initial_epsilon=epsilon, min_epsilon=0.0, decay_factor=0.5)


@pytest.mark.parametrize("strategy", [DistSamplingStrategy(), GreedySamplingStrategy(), _epsilon_greedy(0.3)])
def test_behavior_probs_match_sampling(strategy):
    torch.manual_seed(0)
    behavior = strategy.behavior_probs(PROBS)

    assert torch.allclose(behavior.sum(-1), torch.ones(2))
    assert torch.allclose(_empirical_frequencies(strategy, PROBS), behavior, atol=0.02)


def test_epsilon_greedy_behavior_probs():
    behavior = _epsilon_greedy(0.3).behavior_probs(PROBS)
    assert torch.allclose(behavior, torch.tensor([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]]))


def test_epsilon_greedy_explores_per_worker():
    torch.manual_seed(0)
    probs = torch.tensor([[0.0, 1.0]]).expand(1000, 2)
    actions = _epsilon_greedy(0.5).sample_action(probs)
    # In a single call, some workers explore while others act greedily
    assert 0.15 < (actions == 0).float().mean() < 0.35


def test_epsilon_greedy_update():
    strategy = AdaptiveEpsilonGreedySamplingStrategy(initial_epsilon=0.8, min_epsilon=0.1, decay_factor=0.25)

    strategy.update(1.0)
    assert strategy._epsilon == pytest.approx(0.2)
    # No improvement: slower decay
    strategy.update(0.0)
    assert strategy._epsilon == pytest.approx(0.1)
    strategy.update(5.0)
    assert strategy._epsilon == pytest.approx(0.1)
    assert strategy.get_metrics() == {"exploration_epsilon": pytest.approx(0.1)}
    assert DistSamplingStrategy().get_metrics() == {}
