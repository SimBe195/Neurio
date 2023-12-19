from abc import ABC, abstractmethod
import random

import mlflow
import torch
from dataclasses import dataclass


@dataclass
class SamplingStrategy(ABC):
    pass

    @abstractmethod
    def sample_action(self, action_probs: torch.Tensor) -> torch.Tensor:
        pass

    def update(self, performance: float) -> None:
        pass


@dataclass
class DistSamplingStrategy(SamplingStrategy):
    name: str = "dist_sampling"

    def sample_action(self, action_probs: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Categorical(probs=action_probs).sample()


@dataclass
class GreedySamplingStrategy(SamplingStrategy):
    name: str = "greedy"

    def sample_action(self, action_probs: torch.Tensor) -> torch.Tensor:
        return torch.argmax(action_probs, dim=-1)


@dataclass
class AdaptiveEpsilonGreedySamplingStrategy(SamplingStrategy):
    initial_epsilon: float
    min_epsilon: float
    decay_factor: float
    name: str = "adaptive epsilon-greedy"

    def __post_init__(self):
        assert 0 <= self.initial_epsilon <= 1
        assert 0 <= self.min_epsilon <= self.initial_epsilon
        assert 0 <= self.decay_factor <= 1

        self._epsilon = self.initial_epsilon
        self._prev_performance = float("-inf")
        self._update_step = 0

    def sample_action(self, action_probs: torch.Tensor) -> torch.Tensor:
        if random.random() < self._epsilon:
            return torch.randint(
                low=0, high=action_probs.size(-1), size=action_probs.shape[:-1], device=action_probs.device
            )
        else:
            return torch.argmax(action_probs, dim=-1)

    def update(self, performance: float) -> None:
        if performance > self._prev_performance:
            self._epsilon *= self.decay_factor
        else:
            self._epsilon *= self.decay_factor**0.5

        self._epsilon = max(self._epsilon, self.min_epsilon)
        self._prev_performance = performance

        mlflow.log_metric("exploration_epsilon", self._epsilon, self._update_step)
        self._update_step += 1
