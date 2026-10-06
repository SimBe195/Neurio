from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class SamplingStrategy(ABC):
    pass

    @abstractmethod
    def behavior_probs(self, action_probs: torch.Tensor) -> torch.Tensor:
        """
        Probabilities with which sample_action() picks each action, given the policy's action probabilities.
        These are needed for the importance ratio in off-policy corrections such as PPO.
        """
        pass

    def sample_action(self, action_probs: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Categorical(probs=self.behavior_probs(action_probs)).sample()

    def update(self, performance: float) -> None:
        pass

    def get_metrics(self) -> dict[str, float]:
        return {}

    def state_dict(self) -> dict:
        """What update() changed, to resume training."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


def _greedy_probs(action_probs: torch.Tensor) -> torch.Tensor:
    greedy_actions = torch.argmax(action_probs, dim=-1)
    return torch.nn.functional.one_hot(greedy_actions, action_probs.size(-1)).to(action_probs.dtype)


@dataclass
class DistSamplingStrategy(SamplingStrategy):
    name: str = "dist_sampling"

    def behavior_probs(self, action_probs: torch.Tensor) -> torch.Tensor:
        return action_probs


@dataclass
class GreedySamplingStrategy(SamplingStrategy):
    name: str = "greedy"

    def behavior_probs(self, action_probs: torch.Tensor) -> torch.Tensor:
        return _greedy_probs(action_probs)

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

    def behavior_probs(self, action_probs: torch.Tensor) -> torch.Tensor:
        # A random action can coincide with the greedy one
        return self._epsilon / action_probs.size(-1) + (1 - self._epsilon) * _greedy_probs(action_probs)

    def sample_action(self, action_probs: torch.Tensor) -> torch.Tensor:
        # Each worker decides independently whether to explore
        explore = torch.rand(action_probs.shape[:-1], device=action_probs.device) < self._epsilon
        random_actions = torch.randint(
            low=0, high=action_probs.size(-1), size=action_probs.shape[:-1], device=action_probs.device
        )
        return torch.where(explore, random_actions, torch.argmax(action_probs, dim=-1))

    def update(self, performance: float) -> None:
        if performance > self._prev_performance:
            self._epsilon *= self.decay_factor
        else:
            self._epsilon *= self.decay_factor**0.5

        self._epsilon = max(self._epsilon, self.min_epsilon)
        self._prev_performance = performance

    def get_metrics(self) -> dict[str, float]:
        return {"exploration_epsilon": self._epsilon}

    def state_dict(self) -> dict:
        return {"epsilon": self._epsilon, "prev_performance": self._prev_performance}

    def load_state_dict(self, state: dict) -> None:
        self._epsilon, self._prev_performance = state["epsilon"], state["prev_performance"]
