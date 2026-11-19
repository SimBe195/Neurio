import math

import torch
from jaxtyping import Bool, Float


class RewardScale:
    """
    Scales rewards by the running standard deviation of the discounted return, as Stable-Baselines3's and CleanRL's
    reward normalization do. The values the critic learns then stay of the same order whatever the reward, gamma or
    level mix, so that its loss does not outweigh the policy's in the shared network.
    """

    def __init__(self, num_workers: int, gamma: float, device: torch.device) -> None:
        self.gamma = gamma
        self.returns = torch.zeros(num_workers, device=device)
        # Small count and unit variance at first, so that the first rewards are not scaled by a variance of 0
        self.count = 1e-4
        self.mean = 0.0
        self.var = 1.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var + 1e-8)

    def __call__(
        self, rewards: Float[torch.Tensor, " worker"], dones: Bool[torch.Tensor, " worker"]
    ) -> Float[torch.Tensor, " worker"]:
        """Scales the rewards of one step, after updating the statistics with the returns they lead to."""
        self.returns = self.returns * self.gamma + rewards
        self._update(self.returns)
        self.returns[dones] = 0
        return rewards / self.std

    def _update(self, returns: Float[torch.Tensor, " worker"]) -> None:
        # Merges the mean and variance of the batch into the running ones (Chan et al.)
        batch_mean, batch_var, batch_count = returns.mean().item(), returns.var(correction=0).item(), len(returns)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.var = (
            self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        ) / total
        self.count = total

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean, "var": self.var}

    def load_state_dict(self, state: dict) -> None:
        self.count, self.mean, self.var = state["count"], state["mean"], state["var"]
