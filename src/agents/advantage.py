from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class GaeEstimator:
    gamma: float
    tau: float
    normalize: bool = False

    def get_advantage_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape (T, W) or (T)
        assert rewards.dim() == values.dim() == dones.dim() <= 2

        # Value has time T+1 to contain the next_value after the last state
        assert rewards.shape[0] == values.shape[0] - 1 == dones.shape[0]

        # Worker dimension is the same for every one
        if rewards.dim() == 2:
            assert rewards.shape[1] == values.shape[1] == dones.shape[1]

        # Shape (T, W) or (T)
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]

        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.tau * gae

            advantages[t] = gae

        if self.normalize:
            advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

        returns = advantages + values[:-1]

        return advantages, returns
