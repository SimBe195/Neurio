from typing import Tuple

import torch
from dataclasses import dataclass


@dataclass
class GaeEstimator:
    """
    Implements the generalized advantage estimation mechanism. The GAE is defined by the formula
    GAE_t = sum_{k=0}^{T-1} (gamma * tau)^k * delta_{t + k}
    where
    - gamma is the discount factor
    - tau is the GAE smoothing parameter
    - T is the time horizon
    - delta_t is the temporal difference error at time t, defined as delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    - r_t is the reward given for the action taken at time t
    - V(s_t) is the value of the state at time t

    - Return_t = V(s_t) + GAE_t
    """

    gamma: float
    tau: float

    def get_advantage_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the Generalized Advantage Estimation (GAE) and returns.

        Args:
            rewards: Tensor of rewards of shape (T) or (T, W).
            values: Tensor of values of shape (T+1) or (T+1, W).
            dones: Tensor indicating if an episode is done of shape (T) or (T, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the advantages and returns of shape (T) or (T, W).
        """
        # Shape (T, W) or (T)
        if not (rewards.dim() == values.dim() == dones.dim() <= 2):
            raise ValueError("All inputs must have the same number of dimensions (<= 2).")

        # Value has time T+1 to contain the next_value after the last state
        if not (rewards.shape[0] == values.shape[0] - 1 == dones.shape[0]):
            raise ValueError("Mismatch in the shape of rewards, values, and dones.")

        # Worker dimension is the same for every one
        if rewards.dim() == 2 and not (rewards.shape[1] == values.shape[1] == dones.shape[1]):
            raise ValueError("Mismatch in the worker dimension.")

        # Shape (T, W) or (T)
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]

        advantages = torch.zeros_like(rewards)
        gae = 0
        for t, (done, delta) in reversed(list(enumerate(zip(dones, deltas)))):
            gae = delta + self.gamma * self.tau * (1 - done) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        return advantages, returns
