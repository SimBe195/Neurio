from typing import Tuple

import torch
from dataclasses import dataclass
from jaxtyping import Float


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
        rewards: Float[torch.Tensor, "buffer worker"],
        values: Float[torch.Tensor, "buffer+1 worker"],
        dones: Float[torch.Tensor, "buffer worker"],
    ) -> Tuple[Float[torch.Tensor, "buffer worker"], Float[torch.Tensor, "buffer worker"]]:
        """
        Calculate the Generalized Advantage Estimation (GAE) and returns.

        Args:
            rewards: Tensor of rewards of shape (T, W).
            values: Tensor of values of shape (T+1, W).
            dones: Tensor indicating if an episode is done of shape (T, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the advantages and returns of shape (T, W).
        """

        # Shape (T, W)
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]

        advantages = torch.zeros_like(rewards)
        gae = 0
        for t, (done, delta) in reversed(list(enumerate(zip(dones, deltas)))):
            gae = delta + self.gamma * self.tau * (1 - done) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        return advantages, returns
