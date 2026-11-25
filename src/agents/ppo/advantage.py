from dataclasses import dataclass

import torch
from jaxtyping import Float, Real


@dataclass
class GaeEstimator:
    """
    Generalized advantage estimation (Schulman et al., 2016): GAE_t = sum_k (gamma * lam)^k * delta_{t+k} with the TD
    errors delta_t = r_t + gamma * V(s_{t+1}) - V(s_t), not continued past the end of an episode. The returns, the
    critic's targets, are V(s_t) + GAE_t.
    """

    gamma: float
    lam: float

    def get_advantage_returns(
        self,
        rewards: Float[torch.Tensor, " buffer *worker"],
        values: Float[torch.Tensor, " buffer+1 *worker"],
        dones: Real[torch.Tensor, " buffer *worker"],
    ) -> tuple[Float[torch.Tensor, " buffer *worker"], Float[torch.Tensor, " buffer *worker"]]:
        """
        :param rewards: Of each step, optionally per worker
        :param values: Of each step and of the state after the last one
        :param dones: Whether the episode ended with each step
        :return: The advantages and returns of each step
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
        for t, (done, delta) in reversed(list(enumerate(zip(dones, deltas, strict=True)))):
            gae = delta + self.gamma * self.lam * (1 - done) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        return advantages, returns
