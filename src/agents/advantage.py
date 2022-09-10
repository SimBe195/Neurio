from typing import Tuple

import numpy as np


def gae_advantage_estimate(
    rewards: np.array,
    values: np.array,
    dones: np.array,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.array, np.array]:
    assert len(rewards) == len(values) - 1 == len(dones)

    returns = []
    advantages = []
    gae = 0
    for reward, value, next_value, done in reversed(
        list(
            zip(
                rewards,
                values[:-1],
                values[1:],
                dones,
            )
        )
    ):
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        returns.insert(0, gae + value)
        advantages.insert(0, gae)

    returns = np.stack(returns, axis=0).astype(np.float32)
    advantages = np.stack(advantages, axis=0).astype(np.float32)
    advantages = (advantages - np.mean(advantages, axis=0)[None, ...]) / (
        np.std(advantages, axis=0)[None, ...] + 1e-10
    )

    return advantages, returns
