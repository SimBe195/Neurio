from typing import Any, Dict, Tuple

import numpy as np
from gym import RewardWrapper
from omegaconf import DictConfig


class CustomRewardWrapper(RewardWrapper):
    def __init__(self, config: DictConfig, *args, **kwargs) -> None:
        super().__init__(new_step_api=True, *args, **kwargs)
        self.time_penalty = (
            config.tick_penalty / 24
        )  # tick every 0.4 second = 24 frames
        self.speed_reward_weight = config.speed_reward_weight
        self.score_reward_weight = config.score_reward_weight
        self.death_penalty = config.death_penalty
        self.level_finish_reward = config.level_finish_reward

        self.curr_score = 0
        self.curr_x_pos = 40

    def reset(self) -> np.array:
        self.curr_score = 0
        self.curr_x_pos = 40
        return super().reset()

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        state, reward, terminated, truncated, info = super().step(action)

        # Max speed is 3 per frame
        new_x_pos = info["x_pos"]
        x_diff = new_x_pos - self.curr_x_pos
        if abs(x_diff) > 20:
            # Not normal movement, this is a teleport of some sort
            x_diff = 0
        self.curr_x_pos = new_x_pos

        next_score = info["score"]
        score_diff = next_score - self.curr_score
        self.curr_score = next_score

        reward = (
            -self.time_penalty
            + self.speed_reward_weight * x_diff
            + self.score_reward_weight * score_diff
        )
        reward = np.clip(reward, -15, 15)

        if terminated or truncated:
            if info["flag_get"]:
                reward += self.level_finish_reward
            else:
                reward -= self.death_penalty

        reward = reward / 10

        return state, reward, terminated, truncated, info

    def reward(self, reward: float) -> float:
        return reward
