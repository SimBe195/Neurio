import numpy as np
from typing import Any, Dict, Tuple
from gym import RewardWrapper


class BasicRewardWrapper(RewardWrapper):
    def reward(self, reward: float) -> float:
        return reward


class RewardWrapperV1(BasicRewardWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.curr_score = 0
        self.curr_x_pos = 40

    def reset(self) -> np.array:
        self.curr_score = 0
        self.curr_x_pos = 40
        return super().reset()

    def step(self, action: int) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        state, reward, terminated, truncated, info = super().step(action)

        time_penalty = -0.02

        # Max speed is 3 per frame
        new_x_pos = info["x_pos"]
        x_diff = new_x_pos - self.curr_x_pos
        self.curr_x_pos = new_x_pos

        next_score = info["score"]
        score_diff = next_score - self.curr_score
        self.curr_score = next_score

        reward = time_penalty + x_diff + score_diff / 40

        if terminated:
            if info["flag_get"]:
                reward += 15.0
            else:
                reward -= 50.0

        reward = np.clip(reward, -15.0, 15.0)

        return state, reward, terminated, truncated, info
