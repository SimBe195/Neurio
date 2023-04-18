import statistics
from abc import ABC, abstractmethod
from collections import deque


class RewardTracker(ABC):
    def __init__(self, history_size: int = 100) -> None:
        self.datapoints = deque(maxlen=history_size)
    
    def record_reward(self, reward: float) -> None:
        self.datapoints.append(reward)
    
    @abstractmethod
    def get_value(self) -> float:
        pass

    def reset(self) -> None:
        self.datapoints.clear()


class MovingAverageReward(RewardTracker):
    def get_value(self) -> float:
        if not self.datapoints:
            return 0
        return statistics.mean(self.datapoints)

class MaxReward(RewardTracker):
    def get_value(self) -> float:
        if not self.datapoints:
            return 0
        return max(self.datapoints)

class MinReward(RewardTracker):
    def get_value(self) -> float:
        if not self.datapoints:
            return 0
        return min(self.datapoints)