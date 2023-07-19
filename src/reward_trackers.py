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


class MovingAvgReward(RewardTracker):
    def get_value(self) -> float:
        if not self.datapoints:
            return 0
        return statistics.mean(self.datapoints)


class BestMovingAvgReward(MovingAvgReward):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.best_avg = 0

    def reset(self) -> None:
        super().reset()
        self.best_avg = 0

    def record_reward(self, reward: float) -> None:
        super().record_reward(reward)
        self.best_avg = max(self.best_avg, super().get_value())

    def get_value(self) -> float:
        return self.best_avg


class MovingMaxReward(RewardTracker):
    def get_value(self) -> float:
        if not self.datapoints:
            return 0
        return max(self.datapoints)


class MovingMinReward(RewardTracker):
    def get_value(self) -> float:
        if not self.datapoints:
            return 0
        return min(self.datapoints)
