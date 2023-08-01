import math
import statistics
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional


class RewardTracker(ABC):
    def record_reward(self, reward: float) -> None:
        pass

    @abstractmethod
    def get_value(self) -> float:
        pass

    def reset(self) -> None:
        pass


class HistoryRewardTracker(RewardTracker):
    def __init__(self, history_size: Optional[int] = None) -> None:
        if history_size is not None:
            self.datapoints = deque(maxlen=history_size)
        else:
            self.datapoints = []

    def record_reward(self, reward: float) -> None:
        self.datapoints.append(reward)

    def reset(self) -> None:
        self.datapoints.clear()


class MovingAvgReward(HistoryRewardTracker):
    def get_value(self) -> float:
        if not self.datapoints:
            return 0
        return statistics.mean(self.datapoints)


class MovingMaxReward(HistoryRewardTracker):
    def get_value(self) -> float:
        if not self.datapoints:
            return 0
        return max(self.datapoints)


class MovingMinReward(HistoryRewardTracker):
    def get_value(self) -> float:
        if not self.datapoints:
            return 0
        return min(self.datapoints)


class SumReward(RewardTracker):
    def __init__(self, base_trackers: List[RewardTracker]) -> None:
        self.base_trackers = base_trackers

    def get_value(self) -> float:
        return sum([tracker.get_value() for tracker in self.base_trackers])


class BestReward(RewardTracker):
    def __init__(self, base_tracker: RewardTracker) -> None:
        self.base_tracker = base_tracker
        self.best = 0

    def reset(self) -> None:
        self.base_tracker.reset()
        self.best = 0

    def get_value(self) -> float:
        self.best = max(self.best, self.base_tracker.get_value())
        return self.best
