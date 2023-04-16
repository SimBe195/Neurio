import statistics
from abc import ABC, abstractmethod
from collections import deque


class RewardTracker(ABC):
    @abstractmethod
    def record_reward(self, reward: float) -> None:
        pass
    
    @abstractmethod
    def get_value(self) -> float:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class MovingAverageReward(RewardTracker):
    def __init__(self, num_datapoints: int = 100) -> None:
        self.datapoints = deque(maxlen=num_datapoints)

    def record_reward(self, reward: float) -> None:
        self.datapoints.append(reward)

    def get_value(self) -> float:
        if not self.datapoints:
            return 0
        return statistics.mean(self.datapoints)

    def reset(self) -> None:
        self.datapoints.clear()


class MaxReward(RewardTracker):
    def __init__(self) -> None:
        self.max = 0
    
    def record_reward(self, reward: float) -> None:
        self.max = max(self.max, reward)
        
    def get_value(self) -> float:
        return self.max

    def reset(self) -> None:
        self.max = 0

class MinReward(RewardTracker):
    def __init__(self, history_size: int = 100) -> None:
        self.datapoints = deque(maxlen=history_size)
    
    def record_reward(self, reward: float) -> None:
        self.datapoints.append(reward)
        
    def get_value(self) -> float:
        if not self.datapoints:
            return 0
        return min(self.datapoints)

    def reset(self) -> None:
        self.datapoints.clear()