import statistics
from collections import deque


class MovingAverage:
    def __init__(self, num_datapoints: int = 100) -> None:
        self.datapoints = deque(maxlen=num_datapoints)

    def record_data(self, data: float) -> None:
        self.datapoints.append(data)

    def get_value(self) -> float:
        if not self.datapoints:
            return 0
        return statistics.mean(self.datapoints)

    def clear(self) -> None:
        self.datapoints.clear()
