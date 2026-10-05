import math
from collections import defaultdict

import numpy as np

from config.exploration import ExplorationConfig


class NoveltyBonus:
    """
    Count-based novelty bonus over position cells, shared by all workers. Counts are kept for the whole training, so
    the bonus for well known places fades while new ground stays attractive.
    """

    def __init__(self, config: ExplorationConfig) -> None:
        self.scale = config.bonus_scale
        self.cell_size = config.cell_size
        self.counts: defaultdict[tuple[str, int, int], int] = defaultdict(int)

    def __call__(self, positions: list[tuple[str, int, int]]) -> np.ndarray:
        """
        Count the cells reached by the last step of each worker and return their bonus.

        :param positions: Level, x and y position of each worker
        """
        bonus = np.zeros(len(positions), dtype=np.float32)
        for worker, (level, x, y) in enumerate(positions):
            cell = (level, x // self.cell_size, y // self.cell_size)
            self.counts[cell] += 1
            bonus[worker] = self.scale / math.sqrt(self.counts[cell])
        return bonus

    @property
    def visited_cells(self) -> int:
        return len(self.counts)
