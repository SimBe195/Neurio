from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from config.exploration import ExplorationConfig

# Where Mario is: world, stage, area, x and y position
Position = tuple[int, ...]


class NoveltyBonus:
    """
    Count-based novelty bonus over cells of Mario's position, shared by all workers, see ExplorationConfig. The counts
    are kept for the whole training, so that the bonus fades where the agent has often been while new ground stays
    attractive.
    """

    def __init__(self, config: ExplorationConfig) -> None:
        self.scale = config.bonus_scale
        self.cell_size = config.cell_size
        self.counts: defaultdict[tuple[int, ...], int] = defaultdict(int)

    def visit(self, positions: list[Position]) -> np.ndarray:
        """Count the cells reached by the last step of each worker and return how often each has been visited."""
        visits = np.zeros(len(positions), dtype=np.int64)
        for worker, (*area, x, y) in enumerate(positions):
            cell = (*area, x // self.cell_size, y // self.cell_size)
            self.counts[cell] += 1
            visits[worker] = self.counts[cell]
        return visits

    def visits(self, positions: list[Position]) -> np.ndarray:
        """How often the cells of the positions have been visited, at least once, without counting a visit."""
        cells = [(*area, x // self.cell_size, y // self.cell_size) for *area, x, y in positions]
        return np.array([max(self.counts.get(cell, 0), 1) for cell in cells], dtype=np.int64)

    def bonus(self, visits: np.ndarray) -> np.ndarray:
        return (self.scale / np.sqrt(visits)).astype(np.float32)

    @property
    def visited_cells(self) -> int:
        return len(self.counts)

    def state_dict(self) -> dict:
        return {"counts": dict(self.counts)}

    def load_state_dict(self, state: dict) -> None:
        self.counts = defaultdict(int, state["counts"])


@dataclass
class Wall:
    """How far the episodes of a level got, in cells of distance, see environment.ProgressShaping."""

    # Furthest cell any episode reached
    best: int = -1
    episodes_since_progress: int = 0
    # Cells up to the wall in which untried actions are taken, 0 while the level is not stuck
    window: int = 0


class UntriedActions:
    """
    Which actions any worker has taken in each cell of Mario's position, of `cell_size` pixels of the same area of the
    same level, apart for Mario on the ground and in the air, and where the levels are stuck.

    The actor generalizes across places, so an action that is useless almost everywhere, e.g. pressing down, becomes
    unlikely also where it has never been tried, e.g. on 8-4's floating pipe. A level is stuck at a wall once
    `patience` episodes in a row neither reached the flag nor got a cell further than any before. Then a worker that is
    in the window of cells of distance up to the wall takes an action that has never been taken in its cell, like UCB1
    plays every arm once. The window starts at the wall and moves back a cell after every iteration in which nothing
    untried was found in it, so that the search goes from the wall backwards to where the way on is, e.g. the pipe
    before 8-4's loop. That includes iterations in which no episode got into the window: the wall is the furthest any
    episode got, which most episodes do not reach. Progress past the wall closes it. Elsewhere, the policy acts as
    usual: at the frontier, where taking every action in every new cell keeps Mario from getting further, and on the
    routes it has learned.
    """

    def __init__(self, cell_size: int, num_actions: int, patience: int) -> None:
        self.cell_size = cell_size
        self.num_actions = num_actions
        self.patience = patience
        # Per cell, a bit for each action taken there
        self.tried: defaultdict[tuple[int, ...], int] = defaultdict(int)
        self.walls: defaultdict[str, Wall] = defaultdict(Wall)
        # Stuck levels in whose windows untried actions were found in this iteration
        self._found_untried: set[str] = set()

    def _cell(self, position: Position, on_ground: bool) -> tuple[int, ...]:
        *area, x, y = position
        return (*area, x // self.cell_size, y // self.cell_size, int(on_ground))

    def _in_window(self, level: str, distance: int) -> bool:
        wall = self.walls.get(level)
        return wall is not None and wall.best - wall.window < distance // self.cell_size <= wall.best

    def untried(
        self, levels: list[str], distances: list[int], positions: list[Position], on_ground: list[bool]
    ) -> np.ndarray:
        """Per worker, which actions have never been taken in its cell, if it is in the window before a wall."""
        untried = np.zeros((len(positions), self.num_actions), dtype=bool)
        for worker, (level, distance, position, ground) in enumerate(
            zip(levels, distances, positions, on_ground, strict=True)
        ):
            if self._in_window(level, distance):
                untried[worker] = (self.tried[self._cell(position, ground)] >> np.arange(self.num_actions)) & 1 == 0
                if untried[worker].any():
                    self._found_untried.add(level)
        return untried

    def record(self, positions: list[Position], on_ground: list[bool], actions: list[int]) -> None:
        for position, ground, action in zip(positions, on_ground, actions, strict=True):
            self.tried[self._cell(position, ground)] |= 1 << action

    def record_episode(self, level: str, flag_get: bool, max_distance: int) -> None:
        """Update the level's wall with the furthest distance the episode got."""
        wall = self.walls[level]
        cell = max_distance // self.cell_size
        if flag_get or cell > wall.best:
            wall.best, wall.episodes_since_progress, wall.window = max(wall.best, cell), 0, 0
            return
        wall.episodes_since_progress += 1
        if wall.episodes_since_progress >= self.patience:
            wall.window = max(wall.window, 1)

    def end_iteration(self) -> None:
        """Move the windows in which nothing untried was found back a cell, at most to the start of the level."""
        for level, wall in self.walls.items():
            if wall.window > 0 and level not in self._found_untried:
                wall.window = min(wall.window + 1, wall.best + 1)
        self._found_untried = set()

    def get_metrics(self) -> dict[str, float]:
        metrics = {f"exploration/window_{level}": wall.window * self.cell_size for level, wall in self.walls.items()}
        return {"exploration/stuck_levels": sum(wall.window > 0 for wall in self.walls.values()), **metrics}

    def state_dict(self) -> dict:
        return {"tried": dict(self.tried), "walls": {level: asdict(wall) for level, wall in self.walls.items()}}

    def load_state_dict(self, state: dict) -> None:
        self.tried = defaultdict(int, state["tried"])
        self.walls = defaultdict(Wall, {level: Wall(**wall) for level, wall in state["walls"].items()})
