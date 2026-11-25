from dataclasses import dataclass


@dataclass
class ExplorationConfig:
    """
    Curiosity driven exploration with a count-based novelty bonus: every step, the agent learns from `bonus_scale /
    sqrt(visits)` on top of its reward, where visits counts how often any worker has been in the same cell of
    `cell_size` pixels of the same area of the same level. Rarely seen places are thereby worth exploring, e.g. the
    other paths of a castle maze.

    On a level that is stuck at a wall, the agent also takes the actions that have never been taken in its cell,
    starting at the wall and going back from it, see exploration.UntriedActions. The cells are the same.
    """

    bonus_scale: float
    cell_size: int

    def __post_init__(self) -> None:
        assert self.bonus_scale > 0
        assert self.cell_size > 0
