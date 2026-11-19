from dataclasses import dataclass


@dataclass
class ExplorationConfig:
    """
    Curiosity driven exploration with a count-based novelty bonus: every step, the agent learns from `bonus_scale /
    sqrt(visits)` on top of its reward, where visits counts how often any worker has been in the same cell of
    `cell_size` pixels of the same area of the same level. Rarely seen places are thereby worth exploring, e.g. the
    other paths of a castle maze. 0 disables the bonus.

    With `try_all_actions`, a worker on a level that is stuck at a wall takes the actions that have never been taken in
    its cell, on the ground or in the air, starting at the wall and going back from it, see exploration.UntriedActions.
    The level counts as stuck after `success_window` episodes without progress, see config.levels.LevelConfig. Only
    for agents that learn off-policy, e.g. not PPO.
    """

    bonus_scale: float = 1.0
    cell_size: int = 16
    try_all_actions: bool = False

    def __post_init__(self) -> None:
        assert self.bonus_scale >= 0
        assert self.cell_size > 0
