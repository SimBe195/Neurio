from dataclasses import dataclass


@dataclass
class ExplorationConfig:
    """
    Curiosity driven exploration with a count-based novelty bonus: every step, the agent gets `bonus_scale /
    sqrt(visits)` on top of the game's reward, where visits counts how often any worker has reached the same cell of
    `cell_size` pixels of the same level. Rarely seen places are thereby worth exploring. 0 disables the bonus.
    """

    bonus_scale: float = 0.5
    cell_size: int = 16

    def __post_init__(self) -> None:
        assert self.bonus_scale >= 0
        assert self.cell_size > 0
