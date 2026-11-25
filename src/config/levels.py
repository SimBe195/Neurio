import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

ALL_LEVELS = [f"{world}-{stage}" for world in range(1, 9) for stage in range(1, 5)]

# Levels and their sampling probabilities
LevelDistribution = tuple[list[str], list[float]]


def _check_levels(levels: list[str]) -> None:
    assert len(levels) > 0, "At least one level is required."
    for level in levels:
        assert re.fullmatch(r"[1-8]-[1-4]", level), f"Invalid level {level}, expected <world>-<stage>, e.g. 1-1."
    assert len(set(levels)) == len(levels), "Levels must be unique."


@dataclass
class LevelConfig(ABC):
    """
    Determines which levels the agent plays. A level is sampled at the start of each episode, see
    environment.LevelSelectEnv.
    """

    # Number of most recent episodes per level whose results count: they are reported, decide on unlocking (see
    # CurriculumLevelsConfig) and on whether a level is stuck (see exploration.UntriedActions)
    success_window: int = field(default=50, kw_only=True)

    def __post_init__(self) -> None:
        assert self.success_window >= 1

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def initial_distribution(self) -> LevelDistribution:
        """Levels and their sampling probabilities at the start of training."""
        ...

    def evaluation_levels(self) -> list[str]:
        """Levels a trained agent is evaluated on, in the order test.py plays them."""
        return self.initial_distribution()[0]


@dataclass
class SingleLevelConfig(LevelConfig):
    level: str = "1-1"

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_levels([self.level])

    @property
    def name(self) -> str:
        return self.level

    def initial_distribution(self) -> LevelDistribution:
        return [self.level], [1.0]


@dataclass
class CurriculumLevelsConfig(LevelConfig):
    """
    Starts with the first level and adds the next one to the pool once the newest level reached its flag in any of its
    last `success_window` episodes, and the flag was reached in at least `unlock_mean_success_rate` of the recent
    episodes on average over all unlocked levels, each of which needs at least `min_level_episodes` of them. The
    newest level thereby learns from its best episodes while the next ones are already played, and levels the agent
    forgot block progress.

    Levels are sampled by need: the weight of a level is its failure rate, at least `min_level_weight`, and 1 for
    levels without results yet. So the agent mostly practices what it cannot do yet, including forgotten levels.
    """

    levels: list[str] = field(default_factory=lambda: list(ALL_LEVELS))
    unlock_mean_success_rate: float = 0.5
    min_level_episodes: int = 10
    min_level_weight: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_levels(self.levels)
        assert 0 < self.unlock_mean_success_rate <= 1
        assert 1 <= self.min_level_episodes <= self.success_window
        assert 0 < self.min_level_weight <= 1

    @property
    def name(self) -> str:
        return f"curriculum-{len(self.levels)}"

    def initial_distribution(self) -> LevelDistribution:
        return [self.levels[0]], [1.0]

    def evaluation_levels(self) -> list[str]:
        return list(self.levels)
