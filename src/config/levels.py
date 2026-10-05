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


def level_index(level: str) -> int:
    """Position of a level in the order of the game, starting at 0 for 1-1."""
    world, stage = (int(part) for part in level.split("-"))
    return (world - 1) * 4 + (stage - 1)


@dataclass
class LevelConfig(ABC):
    """
    Determines which levels the agent plays. Single stage modes sample a level at the start of each episode and end
    the episode when Mario dies or reaches the flag.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def initial_distribution(self) -> LevelDistribution | None:
        """Levels and their sampling probabilities at the start of training, None if levels are not sampled."""
        ...

    def evaluation_distribution(self) -> LevelDistribution | None:
        """Levels and their sampling probabilities for evaluating a trained agent."""
        return self.initial_distribution()


def _uniform(levels: list[str]) -> LevelDistribution:
    return list(levels), [1 / len(levels)] * len(levels)


@dataclass
class SingleLevelConfig(LevelConfig):
    level: str = "1-1"

    def __post_init__(self) -> None:
        _check_levels([self.level])

    @property
    def name(self) -> str:
        return self.level

    def initial_distribution(self) -> LevelDistribution:
        return [self.level], [1.0]


@dataclass
class RandomLevelsConfig(LevelConfig):
    """Each episode is played on a level drawn uniformly from the pool."""

    levels: list[str] = field(default_factory=lambda: list(ALL_LEVELS))

    def __post_init__(self) -> None:
        _check_levels(self.levels)

    @property
    def name(self) -> str:
        return f"random-{len(self.levels)}"

    def initial_distribution(self) -> LevelDistribution:
        return _uniform(self.levels)


@dataclass
class CurriculumLevelsConfig(LevelConfig):
    """
    Starts with the first `initial_levels` levels and adds the next level to the pool once the agent masters the
    pool: the flag was reached in at least `unlock_success_rate` of the last `success_window` episodes on the newest
    level, and on average over all unlocked levels, each of which needs at least `min_level_episodes` recent episodes.
    Levels the agent forgot therefore block progress.

    Levels are sampled by need: the weight of a level is its failure rate, at least `min_level_weight`, and 1 for
    levels without results yet. So the agent mostly practices what it cannot do yet, including forgotten levels.
    """

    levels: list[str] = field(default_factory=lambda: list(ALL_LEVELS))
    initial_levels: int = 1
    unlock_success_rate: float = 0.5
    success_window: int = 50
    min_level_episodes: int = 10
    min_level_weight: float = 0.1

    def __post_init__(self) -> None:
        _check_levels(self.levels)
        assert 1 <= self.initial_levels <= len(self.levels)
        assert 0 < self.unlock_success_rate <= 1
        assert self.success_window >= 1
        assert 1 <= self.min_level_episodes <= self.success_window
        assert 0 < self.min_level_weight <= 1

    @property
    def name(self) -> str:
        return f"curriculum-{len(self.levels)}"

    def initial_distribution(self) -> LevelDistribution:
        return _uniform(self.levels[: self.initial_levels])

    def evaluation_distribution(self) -> LevelDistribution:
        return _uniform(self.levels)


@dataclass
class ProgressionLevelsConfig(LevelConfig):
    """
    Plays the full game from 1-1 on: after reaching the flag, the agent continues with the next level. The episode
    ends when Mario loses his first life.
    """

    @property
    def name(self) -> str:
        return "progression"

    def initial_distribution(self) -> None:
        return None
