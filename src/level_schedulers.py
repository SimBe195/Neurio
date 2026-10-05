import logging
from collections import deque

from config.levels import CurriculumLevelsConfig, LevelConfig, LevelDistribution

log = logging.getLogger(__name__)


class LevelScheduler:
    """Decides from which levels episodes are sampled. This base version keeps the level config's distribution."""

    def __init__(self, config: LevelConfig) -> None:
        self.config = config

    def distribution(self) -> LevelDistribution | None:
        return self.config.initial_distribution()

    def record_episode(self, level: str, success: bool) -> None:
        pass

    def update(self) -> bool:
        """Called after each training iteration. Returns whether the distribution changed."""
        return False

    def get_metrics(self) -> dict[str, float]:
        return {}

    def score(self) -> float | None:
        """Competence over the scheduled levels, higher is better, None if not tracked."""
        return None


class CurriculumLevelScheduler(LevelScheduler):
    """Unlocks levels on mastery and samples them by need, see CurriculumLevelsConfig."""

    # Minimum change of a level weight for the new distribution to be sent to the workers
    WEIGHT_TOLERANCE = 0.02

    def __init__(self, config: CurriculumLevelsConfig) -> None:
        super().__init__(config)
        self.config: CurriculumLevelsConfig = config
        self.num_unlocked = config.initial_levels
        self.results: dict[str, deque[bool]] = {level: deque(maxlen=config.success_window) for level in config.levels}
        # The distribution the workers currently use
        self._sent_distribution = config.initial_distribution()

    @property
    def unlocked_levels(self) -> list[str]:
        return self.config.levels[: self.num_unlocked]

    @property
    def newest_level(self) -> str:
        return self.unlocked_levels[-1]

    def success_rate(self, level: str) -> float | None:
        results = self.results[level]
        return sum(results) / len(results) if results else None

    def distribution(self) -> LevelDistribution:
        weights = []
        for level in self.unlocked_levels:
            rate = self.success_rate(level)
            weights.append(1.0 if rate is None else max(1 - rate, self.config.min_level_weight))
        total = sum(weights)
        return self.unlocked_levels, [weight / total for weight in weights]

    def record_episode(self, level: str, success: bool) -> None:
        if level in self.results:
            self.results[level].append(success)

    def is_mastered(self) -> bool:
        """Whether the agent masters the newest level and, on average, all unlocked levels."""
        if len(self.results[self.newest_level]) < self.config.success_window:
            return False
        if any(len(self.results[level]) < self.config.min_level_episodes for level in self.unlocked_levels):
            return False
        rates = [self.success_rate(level) or 0.0 for level in self.unlocked_levels]
        threshold = self.config.unlock_success_rate
        return rates[-1] >= threshold and sum(rates) / len(rates) >= threshold

    def update(self) -> bool:
        if self.num_unlocked < len(self.config.levels) and self.is_mastered():
            self.num_unlocked += 1
            log.info(f"Unlocked level {self.newest_level}.")

        # The weights follow the success rates, so the distribution can change without an unlock
        levels, weights = self.distribution()
        sent_levels, sent_weights = self._sent_distribution
        changed = levels != sent_levels or any(
            abs(weight - sent) > self.WEIGHT_TOLERANCE for weight, sent in zip(weights, sent_weights, strict=True)
        )
        if changed:
            self._sent_distribution = (levels, weights)
        return changed

    def get_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {"num_unlocked_levels": self.num_unlocked, "levels_cleared_score": self.score()}
        for level in self.unlocked_levels:
            if (rate := self.success_rate(level)) is not None:
                metrics[f"success_rate_{level}"] = rate
        return metrics

    def score(self) -> float:
        """Sum of the recent success rates over all levels, i.e. the expected number of levels cleared."""
        return sum(self.success_rate(level) or 0.0 for level in self.config.levels)


def get_level_scheduler(config: LevelConfig) -> LevelScheduler:
    if isinstance(config, CurriculumLevelsConfig):
        return CurriculumLevelScheduler(config)
    return LevelScheduler(config)
