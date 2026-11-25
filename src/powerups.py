from collections import Counter
from itertools import pairwise

from config.levels import ALL_LEVELS
from environment import Powerup


class StartPowerups:
    """
    With which power-up Mario starts each level, as if carried over from the previous one in game order: as often as
    episodes of the previous level reached its flag with each power-up, counted over a period of iterations. The
    distribution of the counts of one period applies during the next, so that each level practices the starts that
    the previous one actually leads to. 1-1 starts small, like the game, and so does a level before the previous one
    has reached its flag in a period.
    """

    def __init__(self, period: int = 1000) -> None:
        """:param period: Iterations over which the finishes are counted"""
        self.period = period
        self.finishes: dict[str, Counter[Powerup]] = {}
        self.distributions: dict[str, list[float]] = {}
        self._iterations = 0

    def record_episode(self, level: str, flag_get: bool, powerup: Powerup) -> None:
        if flag_get:
            self.finishes.setdefault(level, Counter())[powerup] += 1

    def update(self) -> bool:
        """Called after each training iteration. Returns whether the distributions changed."""
        self._iterations += 1
        if self._iterations % self.period:
            return False
        distributions = {}
        for previous, level in pairwise(ALL_LEVELS):
            counts = self.finishes.get(previous, Counter())
            if total := counts.total():
                distributions[level] = [counts[powerup] / total for powerup in Powerup]
        self.finishes = {}
        changed = distributions != self.distributions
        self.distributions = distributions
        return changed

    def get_metrics(self) -> dict[str, float]:
        """Per level, the share of the starts with a power-up in the current period."""
        return {f"powerups/start_{level}": 1 - probs[Powerup.SMALL] for level, probs in self.distributions.items()}

    def state_dict(self) -> dict:
        return {
            "finishes": {level: {int(p): n for p, n in counts.items()} for level, counts in self.finishes.items()},
            "distributions": self.distributions,
            "iterations": self._iterations,
        }

    def load_state_dict(self, state: dict) -> None:
        self.finishes = {
            level: Counter({Powerup(p): n for p, n in counts.items()}) for level, counts in state["finishes"].items()
        }
        self.distributions = state["distributions"]
        self._iterations = state["iterations"]
