import numpy as np
import pytest

from config.exploration import ExplorationConfig
from novelty import NoveltyBonus


def test_bonus_decays_with_visits():
    novelty = NoveltyBonus(ExplorationConfig(bonus_scale=0.5, cell_size=16))
    bonuses = [novelty([("1-1", 40, 79)])[0] for _ in range(4)]
    assert bonuses == pytest.approx([0.5, 0.5 / np.sqrt(2), 0.5 / np.sqrt(3), 0.25])


def test_cells_by_level_and_position():
    novelty = NoveltyBonus(ExplorationConfig(bonus_scale=1.0, cell_size=16))
    novelty([("1-1", 40, 79)])
    # Same cell, a different cell, and the same position on another level
    same, other_x, other_level = (
        novelty([position])[0] for position in [("1-1", 47, 64), ("1-1", 48, 79), ("1-2", 40, 79)]
    )
    assert same == pytest.approx(1 / np.sqrt(2))
    assert other_x == other_level == 1.0
    assert novelty.visited_cells == 3


def test_counts_are_shared_by_workers():
    novelty = NoveltyBonus(ExplorationConfig(bonus_scale=1.0))
    assert novelty([("1-1", 40, 79), ("1-1", 40, 79)]).tolist() == pytest.approx([1.0, 1 / np.sqrt(2)])
