import numpy as np
import pytest

from config.exploration import ExplorationConfig
from exploration import NoveltyBonus, UntriedActions


def test_bonus_decays_with_visits():
    novelty = NoveltyBonus(ExplorationConfig(bonus_scale=0.5, cell_size=16))
    visits = np.concatenate([novelty.visit([(1, 1, 0, 40, 79)]) for _ in range(4)])
    assert visits.tolist() == [1, 2, 3, 4]
    assert novelty.bonus(visits) == pytest.approx([0.5, 0.5 / np.sqrt(2), 0.5 / np.sqrt(3), 0.25])


def test_cells_by_area_and_position():
    novelty = NoveltyBonus(ExplorationConfig(bonus_scale=1.0, cell_size=16))
    novelty.visit([(1, 1, 0, 40, 79)])
    # The same cell, another x, another y, and the same position in another area and on another level
    positions = [(1, 1, 0, 47, 64), (1, 1, 0, 48, 79), (1, 1, 0, 40, 80), (1, 1, 2, 40, 79), (1, 2, 0, 40, 79)]
    same, *others = (novelty.visit([position])[0] for position in positions)
    assert same == 2
    assert others == [1] * 4
    assert novelty.visited_cells == 5


def test_counts_are_shared_by_workers():
    novelty = NoveltyBonus(ExplorationConfig(bonus_scale=1.0, cell_size=16))
    assert novelty.visit([(1, 1, 0, 40, 79), (1, 1, 0, 40, 79)]).tolist() == [1, 2]


def test_counts_are_restored():
    novelty = NoveltyBonus(ExplorationConfig(bonus_scale=1.0, cell_size=16))
    novelty.visit([(1, 1, 0, 40, 79)] * 3)
    restored = NoveltyBonus(ExplorationConfig(bonus_scale=1.0, cell_size=16))
    restored.load_state_dict(novelty.state_dict())
    assert restored.visit([(1, 1, 0, 40, 79)]).tolist() == [4]


def _stuck_at(distance: int) -> UntriedActions:
    """Untried actions with 1-1 stuck at the cell of `distance`."""
    untried = UntriedActions(cell_size=16, num_actions=3, patience=2)
    for _ in range(3):
        untried.record_episode("1-1", False, distance)
    return untried


def test_actions_are_untried_until_taken_in_their_cell():
    untried = _stuck_at(40)
    position = (1, 1, 0, 40, 79)
    assert untried.untried(["1-1"], [40], [position], [True]).tolist() == [[True] * 3]
    # Taken by two workers
    untried.record([position, position], [True, True], [0, 2])
    # Elsewhere in the same cell, in the air there, and in the next cell
    positions = [(1, 1, 0, 47, 64), (1, 1, 0, 40, 79), (1, 1, 0, 48, 79)]
    untried_now = untried.untried(["1-1"] * 3, [40] * 3, positions, [True, False, True])
    assert untried_now.tolist() == [[False, True, False], [True] * 3, [True] * 3]

    restored = UntriedActions(cell_size=16, num_actions=3, patience=2)
    restored.load_state_dict(untried.state_dict())
    assert restored.untried(["1-1"], [40], [position], [True]).tolist() == [[False, True, False]]


def test_actions_are_only_untried_before_a_wall():
    untried = UntriedActions(cell_size=16, num_actions=3, patience=2)
    position = (1, 1, 0, 40, 79)
    untried.record_episode("1-1", False, 40)
    # Not stuck yet: one episode without getting further, and one getting a cell further
    untried.record_episode("1-1", False, 47)
    untried.record_episode("1-1", False, 48)
    untried.record_episode("1-1", False, 40)
    assert not untried.untried(["1-1"], [40], [position], [True]).any()
    untried.record_episode("1-1", False, 40)
    assert untried.untried(["1-1"], [48], [position], [True]).all()
    # Before the window, after it and on another level
    assert not untried.untried(["1-1", "1-1", "1-2"], [47, 64, 48], [position] * 3, [True] * 3).any()
    assert untried.get_metrics() == {"exploration/stuck_levels": 1, "exploration/window_1-1": 16}


def test_window_moves_back_from_the_wall_until_progress():
    untried = _stuck_at(40)
    position = (1, 1, 0, 40, 79)
    untried.untried(["1-1"], [40], [position], [True])
    untried.record([position] * 3, [True] * 3, [0, 1, 2])
    # Untried actions were found in the window in this iteration
    untried.end_iteration()
    assert untried.get_metrics()["exploration/window_1-1"] == 16
    assert not untried.untried(["1-1"], [40], [position], [True]).any()
    untried.end_iteration()
    assert untried.get_metrics()["exploration/window_1-1"] == 32
    # Also when no episode got into the window
    untried.end_iteration()
    assert untried.get_metrics()["exploration/window_1-1"] == 48
    earlier = (1, 1, 0, 20, 79)
    assert untried.untried(["1-1"], [20], [earlier], [True]).all()

    restored = UntriedActions(cell_size=16, num_actions=3, patience=2)
    restored.load_state_dict(untried.state_dict())
    assert restored.get_metrics()["exploration/window_1-1"] == 48

    # A flag closes the wall without moving it
    untried.record_episode("1-1", True, 20)
    assert not untried.untried(["1-1"], [20], [earlier], [True]).any()
    assert untried.walls["1-1"].best == 2
