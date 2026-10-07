import pytest

from config.levels import (
    ALL_LEVELS,
    CurriculumLevelsConfig,
    ProgressionLevelsConfig,
    RandomLevelsConfig,
    SingleLevelConfig,
    level_index,
)
from level_schedulers import CurriculumLevelScheduler, LevelScheduler, get_level_scheduler


def test_all_levels_in_game_order():
    assert len(ALL_LEVELS) == 32
    assert ALL_LEVELS[:5] == ["1-1", "1-2", "1-3", "1-4", "2-1"]
    assert [level_index(level) for level in ALL_LEVELS] == list(range(32))


@pytest.mark.parametrize("level", ["0-1", "9-1", "1-5", "11", "a-b"])
def test_invalid_levels_are_rejected(level):
    with pytest.raises(AssertionError):
        SingleLevelConfig(level)


def test_duplicate_levels_are_rejected():
    with pytest.raises(AssertionError):
        RandomLevelsConfig(["1-1", "1-1"])


def test_distributions():
    assert SingleLevelConfig("4-2").initial_distribution() == (["4-2"], [1.0])
    assert RandomLevelsConfig(["1-1", "1-2"]).initial_distribution() == (["1-1", "1-2"], [0.5, 0.5])
    assert ProgressionLevelsConfig().initial_distribution() is None

    curriculum = CurriculumLevelsConfig(levels=["1-1", "1-2", "1-3", "1-4"], initial_levels=2, min_level_weight=0.1)
    # Without results, all unlocked levels are equally likely
    assert curriculum.initial_distribution() == (["1-1", "1-2"], [0.5, 0.5])
    assert curriculum.evaluation_distribution() == (curriculum.levels, [0.25] * 4)


def test_names():
    assert SingleLevelConfig("1-1").name == "1-1"
    assert RandomLevelsConfig().name == "random-32"
    assert CurriculumLevelsConfig().name == "curriculum-32"
    assert ProgressionLevelsConfig().name == "progression"


def test_static_scheduler():
    scheduler = get_level_scheduler(RandomLevelsConfig(["1-1", "1-2"]))
    assert type(scheduler) is LevelScheduler
    scheduler.record_episode("1-1", True)
    assert not scheduler.update()
    assert scheduler.distribution() == (["1-1", "1-2"], [0.5, 0.5])


def _curriculum(**kwargs) -> CurriculumLevelScheduler:
    kwargs = {"levels": ["1-1", "1-2", "1-3"], "success_window": 4, "min_level_episodes": 2, **kwargs}
    scheduler = get_level_scheduler(CurriculumLevelsConfig(**kwargs))
    assert isinstance(scheduler, CurriculumLevelScheduler)
    return scheduler


def _record(scheduler, level, results):
    for success in results:
        scheduler.record_episode(level, success)


def test_curriculum_unlocks_after_mastery():
    scheduler = _curriculum(unlock_success_rate=0.5)
    _record(scheduler, "1-1", [True, False, True])
    # Window of the newest level not filled yet
    scheduler.update()
    assert scheduler.num_unlocked == 1

    _record(scheduler, "1-1", [False])
    scheduler.update()
    assert scheduler.num_unlocked == 2
    assert scheduler.distribution()[0] == ["1-1", "1-2"]
    assert scheduler.get_metrics() == {
        "num_unlocked_levels": 2,
        "success_rate_1-1": 0.5,
        "levels_cleared_score": 0.5,
        "mean_levels_cleared_score": pytest.approx((2 / 3 + 0.5) / 2),
    }


def test_curriculum_requires_mastery_of_all_unlocked_levels():
    scheduler = _curriculum(initial_levels=2, unlock_mean_success_rate=0.6)
    _record(scheduler, "1-2", [True] * 4)

    # The newest level is mastered, but the earlier one has too few recent episodes
    _record(scheduler, "1-1", [False])
    scheduler.update()
    assert scheduler.num_unlocked == 2

    # Enough episodes, but the earlier level was forgotten: the mean success (1 + 0) / 2 is below 0.6
    _record(scheduler, "1-1", [False])
    scheduler.update()
    assert scheduler.num_unlocked == 2

    # Relearned: the mean success (1 + 0.5) / 2 reaches the threshold
    _record(scheduler, "1-1", [True, True])
    scheduler.update()
    assert scheduler.num_unlocked == 3


def test_hard_level_does_not_block_progress():
    scheduler = _curriculum(levels=["1-1", "1-2", "1-3", "1-4"], initial_levels=3, unlock_mean_success_rate=0.7)
    _record(scheduler, "1-1", [True] * 4)
    _record(scheduler, "1-2", [True, False, False, False])
    # The newest level is mastered, the mean (1 + 0.25 + 0.5) / 3 is too low
    _record(scheduler, "1-3", [True, True, False, False])
    scheduler.update()
    assert scheduler.num_unlocked == 3
    # With the newest level better, the hard level no longer blocks: (1 + 0.25 + 1) / 3
    _record(scheduler, "1-3", [True] * 4)
    scheduler.update()
    assert scheduler.num_unlocked == 4


def test_forgotten_level_blocks_progress():
    scheduler = _curriculum(initial_levels=2, unlock_success_rate=0.75, unlock_mean_success_rate=0.75)
    _record(scheduler, "1-2", [True] * 4)
    _record(scheduler, "1-1", [False] * 4)
    for _ in range(10):
        scheduler.update()
    assert scheduler.num_unlocked == 2
    # And the forgotten level is now the most likely one
    _levels, weights = scheduler.distribution()
    assert weights[0] > weights[1]


def test_distribution_updates_without_unlock():
    scheduler = _curriculum(initial_levels=2)
    # Initially uniform, as already used by the workers
    assert not scheduler.update()
    _record(scheduler, "1-1", [True, True])
    assert scheduler.update()
    assert scheduler.distribution()[1] == pytest.approx([0.1 / 1.1, 1 / 1.1])
    # No change since the last update
    assert not scheduler.update()


def test_curriculum_samples_by_need():
    scheduler = _curriculum(levels=["1-1", "1-2", "1-3", "1-4"], initial_levels=4, min_level_weight=0.1)
    _record(scheduler, "1-1", [True])
    _record(scheduler, "1-2", [True, False])
    _record(scheduler, "1-3", [False])
    levels, weights = scheduler.distribution()
    assert levels == ["1-1", "1-2", "1-3", "1-4"]
    # Weights 0.1 (floor for a mastered level), 0.5, 1 and 1 (no results yet)
    assert weights == pytest.approx([0.1 / 2.6, 0.5 / 2.6, 1 / 2.6, 1 / 2.6])


def test_curriculum_stops_at_last_level():
    scheduler = _curriculum(initial_levels=3)
    for level in ["1-1", "1-2", "1-3"]:
        _record(scheduler, level, [True] * 4)
    scheduler.update()
    assert scheduler.num_unlocked == 3


def test_mean_score_over_updates():
    scheduler = _curriculum(initial_levels=2)
    assert scheduler.mean_score() == 0.0
    _record(scheduler, "1-1", [True])
    scheduler.update()
    _record(scheduler, "1-2", [True])
    scheduler.update()
    assert scheduler.mean_score() == pytest.approx((1 + 2) / 2)


def test_curriculum_score_and_unknown_levels():
    scheduler = _curriculum(initial_levels=2)
    scheduler.record_episode("8-4", True)
    assert scheduler.score() == 0.0
    _record(scheduler, "1-1", [True, False])
    _record(scheduler, "1-2", [True])
    assert scheduler.score() == pytest.approx(1.5)
    assert LevelScheduler(SingleLevelConfig()).score() is None


def test_curriculum_state_is_restored():
    scheduler = _curriculum(unlock_success_rate=0.5)
    _record(scheduler, "1-1", [True] * 4)
    scheduler.update()
    restored = _curriculum(unlock_success_rate=0.5)
    restored.load_state_dict(scheduler.state_dict())
    assert restored.num_unlocked == 2
    assert restored.success_rate("1-1") == 1.0
    assert restored.mean_score() == scheduler.mean_score()
    assert restored.distribution() == scheduler.distribution()
