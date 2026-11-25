import pytest

from config.levels import ALL_LEVELS, CurriculumLevelsConfig, SingleLevelConfig
from level_schedulers import CurriculumLevelScheduler, LevelScheduler, get_level_scheduler


def test_all_levels_in_game_order():
    assert len(ALL_LEVELS) == 32
    assert ALL_LEVELS[:5] == ["1-1", "1-2", "1-3", "1-4", "2-1"]
    assert ALL_LEVELS[-1] == "8-4"


@pytest.mark.parametrize("level", ["0-1", "9-1", "1-5", "11", "a-b"])
def test_invalid_levels_are_rejected(level):
    with pytest.raises(AssertionError):
        SingleLevelConfig(level)


def test_a_level_needs_episodes_to_count():
    with pytest.raises(AssertionError):
        SingleLevelConfig("1-1", success_window=0)


def test_duplicate_levels_are_rejected():
    with pytest.raises(AssertionError):
        CurriculumLevelsConfig(levels=["1-1", "1-1"])


def test_distributions():
    assert SingleLevelConfig("4-2").initial_distribution() == (["4-2"], [1.0])
    assert SingleLevelConfig("4-2").evaluation_levels() == ["4-2"]
    curriculum = CurriculumLevelsConfig(levels=["1-1", "1-2", "1-3", "1-4"])
    # The curriculum starts with its first level and is evaluated on all
    assert curriculum.initial_distribution() == (["1-1"], [1.0])
    assert curriculum.evaluation_levels() == curriculum.levels


def test_names():
    assert SingleLevelConfig("1-1").name == "1-1"
    assert CurriculumLevelsConfig().name == "curriculum-32"


def test_static_scheduler():
    scheduler = get_level_scheduler(SingleLevelConfig("1-1", success_window=2))
    assert type(scheduler) is LevelScheduler
    for flag_get, distance in [(False, 100), (True, 3000), (True, 3200)]:
        scheduler.record_episode("1-1", flag_get, distance)
    assert not scheduler.update()
    assert scheduler.distribution() == (["1-1"], [1.0])
    # The results of the last two episodes
    assert scheduler.get_metrics() == {
        "levels/cleared": 1.0,
        "levels/success_1-1": 1.0,
        "levels/distance_1-1": 3100.0,
    }


def _curriculum(unlocked: int = 1, **kwargs) -> CurriculumLevelScheduler:
    """A curriculum over 1-1 to 1-3 with windows of 4 episodes, with the first `unlocked` levels unlocked."""
    kwargs = {"levels": ["1-1", "1-2", "1-3"], "success_window": 4, "min_level_episodes": 2, **kwargs}
    scheduler = get_level_scheduler(CurriculumLevelsConfig(**kwargs))
    assert isinstance(scheduler, CurriculumLevelScheduler)
    scheduler.num_unlocked = unlocked
    return scheduler


def _record(scheduler, level, results):
    for success in results:
        scheduler.record_episode(level, success, 3000 if success else 1000)


def test_curriculum_unlocks_after_a_flag():
    scheduler = _curriculum(unlock_mean_success_rate=0.25)
    _record(scheduler, "1-1", [False, False, True])
    # Window of the newest level not filled yet
    scheduler.update()
    assert scheduler.num_unlocked == 1

    # One flag in a full window is enough
    _record(scheduler, "1-1", [False])
    scheduler.update()
    assert scheduler.num_unlocked == 2
    assert scheduler.distribution()[0] == ["1-1", "1-2"]
    assert scheduler.get_metrics() == {
        "levels/unlocked": 2,
        "levels/success_1-1": 0.25,
        "levels/distance_1-1": 1500.0,
        "levels/cleared": 0.25,
    }


def test_curriculum_waits_for_a_flag_on_the_newest_level():
    scheduler = _curriculum(unlocked=2, unlock_mean_success_rate=0.5)
    _record(scheduler, "1-1", [True] * 4)
    _record(scheduler, "1-2", [False] * 4)
    # The mean (1 + 0) / 2 is high enough, but the newest level has no flag
    scheduler.update()
    assert scheduler.num_unlocked == 2


def test_curriculum_requires_mastery_of_all_unlocked_levels():
    scheduler = _curriculum(unlocked=2, unlock_mean_success_rate=0.6)
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
    scheduler = _curriculum(unlocked=3, levels=["1-1", "1-2", "1-3", "1-4"], unlock_mean_success_rate=0.7)
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
    scheduler = _curriculum(unlocked=2, unlock_mean_success_rate=0.75)
    _record(scheduler, "1-2", [True] * 4)
    _record(scheduler, "1-1", [False] * 4)
    for _ in range(10):
        scheduler.update()
    assert scheduler.num_unlocked == 2
    # And the forgotten level is now the most likely one
    _levels, weights = scheduler.distribution()
    assert weights[0] > weights[1]


def test_distribution_updates_without_unlock():
    scheduler = _curriculum(unlocked=2)
    # Without results, the unlocked levels are equally likely
    assert scheduler.update()
    assert scheduler.distribution() == (["1-1", "1-2"], [0.5, 0.5])
    _record(scheduler, "1-1", [True, True])
    assert scheduler.update()
    assert scheduler.distribution()[1] == pytest.approx([0.1 / 1.1, 1 / 1.1])
    # No change since the last update
    assert not scheduler.update()


def test_curriculum_samples_by_need():
    scheduler = _curriculum(unlocked=4, levels=["1-1", "1-2", "1-3", "1-4"], min_level_weight=0.1)
    _record(scheduler, "1-1", [True])
    _record(scheduler, "1-2", [True, False])
    _record(scheduler, "1-3", [False])
    levels, weights = scheduler.distribution()
    assert levels == ["1-1", "1-2", "1-3", "1-4"]
    # Weights 0.1 (floor for a mastered level), 0.5, 1 and 1 (no results yet)
    assert weights == pytest.approx([0.1 / 2.6, 0.5 / 2.6, 1 / 2.6, 1 / 2.6])


def test_curriculum_stops_at_last_level():
    scheduler = _curriculum(unlocked=3)
    for level in ["1-1", "1-2", "1-3"]:
        _record(scheduler, level, [True] * 4)
    scheduler.update()
    assert scheduler.num_unlocked == 3


def test_curriculum_score_and_unknown_levels():
    scheduler = _curriculum(unlocked=2)
    scheduler.record_episode("8-4", True, 3000)
    assert scheduler.score() == 0.0
    _record(scheduler, "1-1", [True, False])
    _record(scheduler, "1-2", [True])
    assert scheduler.score() == pytest.approx(1.5)


def test_curriculum_state_is_restored():
    scheduler = _curriculum()
    _record(scheduler, "1-1", [True] * 4)
    scheduler.update()
    restored = _curriculum()
    restored.load_state_dict(scheduler.state_dict())
    assert restored.num_unlocked == 2
    assert restored.success_rate("1-1") == 1.0
    assert restored.distribution() == scheduler.distribution()
