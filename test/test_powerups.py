from environment import Powerup
from powerups import StartPowerups


def test_levels_start_like_the_previous_level_was_finished():
    starts = StartPowerups(period=2)
    for powerup in (Powerup.SMALL, Powerup.FIRE, Powerup.FIRE, Powerup.BIG):
        starts.record_episode("1-1", flag_get=True, powerup=powerup)
    starts.record_episode("1-1", flag_get=False, powerup=Powerup.FIRE)
    starts.record_episode("1-2", flag_get=True, powerup=Powerup.BIG)
    # The finishes count until the end of the period
    assert not starts.update() and starts.distributions == {}
    assert starts.update()
    assert starts.distributions == {"1-2": [0.25, 0.25, 0.5], "1-3": [0.0, 1.0, 0.0]}
    assert starts.get_metrics() == {"powerups/start_1-2": 0.75, "powerups/start_1-3": 1.0}

    # The next period counts anew, and levels whose previous one reached no flag start small again
    starts.record_episode("1-1", flag_get=True, powerup=Powerup.SMALL)
    restored = StartPowerups(period=2)
    restored.load_state_dict(starts.state_dict())
    for distributions in (starts, restored):
        assert not distributions.update()
        assert distributions.update()
        assert distributions.distributions == {"1-2": [1.0, 0.0, 0.0]}
