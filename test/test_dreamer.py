from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from agents import StepOutcome
from agents.dreamer import DreamerAgent
from agents.dreamer.agent import cmpo_targets, lambda_return
from agents.dreamer.best_episodes import BestEpisodes
from agents.dreamer.distributions import TwoHot, twohot_bins
from agents.dreamer.preview import PREDICTED_COLOR, SEEN_COLOR, render_preview, save_gif
from agents.dreamer.replay import Latent, Replay
from config.agent import DreamerAgentConfig
from environment import EnvironmentInfo

ENV_INFO = EnvironmentInfo(width=8, height=8, stack_frames=1, image_channels=1, num_actions=3, num_workers=2)


def tiny_config(**overrides) -> DreamerAgentConfig:
    """The configured settings with tiny networks."""
    values = OmegaConf.to_container(OmegaConf.load(Path(__file__).parents[1] / "configs" / "agent" / "dreamer.yaml"))
    assert isinstance(values, dict)
    values.pop("_target_")
    values.update(
        deter=16,
        blocks=2,
        hidden=8,
        stoch=4,
        classes=4,
        depth=2,
        mults=[2, 2],
        kernel=3,
        decoder_blocks=2,
        units=8,
        bins=15,
        batch_size=2,
        batch_length=6,
        replay_size=200,
        imagination_length=3,
        warmup=1,
        learning_rate=1e-3,
    )
    return DreamerAgentConfig(**{**values, **overrides})


@pytest.fixture
def agent(monkeypatch) -> DreamerAgent:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(0)
    # Half of each batch of 2 from the best episodes, so that self-imitation takes part
    return DreamerAgent(tiny_config(sil_fraction=0.5), ENV_INFO)


def _observations(value: int) -> np.ndarray:
    """Frames of both workers filled with the given value, so that they can be recognized in the replay."""
    return np.full((ENV_INFO.num_workers, 1, ENV_INFO.height, ENV_INFO.width, 1), value, np.uint8)


def _outcome(
    observations, rewards, terminated, truncated=(False, False), final_value: int | None = None, rarity=None
) -> StepOutcome:
    final = None if final_value is None else np.array(list(_observations(final_value)))
    rewards = np.asarray(rewards, float)
    return StepOutcome(
        observations,
        rewards,
        np.asarray(terminated),
        np.asarray(truncated),
        ["1-1"] * 2,
        final,
        game_rewards=rewards,
        rarity=None if rarity is None else np.asarray(rarity),
    )


def test_lambda_return():
    rewards = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    values = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    no = torch.zeros(1, 4)

    def returns(is_last=no, is_terminal=no, lam=1.0) -> list[float]:
        return lambda_return(is_last, is_terminal, rewards, values, 0.5, lam)[0].tolist()

    # Lambda 1: discounted rewards up to the end, bootstrapped from the last value
    assert returns() == pytest.approx([1 + 0.5 * 2 + 0.25 * 3 + 0.125 * 40, 2 + 0.5 * 3 + 0.25 * 40, 3 + 0.5 * 40])
    # Lambda 0: one-step targets
    assert returns(lam=0.0) == pytest.approx([1 + 0.5 * 20, 2 + 0.5 * 30, 3 + 0.5 * 40])
    # No value after a terminal step
    assert returns(is_terminal=torch.tensor([[0.0, 0.0, 1.0, 0.0]]))[:2] == pytest.approx([1 + 0.5 * 2, 2])
    # Nothing of the following episode after the last step of one, only its value
    assert returns(is_last=torch.tensor([[0.0, 1.0, 0.0, 0.0]]))[0] == pytest.approx(1 + 0.5 * 20)


def test_cmpo_targets():
    log_prior = torch.log(torch.tensor([0.6, 0.399, 0.001]))
    # Equal advantages keep the prior, larger ones raise an action's probability at most e-fold against the others
    assert torch.allclose(cmpo_targets(log_prior, torch.zeros(3)), log_prior.exp())
    boosted = cmpo_targets(log_prior, torch.tensor([0.0, 0.0, 5.0]))
    assert boosted[2] / boosted[0] == pytest.approx(np.e * 0.001 / 0.6)


def test_twohot():
    bins = twohot_bins(15)
    assert bins[7] == 0 and torch.allclose(bins, -bins.flip(0))
    # Uniform probabilities predict exactly 0
    assert TwoHot(torch.zeros(3, 15), bins).mean().tolist() == [0.0, 0.0, 0.0]
    # The loss is lowest for the distribution that predicts the target
    target = torch.tensor([2.5])
    logits = torch.zeros(1, 15, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.1)
    for _ in range(500):
        optimizer.zero_grad()
        TwoHot(logits, bins).loss(target).sum().backward()
        optimizer.step()
    assert TwoHot(logits, bins).mean().item() == pytest.approx(2.5, rel=1e-2)


def _latent(t: int) -> Latent:
    """A model state that tells the step it was stored for."""
    return Latent(np.full(2, t, np.float16), np.full(1, t % 4, np.uint8))


def _fill(replay: Replay, worker: int, steps: int, start: int = 0) -> None:
    for t in range(start, start + steps):
        replay.add(worker, np.full((1, 1, 1), t % 256, np.uint8), t % 3, float(t), t % 7 == 0, _latent(t))


def test_replay_samples_consecutive_steps_of_one_worker():
    replay = Replay(capacity=40, num_workers=2, frame_shape=(1, 1, 1), length=5, deter=2, stoch=1, seed=0)
    _fill(replay, 0, 30)
    _fill(replay, 1, 3)
    assert len(replay) == 23 and replay.can_sample()
    # The online sequences of worker 0 come first, in order, except those already overwritten
    batch = replay.sample(4)
    assert batch["rewards"][:, 0].tolist() == [10, 15, 20, 25]
    for _ in range(20):
        batch = replay.sample(8)
        rewards = batch["rewards"]
        # Worker 1's 3 steps are too few, and worker 0 only keeps its last 20 steps
        assert np.all(np.diff(rewards, axis=1) == 1) and rewards.min() >= 10
        # Each sequence starts from the state and action of the step before, or fresh from step 10, the oldest kept
        first = rewards[:, 0].astype(int)
        assert batch["is_first"][:, 0].tolist() == ((first == 10) | (first % 7 == 0)).tolist()
        before = first[first > 10] - 1
        assert batch["start_deter"][first > 10, 0].tolist() == before.tolist()
        assert batch["start_stoch"][first > 10, 0].tolist() == (before % 4).tolist()
        assert batch["start_action"][first > 10].tolist() == (before % 3).tolist()
        # A sequence cut off by the start of the next episode ends before it
        cut = batch["is_first"][:, 1:]
        assert np.all(batch["is_last"][:, :-1][cut])


def test_replay_samples_rare_steps_more_often():
    replay = Replay(capacity=200, num_workers=2, frame_shape=(1, 1, 1), length=5, deter=2, stoch=1, seed=0)
    # Worker 0 has overwritten its first 30 steps, worker 1 has one rare step among common ones. Frames hold the worker.
    for worker, steps in ((0, 130), (1, 100)):
        for t in range(steps):
            rarity = 1.0 if (worker, t) == (1, 50) else 0.01
            replay.add(worker, np.full((1, 1, 1), worker, np.uint8), 0, float(t), t == 0, _latent(t), rarity=rarity)
    batch = replay.sample(1000, online=False, num_rare=500)
    rewards, workers = batch["rewards"], batch["frames"][:, 0, 0, 0, 0]
    assert np.all(np.diff(rewards, axis=1) == 1) and rewards[workers == 0].min() >= 30
    # The rare step is picked with probability 1 / (1 + 199 * 0.01), at any position of the sequence
    contains = (rewards == 50).any(1) & (workers == 1)
    uniform, rare = contains[:500], contains[500:]
    assert uniform.mean() < 0.1 and 0.25 < rare.mean() < 0.45
    assert set(rewards[500:][rare, 0].tolist()) == {46, 47, 48, 49, 50}


def _finish(best: BestEpisodes, rewards: list[float], game_reward: float, terminal: bool = True) -> None:
    """An episode of one worker with the given rewards, the last one with the final frame, its game reward at the end"""
    for t, reward in enumerate(rewards[:-1]):
        best.add(0, np.full((1, 1, 1), t, np.uint8), t % 3 + 1, reward, 0.0, _latent(t + 1))
    best.finish(0, np.full((1, 1, 1), 99, np.uint8), rewards[-1], game_reward, terminal, "1-3")


def test_best_episodes_keeps_the_best_per_level():
    best = BestEpisodes(num_workers=1, per_level=2, length=3, discount=0.5, deter=2, stoch=1, seed=0)
    for game_reward in (1.0, 3.0, 2.0, 0.5):
        _finish(best, [0.0, 1.0, 2.0, 4.0], game_reward)
    # Too short for a sequence
    _finish(best, [0.0, 9.0], 10.0)
    assert sorted(e.game_reward for e in best.episodes["1-3"]) == [2.0, 3.0]
    # Returns of the rewards received after each step
    episode = best.episodes["1-3"][0]
    assert episode.returns.tolist() == pytest.approx([1 + 0.5 * 2 + 0.25 * 4, 2 + 0.5 * 4, 4, 0])


def test_best_episodes_sample_sequences():
    best = BestEpisodes(num_workers=1, per_level=1, length=3, discount=0.5, deter=2, stoch=1, seed=0)
    _finish(best, [0.0, 1.0, 2.0, 4.0], 1.0)
    batch = best.sample(16)
    assert batch["frames"].shape == (16, 3, 1, 1, 1) and batch["returns"].shape == (16, 3)
    starts = batch["frames"][:, 0, 0, 0, 0]
    assert set(starts.tolist()) == {0, 1}
    # Sequences from the start of the episode start fresh, the others from the first step's state and action
    assert np.array_equal(batch["is_first"][:, 0], starts == 0) and not batch["is_first"][:, 1:].any()
    assert batch["start_deter"][starts == 1, 0].tolist() == [1.0] * int((starts == 1).sum())
    assert batch["start_action"][starts == 1].tolist() == [1] * int((starts == 1).sum())
    # Only sequences up to the final frame end the episode, which ended in the game
    ends = starts == 1
    assert np.array_equal(batch["is_last"][:, -1], ends) and np.array_equal(batch["is_terminal"][:, -1], ends)
    assert not batch["is_last"][:, :-1].any()


def test_agent_stores_the_reward_with_the_frame_it_leads_to(agent):
    agent.act(_observations(1))
    agent.observe(_outcome(_observations(2), [0.5, 0.5], [False, False], rarity=[0.5, 0.25]))
    agent.act(_observations(2))
    # Worker 0 dies, worker 1 is cut off, both continue in a new episode
    agent.observe(_outcome(_observations(4), [-2.0, 1.0], [True, False], [False, True], 3, rarity=[0.1, 0.2]))
    agent.act(_observations(4))

    replay = agent.replay
    assert replay.added.tolist() == [4, 4]
    assert replay.frames[:, :4, 0, 0, 0].tolist() == [[1, 2, 3, 4]] * 2
    assert replay.rewards[:, :4].tolist() == [[0.0, 0.5, -2.0, 0.0], [0.0, 0.5, 1.0, 0.0]]
    assert replay.is_first[:, :4].tolist() == [[True, False, False, True]] * 2
    assert replay.is_last[:, :4].tolist() == [[False, False, True, False]] * 2
    assert replay.is_terminal[:, :4].tolist() == [[False, False, True, False], [False] * 4]
    # Like the reward, with the frame it was reached in. A new episode's first frame has not been visited yet.
    assert replay.rarity[:, :4] == pytest.approx(np.array([[1, 0.5, 0.1, 1], [1, 0.25, 0.2, 1]]))
    # The model state of each step the agent acted in, none for the final frames. A fresh start's is all zeros.
    assert replay.deter[:, 1].any(-1).all() and not replay.deter[:, 2].any()


def test_untried_actions_are_taken_first(agent):
    # Worker 0 has not tried action 1 yet, worker 1 has tried all
    untried = np.array([[False, True, False], [False, False, False]])
    actions = [agent.act(_observations(1), untried=untried) for _ in range(10)]
    assert [action for action, _ in actions] == [1] * 10
    # And learns from the action it took
    assert agent.replay.actions[0, :10].tolist() == [1] * 10


def test_greedy_actions_are_not_learned_from(agent):
    agent.act(_observations(1), greedy=True)
    agent.observe(_outcome(_observations(2), [1.0, 1.0], [False, False]))
    assert len(agent.replay) == 0
    assert agent.update() == {}


def test_trains_while_playing_and_reports_on_update(agent):
    agent.config.train_ratio = 1.0
    rng = np.random.default_rng(0)
    for _ in range(13):
        observations = rng.integers(0, 256, (2, 1, 8, 8, 1), dtype=np.uint8)
        agent.act(observations)
        agent.train_while_playing()
        agent.observe(_outcome(observations, rng.normal(size=2), [False, False]))
    # Each of the 26 steps of both workers is replayed once in sequences of 2 x 6 steps
    assert agent.updates == 2
    metrics = agent.update()
    assert metrics["progress/updates"] == 2 and "world_model/reconstruction" in metrics
    # The metrics are reported once
    assert agent.update() == {}


def test_update_trains_all_networks(agent):
    agent.config.train_ratio = 1.0
    rng = np.random.default_rng(0)
    for t in range(30):
        observations = rng.integers(0, 256, (2, 1, 8, 8, 1), dtype=np.uint8)
        agent.act(observations)
        done = t % 10 == 9
        agent.observe(_outcome(observations, rng.normal(size=2), [done, False], final_value=0 if done else None))
    before = {name: p.detach().clone() for name, p in agent.model.named_parameters()}
    slow_before = [p.clone() for p in agent.slow_critic.parameters()]

    # 60 steps and 3 final ones, each replayed once in sequences of 2 x 6 steps
    metrics = agent.update()
    assert metrics["progress/updates"] == agent.updates == 5
    assert all(np.isfinite(value) for value in metrics.values())
    assert "exploration/plan2explore_reward" in metrics and "behavior/cmpo_changes" in metrics
    # Worker 0's three episodes of 10 steps and the final frame
    assert agent.best_episodes is not None
    assert sum(len(episodes) for episodes in agent.best_episodes.episodes.values()) == 3
    changed = {name.split(".")[0] for name, p in agent.model.named_parameters() if not torch.equal(p, before[name])}
    assert changed == {
        "encoder",
        "rssm",
        "decoder",
        "reward",
        "continuation",
        "actor",
        "critic",
        "explorer",
        "exploration_critic",
    }
    assert any(not torch.equal(a, b) for a, b in zip(slow_before, agent.slow_critic.parameters(), strict=True))


def test_cmpo_label_smoothing_pulls_up_ruled_out_actions(agent):
    # The actor has ruled out action 0 everywhere, which the uniform mixture still samples now and then
    with torch.no_grad():
        agent.model.actor.out.bias[0] = -60.0
    rng = np.random.default_rng(0)
    for _ in range(10):
        observations = rng.integers(0, 256, (2, 1, 8, 8, 1), dtype=np.uint8)
        agent.act(observations)
        agent.observe(_outcome(observations, rng.normal(size=2), [False, False]))
    gradients = {}
    for smoothing in (0.0, 0.1):
        agent.config.cmpo_label_smoothing = smoothing
        agent._train_step(agent._sample())
        gradients[smoothing] = agent.model.actor.out.bias.grad[0].item()
    # Every other loss changes the action's probability in proportion to it, label smoothing by smoothing / actions
    assert abs(gradients[0.0]) < 1e-12 and gradients[0.1] < -1e-4


def test_state_is_restored_with_the_experience(agent, tmp_path):
    rng = np.random.default_rng(0)
    for t in range(8):
        observations = rng.integers(0, 256, (2, 1, 8, 8, 1), dtype=np.uint8)
        agent.act(observations)
        # Worker 0's episode ends after 7 steps, long enough to be kept for self-imitation
        agent.observe(_outcome(observations, [1.0, 1.0], [t == 6, False], final_value=3 if t == 6 else None))
    agent.updates, agent.update_budget = 3, 0.5
    # Saved and loaded like the state of training, see resume.py
    torch.save(agent.state_dict(), tmp_path / "state.pt")
    restored = DreamerAgent(tiny_config(sil_fraction=0.5), ENV_INFO)
    restored.load_state_dict(torch.load(tmp_path / "state.pt", weights_only=True, mmap=True))

    assert restored.updates == 3 and restored.update_budget == 0.5
    for name, value in agent.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])
    replay, before = restored.replay, agent.replay
    assert np.array_equal(replay.added, before.added) and replay.online == before.online
    for name in replay.STEP_ARRAYS:
        if name != "is_last":
            assert np.array_equal(getattr(replay, name), getattr(before, name)), name
    # The episodes in progress end where they were cut off
    last = before.added - 1
    assert replay.is_last[[0, 1], last].all()
    replay.is_last[[0, 1], last] = before.is_last[[0, 1], last]
    assert np.array_equal(replay.is_last, before.is_last)
    (episode,) = restored.best_episodes.episodes["1-1"]
    (original,) = agent.best_episodes.episodes["1-1"]
    assert np.array_equal(episode.frames, original.frames) and episode.game_reward == original.game_reward == 7.0
    assert restored.replay.sample(2)["frames"].shape[:2] == (2, 6)


def test_predictions_follow_the_given_actions_or_the_policy(agent):
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 256, (3, 8, 8, 8, 1), dtype=np.uint8)
    batch = {
        "frames": frames,
        "actions": rng.integers(0, 3, (3, 8)),
        "is_first": np.zeros((3, 8), bool),
        "start_deter": rng.normal(size=(3, agent.config.deter)).astype(np.float16),
        "start_stoch": rng.integers(0, agent.config.classes, (3, agent.config.stoch), dtype=np.uint8),
        "start_action": rng.integers(0, 3, 3),
    }

    torch.manual_seed(0)
    predicted = agent.predict(batch, context=2)
    assert predicted.shape == frames.shape and predicted.dtype == np.uint8
    # The frames after the context are not seen, so changing them changes nothing
    torch.manual_seed(0)
    changed = frames.copy()
    changed[:, 2:] = 0
    assert np.array_equal(agent.predict({**batch, "frames": changed}, context=2), predicted)
    # The predictions continue from the state before the sequence
    torch.manual_seed(0)
    assert not np.array_equal(agent.predict({**batch, "start_deter": -batch["start_deter"]}, context=2), predicted)
    torch.manual_seed(0)
    assert agent.predict(batch, context=2, dream=True).shape == frames.shape


def test_render_preview():
    true = np.zeros((2, 5, 4, 6, 1), np.uint8)
    predicted = np.full((2, 5, 4, 6, 1), 200, np.uint8)
    video = render_preview(true, predicted, context=2, scale=2)
    # Per step, the sequences side by side, each with the true and predicted frames and their difference, framed by
    # 2 pixels on each side
    assert len(video) == 5
    assert video[0].shape == (3 * (8 + 4), 2 * (12 + 4), 3)
    assert tuple(video[0][0, 0]) == SEEN_COLOR and tuple(video[2][0, 0]) == PREDICTED_COLOR
    assert video[0][2 + 12 + 2, 2, 0] == 200
    assert video[0][2 + 2 * 12 + 2, 2, 0] == (200 + 255) // 2
    # Only the predictions
    assert render_preview(None, predicted, context=2)[0].shape == (8 + 4, 2 * (12 + 4), 3)


def test_save_gif(tmp_path):
    video = [np.full((4, 6, 3), value, np.uint8) for value in (0, 255)]
    save_gif(video, tmp_path / "preview.gif")
    image = Image.open(tmp_path / "preview.gif")
    assert image.n_frames == 2 and image.size == (6, 4)
