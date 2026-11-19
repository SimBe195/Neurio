import copy
import tempfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import mlflow
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from agents.agent import Agent, StepOutcome, checkpoint_name, get_device
from config.agent import DreamerAgentConfig
from environment import EnvironmentInfo, Powerup

from .best_episodes import BestEpisodes
from .distributions import Categorical, sample_index
from .networks import BernoulliHead, Ensemble, PolicyHead, TwoHotHead
from .optimizer import LaProp
from .preview import render_preview, save_gif
from .replay import Latent, Replay
from .world_model import RSSM, Decoder, Encoder, State

# Number of replayed sequences in the preview stored with each checkpoint
PREVIEW_SEQUENCES = 6


class DreamerNetworks(nn.Module):
    """
    The world model (encoder, RSSM, decoder, reward and continuation heads), the actor and the critic, and for
    Plan2Explore the ensemble and the critic of its exploration reward.
    """

    def __init__(self, config: DreamerAgentConfig, env_info: EnvironmentInfo) -> None:
        super().__init__()
        height, width, channels = env_info.height, env_info.width, env_info.image_channels
        self.encoder = Encoder(config, channels)
        with torch.no_grad():
            token_size = self.encoder(torch.zeros(1, height, width, channels, dtype=torch.uint8)).shape[-1]
        self.rssm = RSSM(config, token_size, env_info.num_actions)
        self.decoder = Decoder(config, height, width, channels)
        features = config.deter + config.stoch * config.classes
        self.reward = TwoHotHead(features, config.head_layers, config.units, config.bins)
        self.continuation = BernoulliHead(features, config.head_layers, config.units)
        self.actor = PolicyHead(features, config.actor_critic_layers, config.units, env_info.num_actions, config.unimix)
        self.critic = TwoHotHead(features, config.actor_critic_layers, config.units, config.bins)
        self.explorer: Ensemble | None = None
        self.exploration_critic: TwoHotHead | None = None
        if config.exploration_weight > 0:
            # Predicts the next stochastic state from a state and action
            self.explorer = Ensemble(
                features + env_info.num_actions,
                config.exploration_heads,
                config.actor_critic_layers,
                config.units,
                config.stoch * config.classes,
            )
            self.exploration_critic = TwoHotHead(features, config.actor_critic_layers, config.units, config.bins)


def lambda_return(
    is_last: Float[torch.Tensor, "batch time"],
    is_terminal: Float[torch.Tensor, "batch time"],
    rewards: Float[torch.Tensor, "batch time"],
    bootstrap: Float[torch.Tensor, "batch time"],
    discount: float,
    lam: float,
) -> Float[torch.Tensor, "batch time-1"]:
    """
    TD(lambda) returns of all but the last step, bootstrapped from the given values. The reward of step t is the one
    received on reaching it. Returns do not continue past terminal steps, nor, except for the bootstrap value, past
    the last step of an episode.
    """
    live = (1 - is_terminal[:, 1:]) * discount
    cont = (1 - is_last[:, 1:]) * lam
    interim = rewards[:, 1:] + (1 - cont) * live * bootstrap[:, 1:]
    returns = [bootstrap[:, -1]]
    for t in reversed(range(live.shape[1])):
        returns.append(interim[:, t] + live[:, t] * cont[:, t] * returns[-1])
    return torch.stack(returns[:0:-1], 1)


# Bound of the normalized advantages in the CMPO target, as in Muesli
CMPO_ADVANTAGE_CLIP = 1.0


def cmpo_targets(
    log_prior: Float[torch.Tensor, "*batch actions"], advantages: Float[torch.Tensor, "*batch actions"]
) -> Float[torch.Tensor, "*batch actions"]:
    """
    The clipped MPO policy of Muesli (Hessel et al., 2021): the prior policy times exp(advantage), with the advantages
    clipped to ±CMPO_ADVANTAGE_CLIP, normalized. Each update can thus raise an action's probability at most e^2-fold
    relative to the others, while actions of equal value keep the prior's.
    """
    return torch.softmax(log_prior + advantages.clamp(-CMPO_ADVANTAGE_CLIP, CMPO_ADVANTAGE_CLIP), -1)


class DreamerAgent(Agent):
    """
    DreamerV3 (Hafner et al., 2023), reimplemented from the official JAX code. It learns a world model from replayed
    sequences of its experience, and the actor and critic from rollouts imagined by the world model. See README.md for
    the differences to the original.
    """

    config: DreamerAgentConfig

    def __init__(self, config: DreamerAgentConfig, env_info: EnvironmentInfo) -> None:
        super().__init__(config, env_info)
        assert env_info.stack_frames == 1, "Dreamer remembers earlier frames in its recurrent state."
        self.device = get_device()
        self.model = DreamerNetworks(config, env_info).to(self.device)
        self.imagine: Callable[[State], tuple[State, torch.Tensor]] = self._imagine
        if self.device.type == "cuda":
            # Faster convolutions in bfloat16
            self.model.encoder.to(memory_format=torch.channels_last)
            self.model.decoder.to(memory_format=torch.channels_last)
            # Compiled, the many small operations of each step run as few GPU kernels, and the loops over the steps of
            # a sequence or an imagined rollout as one call each, which more than halves the time of an update.
            # Compiling takes a few minutes at the start.
            self.model.encoder.compile()
            self.model.decoder.compile()
            for step in ("observe_step", "imagine_step", "observe"):
                setattr(self.model.rssm, step, torch.compile(getattr(self.model.rssm, step)))
            self.imagine = torch.compile(self._imagine)
        self.slow_critic = copy.deepcopy(self.model.critic).requires_grad_(False)
        self.slow_exploration_critic = copy.deepcopy(self.model.exploration_critic)
        if self.slow_exploration_critic is not None:
            self.slow_exploration_critic.requires_grad_(False)
        self.optimizer = LaProp(self.model.parameters(), config.learning_rate, config.agc, warmup=config.warmup)
        frame_shape = (env_info.height, env_info.width, env_info.image_channels)
        latent_sizes = {"deter": config.deter, "stoch": config.stoch}
        self.replay = Replay(config.replay_size, self.num_workers, frame_shape, config.batch_length, **latent_sizes)
        self.best_episodes = (
            BestEpisodes(
                self.num_workers, config.sil_episodes_per_level, config.batch_length, config.discount, **latent_sizes
            )
            if config.sil_fraction > 0
            else None
        )
        # Running percentiles of the imagined returns of the game's reward and of the exploration reward
        self.return_ranges = {name: torch.zeros(2, device=self.device) for name in ("reward", "exploration")}
        self.return_quantiles = torch.tensor(config.return_percentiles, device=self.device) / 100
        self.updates = 0
        # Updates owed to the replayed steps, see DreamerAgentConfig.train_ratio
        self.update_budget = 0.0
        # Metrics of the updates since the last `update`, still on the device
        self._metrics: defaultdict[str, list[torch.Tensor]] = defaultdict(list)

        self.state = self.model.rssm.initial(self.num_workers, self.device)
        self.prev_actions = torch.zeros(self.num_workers, env_info.num_actions, device=self.device)
        self.is_first = np.ones(self.num_workers, bool)
        # Reward received on reaching the current frame of each worker, and the game's reward, which ranks episodes
        self.rewards = np.zeros(self.num_workers, np.float32)
        self.game_rewards = np.zeros(self.num_workers, np.float32)
        # How rare the current frame of each worker is, see StepOutcome.rarity
        self.rarity = np.ones(self.num_workers, np.float32)
        self._recording = False

    @property
    def networks(self) -> dict[str, nn.Module]:
        return {"dreamer": self.model}

    def describe(self) -> str:
        def count(module: nn.Module) -> str:
            return f"{sum(p.numel() for p in module.parameters()):,} parameters"

        lines = [f"{name}: {count(module)}" for name, module in self.model.named_children()]
        return "\n".join([*lines, f"total: {count(self.model)}"])

    def state_dict(self) -> dict:
        """Everything training needs to continue, but the episodes in progress, which start anew."""
        return {
            "model": self.model.state_dict(),
            "slow_critic": self.slow_critic.state_dict(),
            "slow_exploration_critic": (
                self.slow_exploration_critic.state_dict() if self.slow_exploration_critic is not None else {}
            ),
            "optimizer": self.optimizer.state_dict(),
            "return_ranges": self.return_ranges,
            "updates": self.updates,
            "update_budget": self.update_budget,
            "replay": self.replay.state_dict(),
            "best_episodes": self.best_episodes.state_dict() if self.best_episodes is not None else {},
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.slow_critic.load_state_dict(state["slow_critic"])
        if self.slow_exploration_critic is not None:
            self.slow_exploration_critic.load_state_dict(state["slow_exploration_critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.return_ranges = {name: value.to(self.device) for name, value in state["return_ranges"].items()}
        self.updates = state["updates"]
        self.update_budget = state["update_budget"]
        # States saved before the experience was stored lack it
        if "replay" in state:
            self.replay.load_state_dict(state["replay"])
        if self.best_episodes is not None and "best_episodes" in state:
            self.best_episodes.load_state_dict(state["best_episodes"])

    @torch.no_grad()
    def act(self, observations: np.ndarray, greedy: bool = False, untried: np.ndarray | None = None) -> list[int]:
        frames = np.asarray(observations)[:, -1]
        tokens = self.model.encoder(torch.as_tensor(frames, device=self.device))
        is_first = torch.as_tensor(self.is_first, device=self.device)
        state = self.model.rssm.observe_step(self.state, self.prev_actions, tokens, is_first)
        policy = self.model.actor(state.features)
        actions = policy.mode() if greedy else policy.sample()
        if untried is not None and untried.any():
            # The untried action by the policy, which only matters for its order: all are tried in turn
            mask = torch.as_tensor(untried, device=self.device)
            untried_actions = sample_index(policy.log_probs.masked_fill(~mask, -torch.inf))
            actions = torch.where(mask.any(-1), untried_actions, actions)
        self._recording = not greedy
        if self._recording:
            # The state of each step is stored, so that replayed sequences can start from it
            deters = state.deter.half().cpu().numpy()
            stochs = state.stoch.argmax(-1).byte().cpu().numpy()
            for worker, action in enumerate(actions.tolist()):
                latent = Latent(deters[worker], stochs[worker])
                reward, is_first = self.rewards[worker], bool(self.is_first[worker])
                self.replay.add(worker, frames[worker], action, reward, is_first, latent, rarity=self.rarity[worker])
                if self.best_episodes is not None:
                    self.best_episodes.add(worker, frames[worker], action, reward, self.game_rewards[worker], latent)
            self.update_budget += self.num_workers * self._updates_per_step
        self.state = State(state.deter.float(), state.stoch.float())
        self.prev_actions = F.one_hot(actions, self.env_info.num_actions).float()
        self.is_first[:] = False
        return actions.tolist()

    @property
    def _updates_per_step(self) -> float:
        return self.config.train_ratio / (self.config.batch_size * self.config.batch_length)

    def observe(self, outcome: StepOutcome) -> None:
        self.rewards = outcome.rewards.astype(np.float32)
        if outcome.game_rewards is not None:
            self.game_rewards = outcome.game_rewards.astype(np.float32)
        if outcome.rarity is not None:
            self.rarity = outcome.rarity.astype(np.float32)
        for worker in map(int, np.flatnonzero(outcome.dones)):
            if self._recording:
                assert outcome.final_observations is not None
                frame = np.asarray(outcome.final_observations[worker])[-1]
                terminal = bool(outcome.terminated[worker])
                reward, rarity = self.rewards[worker], self.rarity[worker]
                self.replay.add(
                    worker, frame, 0, reward, False, None, is_last=True, is_terminal=terminal, rarity=rarity
                )
                self.update_budget += self._updates_per_step
                if self.best_episodes is not None:
                    reward, game_reward = self.rewards[worker], self.game_rewards[worker]
                    group = outcome.levels[worker]
                    if outcome.start_powerups is not None:
                        group += f" {Powerup(int(outcome.start_powerups[worker])).name.lower()}"
                    self.best_episodes.finish(worker, frame, reward, game_reward, terminal, group)
            self.rewards[worker] = 0.0
            self.game_rewards[worker] = 0.0
            self.rarity[worker] = 1.0
            self.is_first[worker] = True

    def train_while_playing(self) -> None:
        """The updates owed to the steps so far, see DreamerAgentConfig.train_ratio."""
        config = self.config
        if len(self.replay) < config.batch_size * config.batch_length or not self.replay.can_sample():
            return
        while self.update_budget >= 1:
            self.update_budget -= 1
            for name, value in self._train_step(self._sample()).items():
                self._metrics[name].append(value)

    def update(self) -> dict[str, float]:
        self.train_while_playing()
        if not self._metrics:
            return {}
        # Copied from the GPU once, so that the updates do not wait for it
        means = torch.stack([torch.stack(values).float().mean() for values in self._metrics.values()]).tolist()
        metrics = {**dict(zip(self._metrics, means, strict=True)), "progress/updates": float(self.updates)}
        self._metrics.clear()
        return metrics

    def _sample(self) -> dict[str, np.ndarray]:
        """
        A batch of replayed sequences, a share of them rare ones (see DreamerAgentConfig.rare_fraction), and the last
        ones from the best episodes once there are some. Then all have "returns" for self-imitation, zero for the
        replayed ones.
        """
        batch_size, rare_fraction = self.config.batch_size, self.config.rare_fraction
        num_best = round(batch_size * self.config.sil_fraction)
        if self.best_episodes is None or not len(self.best_episodes) or not num_best:
            return self.replay.sample(batch_size, num_rare=round(batch_size * rare_fraction))
        num_replayed = batch_size - num_best
        replayed = self.replay.sample(num_replayed, num_rare=round(num_replayed * rare_fraction))
        best = self.best_episodes.sample(num_best)
        replayed["returns"] = np.zeros(replayed["rewards"].shape, np.float32)
        return {name: np.concatenate([replayed[name], best[name]]) for name in best}

    def _train_step(self, batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """:return: The metrics of the update, still on the device"""
        config, model = self.config, self.model
        data = {name: torch.as_tensor(value, device=self.device) for name, value in batch.items()}
        actions = self._onehot(data["actions"])
        start, start_action = self._start(data)
        # The action that led to each frame, the first one's taken in the step before the sequence
        prev_actions = torch.cat([start_action.unsqueeze(1), actions[:, :-1]], 1)
        rewards, is_first = data["rewards"], data["is_first"]
        is_last, is_terminal = data["is_last"].float(), data["is_terminal"].float()
        batch_size, length = rewards.shape
        losses: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor] = {}

        # World model
        posterior = model.rssm.observe(start, prev_actions, model.encoder(data["frames"]), is_first)
        losses["dynamics"], losses["representation"] = model.rssm.kl_losses(posterior)
        features = posterior.features
        reconstruction = model.decoder(posterior).float()
        target = data["frames"].float() / 255
        losses["reconstruction"] = (reconstruction - target).square().sum((-3, -2, -1))
        losses["reward"] = model.reward(features).loss(rewards)
        continuation = (1 - is_terminal) * config.discount
        losses["continuation"] = model.continuation(features).loss(continuation)
        if model.explorer is not None:
            # Each head predicts the next stochastic state, except across the start of an episode and from the final
            # frame, whose action is a placeholder
            inputs = torch.cat([features[:, :-1].detach().float(), actions[:, :-1]], -1)
            next_stoch = posterior.stoch[:, 1:].flatten(2).detach().float()
            valid = (1 - is_first[:, 1:].float()) * (1 - is_last[:, :-1])
            losses["disagreement"] = valid * (model.explorer(inputs) - next_stoch).square().mean((0, -1))

        # Actor and critics on imagined rollouts from every replayed state
        imagined, imagined_actions = self.imagine(posterior.detach().map(lambda x: x.flatten(0, 1)))
        imagined_features = imagined.features
        with torch.no_grad():
            imagined_rewards = model.reward(imagined_features).mean()
            imagined_continuation = model.continuation(imagined_features).prob()
            # Weight of each imagined step: the probability that the episode is still running
            weight = torch.cumprod(imagined_continuation, 1)[:, :-1]
            returns, values, slow_values = self._imagined_returns(
                model.critic, self.slow_critic, imagined_features, imagined_rewards, imagined_continuation
            )
            scale = self._update_return_range("reward", returns)
            advantages = (returns - values[:, :-1]) / scale
            metrics.update({"behavior/imagined_return": returns.mean(), "behavior/return_scale": scale})
            if model.explorer is not None:
                assert model.exploration_critic is not None and self.slow_exploration_critic is not None
                # Rewarded on reaching the state after the action the heads disagree about
                inputs = torch.cat([imagined_features[:, :-1].float(), self._onehot(imagined_actions[:, :-1])], -1)
                novelty = F.pad(model.explorer(inputs).std(0).mean(-1), (1, 0))
                exploration_returns, exploration_values, exploration_slow_values = self._imagined_returns(
                    model.exploration_critic,
                    self.slow_exploration_critic,
                    imagined_features,
                    novelty,
                    imagined_continuation,
                )
                # The disagreement is small but never sparse, so its returns are normalized whatever their range
                exploration_scale = self._update_return_range("exploration", exploration_returns, minimum=1e-3)
                advantages += (
                    config.exploration_weight * (exploration_returns - exploration_values[:, :-1]) / exploration_scale
                )
                metrics["exploration/plan2explore_reward"] = novelty[:, 1:].mean()
        logits = model.actor.logits(imagined_features[:, :-1]).float()
        if config.loss_scales["cmpo"] > 0:
            # Muesli's CMPO regularizer: the actor moves towards its policy reweighted by the advantages of all actions
            # of each replayed state, valued one imagined step ahead, so that it also learns of better actions that it
            # rarely samples, e.g. pressing down on a pipe. The prior is the current policy instead of Muesli's target
            # network. All of this is in the actor's own policy, before the uniform mixture with which actions are
            # sampled. Its target is label smoothed, so that it pulls up even actions the actor had ruled out. The
            # advantages combine the game's and Plan2Explore's like those of the policy gradient, so that the actor
            # also tries actions whose outcome the world model is unsure about, which it has never seen taken there.
            own_log_probs = torch.log_softmax(logits[:, 0], -1)
            with torch.no_grad():
                task_values, exploration_action_values = self._action_values(imagined.map(lambda x: x[:, 0]))
                action_advantages = (task_values - values[:, :1]) / scale
                if exploration_action_values is not None:
                    action_advantages += (
                        config.exploration_weight
                        * (exploration_action_values - exploration_values[:, :1])
                        / exploration_scale
                    )
                targets = cmpo_targets(own_log_probs, action_advantages)
                changed = targets.argmax(-1) != own_log_probs.argmax(-1)
                metrics["behavior/cmpo_changes"] = changed.float().mean()
                smoothing = config.cmpo_label_smoothing
                targets = (1 - smoothing) * targets + smoothing / targets.shape[-1]
            acting = 1 - is_last.flatten()
            losses["cmpo"] = -acting * (targets * own_log_probs).sum(-1)
        policy = Categorical(logits, config.unimix)
        log_probs = policy.log_prob(imagined_actions[:, :-1])
        entropy = policy.entropy()
        losses["policy"] = -weight * (log_probs * advantages + config.entropy_weight * entropy)
        critic = model.critic(imagined_features[:, :-1])
        losses["value"] = weight * (
            critic.loss(returns) + config.slow_regularization * critic.loss(slow_values[:, :-1])
        )
        if model.exploration_critic is not None:
            critic = model.exploration_critic(imagined_features[:, :-1])
            losses["exploration_value"] = weight * (
                critic.loss(exploration_returns)
                + config.slow_regularization * critic.loss(exploration_slow_values[:, :-1])
            )

        # Critic on the replayed sequences, bootstrapped from the imagined returns
        replay_critic = model.critic(features[:, :-1])
        with torch.no_grad():
            replay_returns = lambda_return(
                is_last,
                is_terminal,
                rewards,
                returns[:, 0].unflatten(0, (batch_size, length)),
                config.discount,
                config.lam,
            )
            slow_replay_values = self.slow_critic(features[:, :-1]).mean()
        losses["replay_value"] = (1 - is_last[:, :-1]) * (
            replay_critic.loss(replay_returns) + config.slow_regularization * replay_critic.loss(slow_replay_values)
        )

        if "returns" in data:
            # Self-imitation (Oh et al., 2018): the actions of the best episodes, the last ones of the batch, where
            # their return beat the critic's expectation, except the placeholder action of the final frame
            num_best = round(batch_size * config.sil_fraction)
            sil_features = features[-num_best:].detach()
            with torch.no_grad():
                sil_advantages = (data["returns"][-num_best:] - model.critic(sil_features).mean()).clamp(min=0) / scale
            sil_log_probs = model.actor(sil_features).log_prob(data["actions"][-num_best:])
            losses["self_imitation"] = -(1 - is_last[-num_best:]) * sil_advantages * sil_log_probs

        loss = sum(config.loss_scales[name] * value.float().mean() for name, value in losses.items())
        assert isinstance(loss, torch.Tensor)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self._update_slow_critics()
        self.updates += 1
        world_model = ("reconstruction", "reward", "dynamics")
        return {
            **{f"world_model/{name}": losses[name].detach().mean() for name in world_model},
            **metrics,
            "behavior/entropy": entropy.detach().mean(),
        }

    def _imagined_returns(
        self,
        critic: nn.Module,
        slow_critic: nn.Module,
        features: torch.Tensor,
        rewards: torch.Tensor,
        continuation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """:return: The lambda returns of the imagined rewards, and the critic's and slow critic's values"""
        values = critic(features).mean()
        returns = lambda_return(torch.zeros_like(values), 1 - continuation, rewards, values, 1.0, self.config.lam)
        return returns, values, slow_critic(features).mean()

    def save(self, iteration: int) -> None:
        """Also stores a preview of the world model's predictions on replayed sequences with the checkpoint."""
        super().save(iteration)
        if not self.replay.can_sample():
            return
        batch = self.replay.sample(PREVIEW_SEQUENCES, online=False)
        context = self.config.batch_length // 4
        predicted = self.predict(batch, context)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.gif"
            save_gif(render_preview(batch["frames"], predicted, context), path)
            mlflow.log_artifact(str(path), checkpoint_name(iteration))

    @torch.no_grad()
    def predict(self, batch: dict[str, np.ndarray], context: int, dream: bool = False) -> npt.NDArray[np.uint8]:
        """
        Frames the world model predicts for sequences like those of Replay.sample. Like in training, it starts from
        the state the agent had before each sequence and sees its first `context` frames, which it reconstructs. Then
        it predicts the rest from the given actions alone ("open loop"), or, if `dream`, from actions of its own
        policy, without the true frames after the context.
        """
        data = {name: torch.as_tensor(value, device=self.device) for name, value in batch.items()}
        tokens = self.model.encoder(data["frames"][:, :context])
        actions = data["actions"]
        state, prev_action = self._start(data)
        states = []
        for t in range(actions.shape[1]):
            if t < context:
                state = self.model.rssm.observe_step(state, prev_action, tokens[:, t], data["is_first"][:, t])
                prev_action = self._onehot(actions[:, t])
            else:
                action = self.model.actor(state.features).sample() if dream else actions[:, t - 1]
                state = self.model.rssm.imagine_step(state, self._onehot(action))
            states.append(State(state.deter, state.stoch))
        predicted = self.model.decoder(State.stack(states, 1))
        return (predicted * 255).round().byte().cpu().numpy()

    def _onehot(self, actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(actions, self.env_info.num_actions).float()

    def _start(self, data: dict[str, torch.Tensor]) -> tuple[State, torch.Tensor]:
        """:return: The stored state of the step before each sequence and the action taken in it, see Replay.sample"""
        stoch = F.one_hot(data["start_stoch"].long(), self.config.classes).float()
        return State(data["start_deter"].float(), stoch), self._onehot(data["start_action"])

    @torch.no_grad()
    def _imagine(self, start: State) -> tuple[State, torch.Tensor]:
        """:return: The start states followed by `imagination_length` imagined ones, and the action in each"""
        states, actions = [start], []
        for _ in range(self.config.imagination_length):
            action = self.model.actor(states[-1].features).sample()
            actions.append(action)
            onehot = F.one_hot(action, self.env_info.num_actions).float()
            states.append(self.model.rssm.imagine_step(states[-1], onehot))
        actions.append(self.model.actor(states[-1].features).sample())
        return State.stack([State(s.deter, s.stoch) for s in states], 1), torch.stack(actions, 1)

    def _action_values(
        self, start: State
    ) -> tuple[Float[torch.Tensor, "batch actions"], Float[torch.Tensor, "batch actions"] | None]:
        """
        Of each action in each state: the predicted reward of the step plus the value of the state it leads to, and
        likewise for Plan2Explore, whose reward is the disagreement about that step, if it is enabled.
        """
        model, n = self.model, self.env_info.num_actions
        states = start.map(lambda x: x.repeat_interleave(n, 0))
        actions = torch.eye(n, device=self.device).repeat(len(start.deter), 1)
        features = model.rssm.imagine_step(states, actions).features
        continuation = model.continuation(features).prob()
        values = model.reward(features).mean() + continuation * model.critic(features).mean()
        if model.explorer is None:
            return values.unflatten(0, (-1, n)), None
        assert model.exploration_critic is not None
        novelty = model.explorer(torch.cat([states.features.float(), actions], -1)).std(0).mean(-1)
        exploration = novelty + continuation * model.exploration_critic(features).mean()
        return values.unflatten(0, (-1, n)), exploration.unflatten(0, (-1, n))

    def _update_return_range(self, name: str, returns: torch.Tensor, minimum: float = 1.0) -> torch.Tensor:
        """
        :param minimum: Smallest scale, so that small differences between returns are not magnified, e.g. while
            rewards are still rare
        :return: The scale of the returns: the range between their running percentiles, but at least `minimum`
        """
        percentiles = torch.quantile(returns.float().flatten(), self.return_quantiles)
        rate = self.config.return_norm_rate
        self.return_ranges[name] = (1 - rate) * self.return_ranges[name] + rate * percentiles
        return torch.clamp(self.return_ranges[name][1] - self.return_ranges[name][0], min=minimum)

    @torch.no_grad()
    def _update_slow_critics(self) -> None:
        pairs = [(self.slow_critic, self.model.critic), (self.slow_exploration_critic, self.model.exploration_critic)]
        for slow_critic, critic in pairs:
            if slow_critic is None or critic is None:
                continue
            for slow, current in zip(slow_critic.parameters(), critic.parameters(), strict=True):
                slow.lerp_(current, self.config.slow_critic_rate)
