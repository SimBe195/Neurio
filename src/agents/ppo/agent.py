import numpy as np
import torch
import torchinfo
from jaxtyping import Float, Int64, Shaped
from torch.utils.data import DataLoader, TensorDataset

from agents.agent import Agent, StepOutcome, get_device
from config.agent import PPOAgentConfig
from environment import EnvironmentInfo

from .actor_critic import ActorCritic
from .advantage import GaeEstimator
from .experience_buffer import ExperienceBuffer
from .reward_scale import RewardScale
from .self_imitation import SelfImitationMemory


class PPOAgent(Agent):
    """
    Proximal policy optimization (Schulman et al., 2017) with generalized advantage estimation, self-imitation of the
    agent's own best episodes and rewards scaled by the spread of the return. Each update learns from the steps since
    the previous one.
    """

    config: PPOAgentConfig
    # It learns on-policy, from the probabilities of its own choices
    takes_untried_actions = False

    def __init__(self, config: PPOAgentConfig, env_info: EnvironmentInfo) -> None:
        super().__init__(config, env_info)
        self.device = get_device()
        self.gae_estimator = GaeEstimator(gamma=config.gamma, tau=config.tau)
        self.self_imitation = (
            SelfImitationMemory(self.num_workers, config.gamma, config.sil_episodes_per_level, env_info.stack_frames)
            if config.sil_loss_weight > 0
            else None
        )
        self.reward_scale = RewardScale(self.num_workers, config.gamma, self.device)
        self.experience_buffer = ExperienceBuffer()
        self.actor_critic = ActorCritic(config.model, env_info).to(self.device)
        self.opt = torch.optim.Adam(
            self.actor_critic.parameters(), lr=config.learning_rate.learning_rate, eps=config.adam_eps
        )
        self.scheduler = config.learning_rate.create_scheduler(self.opt)
        self.update_step = 0

        # The network sees the previous action of each worker, none at the start of an episode
        self.prev_actions = torch.zeros(self.num_workers, dtype=torch.int64, device=self.device)
        # States after the last observed step, from which the returns of the buffered steps are bootstrapped
        self.next_states: Float[torch.Tensor, "worker channel height width"] | None = None
        self._recording = False

    @property
    def networks(self) -> dict[str, torch.nn.Module]:
        return {"actor_critic": self.actor_critic}

    def state_dict(self) -> dict:
        return {
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "update_step": self.update_step,
            "self_imitation": self.self_imitation.state_dict() if self.self_imitation is not None else {},
            "reward_scale": self.reward_scale.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor_critic.load_state_dict(state["actor_critic"])
        self.opt.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.update_step = state["update_step"]
        if self.self_imitation is not None:
            self.self_imitation.load_state_dict(state["self_imitation"])
        self.reward_scale.load_state_dict(state["reward_scale"])

    def describe(self) -> str:
        batch_size = self.config.batch_size
        input_size = (batch_size, self.env_info.total_channel_dim, self.env_info.height, self.env_info.width)
        return str(
            torchinfo.summary(
                self.actor_critic,
                input_size=[input_size, (batch_size,)],
                dtypes=[torch.float32, torch.int64],
                col_names=["output_size", "kernel_size", "num_params", "params_percent"],
                verbose=0,
            )
        )

    def act(self, observations: np.ndarray, greedy: bool = False, untried: np.ndarray | None = None) -> list[int]:
        assert untried is None, "PPO learns on-policy, from its own actions only"
        states = self._state_to_tensor(observations)
        with torch.no_grad():
            logits, values = self.actor_critic(states, self.prev_actions)
        if greedy:
            actions = logits.argmax(-1)
        else:
            distribution = torch.distributions.Categorical(logits=logits.float())
            actions = distribution.sample()
            self.experience_buffer.add(
                states=states,
                prev_actions=self.prev_actions,
                actions=actions,
                log_probs=distribution.log_prob(actions),
                values=values,
            )
            if self.self_imitation is not None:
                self.self_imitation.record_actions(states, self.prev_actions, actions)
        self._recording = not greedy
        self.prev_actions = actions
        return actions.tolist()

    def observe(self, outcome: StepOutcome) -> None:
        dones = torch.tensor(outcome.dones, device=self.device)
        if self._recording:
            self._record_outcome(outcome, dones)
        self.prev_actions = self.prev_actions * ~dones

    def _record_outcome(self, outcome: StepOutcome, dones: torch.Tensor) -> None:
        rewards = torch.tensor(outcome.rewards, dtype=torch.float32, device=self.device)
        if self.self_imitation is not None:
            assert outcome.game_rewards is not None
            self.self_imitation.record_rewards(
                rewards, outcome.game_rewards.tolist(), outcome.dones.tolist(), outcome.levels
            )
        rewards = self.reward_scale(rewards, dones)

        # A truncated episode did not end in the game, so its return continues beyond the final observation.
        # The next state starts the next episode, so the final state's discounted value is folded into the reward.
        truncated_workers = np.flatnonzero(np.logical_and(outcome.truncated, np.logical_not(outcome.terminated)))
        if len(truncated_workers) > 0:
            final_observations = outcome.final_observations
            assert final_observations is not None, "Final observations are required to bootstrap truncated episodes."
            worker_idx = torch.as_tensor(truncated_workers, device=self.device)
            with torch.no_grad():
                _, final_values = self.actor_critic(
                    self._state_to_tensor(np.stack([final_observations[w] for w in truncated_workers])),
                    self.prev_actions[worker_idx],
                )
            rewards[worker_idx] += self.gae_estimator.gamma * final_values

        self.experience_buffer.add(rewards=rewards, dones=dones.to(torch.int64))
        self.next_states = self._state_to_tensor(outcome.observations)

    def update(self) -> dict[str, float]:
        assert self.next_states is not None and len(self.experience_buffer) > 0, "No experience to learn from."
        entropies = []
        dataloader = DataLoader(self._create_dataset(), batch_size=self.config.batch_size, shuffle=True)
        for _ in range(self.config.epochs_per_update):
            for states, log_probs, values, actions, prev_actions, advantages, returns in dataloader:
                logits, new_values = self.actor_critic(states, prev_actions)
                actor_loss, distribution = self._actor_loss(logits, actions, log_probs, advantages)
                critic_loss = self._critic_loss(new_values, returns, values)
                entropy = distribution.entropy().mean()
                loss = (
                    actor_loss
                    + self.config.critic_loss_weight * critic_loss
                    - self._current_entropy_loss_weight() * entropy
                )
                self._optimizer_step(loss)
                entropies.append(entropy.detach())
        self.experience_buffer.clear()

        if self.self_imitation is not None:
            self._self_imitation_update()
        self.scheduler.step()
        self.update_step += 1
        return {"behavior/entropy": torch.stack(entropies).mean().item()}

    def _create_dataset(self) -> TensorDataset:
        buffer = self.experience_buffer
        assert self.next_states is not None
        with torch.no_grad():
            # Bootstrap with the value of the states reached after the last buffered step
            _, next_values = self.actor_critic(self.next_states, self.prev_actions)
        advantages, returns = self.gae_estimator.get_advantage_returns(
            buffer.stack("rewards"),
            torch.concat([buffer.stack("values"), next_values.unsqueeze(0)]),
            buffer.stack("dones"),
        )
        tensors = [buffer.stack(name) for name in ("states", "log_probs", "values", "actions", "prev_actions")] + [
            advantages,
            returns,
        ]
        return TensorDataset(*(self._flatten_steps(tensor) for tensor in tensors))

    @staticmethod
    def _flatten_steps(tensor: Shaped[torch.Tensor, "step worker *rest"]) -> Shaped[torch.Tensor, " batch *rest"]:
        return tensor.flatten(0, 1)

    def _actor_loss(
        self,
        logits: Float[torch.Tensor, "minibatch action"],
        actions: Int64[torch.Tensor, " minibatch"],
        old_log_probs: Float[torch.Tensor, " minibatch"],
        advantages: Float[torch.Tensor, " minibatch"],
    ) -> tuple[Float[torch.Tensor, ""], torch.distributions.Categorical]:
        distribution = torch.distributions.Categorical(logits=logits)
        if self.config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = torch.exp(distribution.log_prob(actions) - old_log_probs)
        clip = self.config.clip_param
        loss = -torch.minimum(advantages * ratio, advantages * torch.clip(ratio, 1 - clip, 1 + clip)).mean()
        return loss, distribution

    def _critic_loss(
        self,
        values: Float[torch.Tensor, " minibatch"],
        returns: Float[torch.Tensor, " minibatch"],
        old_values: Float[torch.Tensor, " minibatch"],
    ) -> Float[torch.Tensor, ""]:
        loss = 0.5 * (returns - values) ** 2
        if self.config.clip_value:
            clipped = old_values + torch.clip(values - old_values, -self.config.clip_param, self.config.clip_param)
            loss = torch.maximum(loss, 0.5 * (returns - clipped) ** 2)
        return loss.mean()

    def _optimizer_step(self, loss: Float[torch.Tensor, ""]) -> None:
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=self.config.grad_clip_norm)
        self.opt.step()

    def _current_entropy_loss_weight(self) -> float:
        # Linear decay from the initial to the final weight
        progress = min(self.update_step / self.config.entropy_decay_updates, 1.0)
        initial, final = self.config.initial_entropy_loss_weight, self.config.final_entropy_loss_weight
        return initial + progress * (final - initial)

    def _self_imitation_update(self) -> None:
        """
        Imitate the stored best episodes where their return exceeds the current value estimate, weighted by how much
        it does, and raise the value estimate towards that return.
        """
        assert self.self_imitation is not None
        for _ in range(self.config.sil_updates):
            batch = self.self_imitation.sample(self.config.sil_batch_size)
            if batch is None:
                break
            states, prev_actions, actions, returns = (tensor.to(self.device) for tensor in batch)
            # The memory keeps the game's returns, the critic learns them scaled
            returns = returns / self.reward_scale.std
            logits, values = self.actor_critic(states.to(torch.float32), prev_actions)
            advantages = torch.clamp(returns - values, min=0)
            weights = advantages.detach()
            if weights.sum() <= 0:
                continue
            log_probs = torch.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            policy_loss = -(weights * log_probs).sum() / weights.sum()
            value_loss = 0.5 * torch.mean(advantages**2)
            loss = self.config.sil_loss_weight * policy_loss + self.config.sil_value_loss_weight * value_loss
            self._optimizer_step(loss)

    def _state_to_tensor(self, observations: np.ndarray) -> Float[torch.Tensor, "worker channel height width"]:
        """Stacks the frames of each observation, of shape (worker, frame, height, width, color), as channels."""
        tensor = torch.as_tensor(np.asarray(observations), device=self.device)
        return tensor.permute(0, 1, 4, 2, 3).flatten(1, 2).to(torch.float32)
