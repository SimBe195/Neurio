import numpy as np
import torch
import torchinfo
from jaxtyping import Float, Int64

from agents.agent import Agent, Play, StepOutcome, get_device
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

    Untried actions (see Agent.act) are off-policy for PPO: it learns from them with its policy's probability of them,
    which its clipping bounds. They are rare, and self-imitation learns from them without that restriction.
    """

    config: PPOAgentConfig

    def __init__(self, config: PPOAgentConfig, env_info: EnvironmentInfo) -> None:
        super().__init__(config, env_info)
        self.device = get_device()
        self.gae_estimator = GaeEstimator(gamma=config.gamma, lam=config.lam)
        self.self_imitation = SelfImitationMemory(
            self.num_workers, config.gamma, config.sil_episodes_per_level, env_info.stack_frames
        )
        self.reward_scale = RewardScale(self.num_workers, config.gamma, self.device)
        self.experience_buffer = ExperienceBuffer()
        self.actor_critic = ActorCritic(config.model, env_info).to(self.device)
        self.opt = torch.optim.Adam(self.actor_critic.parameters(), lr=config.learning_rate, eps=config.adam_eps)
        self.update_step = 0

        # The network sees the previous action of each worker, action 0 (no buttons) at the start of an episode
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
            "update_step": self.update_step,
            "self_imitation": self.self_imitation.state_dict(),
            "reward_scale": self.reward_scale.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor_critic.load_state_dict(state["actor_critic"])
        self.opt.load_state_dict(state["optimizer"])
        self.update_step = state["update_step"]
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

    def _policy_probs(self, logits: Float[torch.Tensor, "*batch action"]) -> Float[torch.Tensor, "*batch action"]:
        """The policy, mixed with the uniform distribution so that no action becomes impossible."""
        probs = torch.softmax(logits, -1)
        return (1 - self.config.unimix) * probs + self.config.unimix / probs.shape[-1]

    def _policy(self, logits: Float[torch.Tensor, "*batch action"]) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(probs=self._policy_probs(logits))

    def act(self, observations: np.ndarray, play: Play = Play.TRAIN, untried: np.ndarray | None = None) -> list[int]:
        states = self._state_to_tensor(observations)
        with torch.no_grad():
            logits, values = self.actor_critic(states, self.prev_actions)
        self._recording = play is Play.TRAIN
        if play is Play.GREEDY:
            self.prev_actions = logits.argmax(-1)
            return self.prev_actions.tolist()
        probs = self._policy_probs(logits)
        distribution = torch.distributions.Categorical(probs=probs)
        actions = distribution.sample()
        if self._recording:
            if untried is not None and untried.any():
                # The untried action by the policy, which only matters for its order: all are tried in turn
                mask = torch.as_tensor(untried, device=self.device)
                has_untried = mask.any(-1, keepdim=True)
                untried_probs = probs.masked_fill(has_untried & ~mask, 0.0)
                untried_actions = torch.distributions.Categorical(probs=untried_probs).sample()
                actions = torch.where(has_untried.squeeze(-1), untried_actions, actions)
            self.experience_buffer.add(
                states=states,
                prev_actions=self.prev_actions,
                actions=actions,
                log_probs=distribution.log_prob(actions),
                values=values,
            )
            self.self_imitation.record_actions(states, self.prev_actions, actions)
        self.prev_actions = actions
        return actions.tolist()

    def observe(self, outcome: StepOutcome) -> None:
        dones = torch.tensor(outcome.dones, device=self.device)
        if self._recording:
            self._record_outcome(outcome, dones)
        self.prev_actions = self.prev_actions * ~dones

    def reset(self) -> None:
        self.prev_actions.zero_()

    def _record_outcome(self, outcome: StepOutcome, dones: torch.Tensor) -> None:
        rewards = torch.tensor(outcome.rewards, dtype=torch.float32, device=self.device)
        # The episodes are imitated for the task's reward, without the novelty bonus, which wears off
        task_rewards = torch.tensor(outcome.task_rewards, dtype=torch.float32, device=self.device)
        game_rewards = outcome.game_rewards.tolist()
        self.self_imitation.record_rewards(task_rewards, game_rewards, outcome.dones.tolist(), outcome.episode_groups)
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
        steps = self._steps()
        num_steps = len(steps[0])
        for _ in range(self.config.epochs_per_update):
            order = torch.randperm(num_steps, device=self.device)
            for indices in order.split(self.config.batch_size):
                states, log_probs, actions, prev_actions, advantages, returns = (tensor[indices] for tensor in steps)
                logits, new_values = self.actor_critic(states, prev_actions)
                actor_loss, distribution = self._actor_loss(logits, actions, log_probs, advantages)
                critic_loss = 0.5 * (returns - new_values).square().mean()
                entropy = distribution.entropy().mean()
                loss = (
                    actor_loss
                    + self.config.critic_loss_weight * critic_loss
                    - self._current_entropy_loss_weight() * entropy
                )
                self._optimizer_step(loss)
                entropies.append(entropy.detach())
        self.experience_buffer.clear()
        self._self_imitation_update()
        self.update_step += 1
        return {"behavior/entropy": torch.stack(entropies).mean().item()}

    def _steps(self) -> list[torch.Tensor]:
        """:return: The states, log-probabilities, actions, previous actions, advantages and returns of all steps"""
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
        tensors = [buffer.stack(name) for name in ("states", "log_probs", "actions", "prev_actions")] + [
            advantages,
            returns,
        ]
        # The steps of all workers
        return [tensor.flatten(0, 1) for tensor in tensors]

    def _actor_loss(
        self,
        logits: Float[torch.Tensor, "minibatch action"],
        actions: Int64[torch.Tensor, " minibatch"],
        old_log_probs: Float[torch.Tensor, " minibatch"],
        advantages: Float[torch.Tensor, " minibatch"],
    ) -> tuple[Float[torch.Tensor, ""], torch.distributions.Categorical]:
        distribution = self._policy(logits)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = torch.exp(distribution.log_prob(actions) - old_log_probs)
        clip = self.config.clip_param
        loss = -torch.minimum(advantages * ratio, advantages * torch.clip(ratio, 1 - clip, 1 + clip)).mean()
        return loss, distribution

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
        for _ in range(self.config.sil_updates):
            batch = self.self_imitation.sample(self.config.sil_batch_size)
            if batch is None:
                break
            states, prev_actions, actions, returns = (tensor.to(self.device) for tensor in batch)
            # The memory keeps unscaled returns of the reward without the novelty bonus, the critic learns scaled ones
            returns = returns / self.reward_scale.std
            logits, values = self.actor_critic(states.to(torch.float32), prev_actions)
            advantages = torch.clamp(returns - values, min=0)
            weights = advantages.detach()
            if weights.sum() <= 0:
                continue
            log_probs = self._policy(logits).log_prob(actions)
            policy_loss = -(weights * log_probs).sum() / weights.sum()
            value_loss = 0.5 * torch.mean(advantages**2)
            loss = self.config.sil_loss_weight * policy_loss + self.config.sil_value_loss_weight * value_loss
            self._optimizer_step(loss)

    def _state_to_tensor(self, observations: np.ndarray) -> Float[torch.Tensor, "worker channel height width"]:
        """Stacks the frames of each observation, of shape (worker, frame, height, width, color), as channels."""
        tensor = torch.as_tensor(np.asarray(observations), device=self.device)
        return tensor.permute(0, 1, 4, 2, 3).flatten(1, 2).to(torch.float32)
