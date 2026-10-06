import logging
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
import torchinfo
from jaxtyping import Float, Int64, Shaped
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, TensorDataset

from config.agent import PPOAgentConfig
from environment import EnvironmentInfo
from models import get_model

from .advantage import GaeEstimator
from .agent import Agent, StepOutcome
from .experience_buffer import ExperienceBuffer
from .self_imitation import SelfImitationMemory

log = logging.getLogger(__name__)


class PPOAgent(Agent):
    config: PPOAgentConfig

    def __init__(self, config: PPOAgentConfig, env_info: EnvironmentInfo) -> None:
        super().__init__(config, env_info)

        self.gae_estimator = GaeEstimator(
            gamma=self.config.gamma,
            tau=self.config.tau,
        )
        self.sil_loss_weight = self.config.sil_loss_weight
        self.self_imitation = (
            SelfImitationMemory(
                self.num_workers, self.config.gamma, self.config.sil_episodes_per_level, self.env_info.stack_frames
            )
            if self.sil_loss_weight > 0
            else None
        )
        self.critic_loss_weight = self.config.critic_loss_weight
        self.initial_entropy_loss_weight = self.config.initial_entropy_loss_weight
        self.final_entropy_loss_weight = self.config.final_entropy_loss_weight
        self.normalize_advantages = self.config.normalize_advantages
        lr = self.config.learning_rate.learning_rate

        self.batch_size = self.config.batch_size
        self.grad_clip_norm = self.config.grad_clip_norm
        self.clip_value = self.config.clip_value
        self.clip_param = self.config.clip_param

        self.epochs_per_update = self.config.epochs_per_update
        self.total_updates = self.config.total_updates

        self.cpu = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(self.device)
            # The input size is fixed, so the fastest convolution algorithms can be chosen once
            torch.backends.cudnn.benchmark = True
            log.info(f"Using GPU {device_name}")
        else:
            self.device = self.cpu
            log.info("Using CPU")

        self.experience_buffer = ExperienceBuffer(
            self.num_workers, size=self.config.exp_buffer_size, device=self.device
        )

        self.actor_critic = get_model(config=self.config.model, env_info=self.env_info).to(self.device)

        self.sampling_strategy = self.config.sampling_strategy

        self.opt = torch.optim.Adam(
            params=self.actor_critic.parameters(),
            lr=lr,
            eps=self.config.adam_eps,
        )

        self.scheduler = self.config.learning_rate.create_scheduler(self.opt)

        self.update_step = 0

    @property
    def _checkpointed(self) -> dict[str, torch.nn.Module | torch.optim.Optimizer | LRScheduler]:
        return {"actor_critic": self.actor_critic, "optimizer": self.opt, "scheduler": self.scheduler}

    def state_dict(self) -> dict:
        return {
            **{name: module.state_dict() for name, module in self._checkpointed.items()},
            "update_step": self.update_step,
            "sampling_strategy": self.sampling_strategy.state_dict(),
            "self_imitation": self.self_imitation.state_dict() if self.self_imitation is not None else {},
        }

    def load_state_dict(self, state: dict) -> None:
        for name, module in self._checkpointed.items():
            module.load_state_dict(state[name])
        self.update_step = state["update_step"]
        self.sampling_strategy.load_state_dict(state["sampling_strategy"])
        if self.self_imitation is not None:
            self.self_imitation.load_state_dict(state["self_imitation"])

    def _compute_probs_values(
        self,
    ) -> tuple[Float[torch.Tensor, "worker action"], Float[torch.Tensor, " worker"]]:
        state = self.experience_buffer.get_last_states()
        action = self.experience_buffer.get_last_actions()
        logits, values = self.actor_critic.forward(state, action)

        return torch.softmax(logits, dim=-1), values

    def next_actions(self, train: bool = True) -> tuple[list[int], list[float]]:
        with torch.no_grad():
            probs, values = self._compute_probs_values()
            actions = self.sampling_strategy.sample_action(probs)

            # Log-probs of the policy that actually picked the actions, so that the PPO ratio is a correct
            # importance weight also for sampling strategies that deviate from the policy
            behavior_probs = self.sampling_strategy.behavior_probs(probs)
            tiny = torch.finfo(behavior_probs.dtype).tiny
            log_probs = torch.log(behavior_probs.clamp(min=tiny)).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        if self.self_imitation is not None:
            self.self_imitation.record_actions(
                self.experience_buffer.get_last_states(), self.experience_buffer.get_last_actions(), actions
            )
        self.experience_buffer.buffer_values(values)
        self.experience_buffer.buffer_log_probs(log_probs)
        self.experience_buffer.buffer_actions(actions)
        return [int(action) for action in actions.to(self.cpu, copy=True)], [
            float(lp) for lp in log_probs.to(self.cpu, copy=True)
        ]

    def describe(self) -> str:
        input_size = (self.batch_size, self.env_info.total_channel_dim, self.env_info.height, self.env_info.width)
        return str(
            torchinfo.summary(
                self.actor_critic,
                input_size=[input_size, (self.batch_size,)],
                dtypes=[torch.float32, torch.int64],
                col_names=["output_size", "kernel_size", "num_params", "params_percent"],
                verbose=0,
            )
        )

    def give_reward(self, outcome: StepOutcome) -> None:
        rewards = torch.tensor(outcome.rewards, dtype=torch.float32, device=self.device)
        if self.self_imitation is not None:
            # The game's reward only: the agent should imitate what was good, not what was new
            self.self_imitation.record_rewards(rewards, outcome.dones.tolist(), outcome.levels)
        if outcome.bonus is not None:
            rewards += torch.tensor(outcome.bonus, dtype=torch.float32, device=self.device)

        # A truncated episode did not end in the environment, so its return continues beyond the final state.
        # Since the final state is replaced by the next episode's start, fold its discounted value into the reward.
        truncated_workers = np.flatnonzero(np.logical_and(outcome.truncated, np.logical_not(outcome.terminated)))
        if len(truncated_workers) > 0:
            final_states = outcome.final_states
            assert final_states is not None, "Final states are required to bootstrap truncated episodes."
            worker_idx = torch.as_tensor(truncated_workers, device=self.device)
            with torch.no_grad():
                _, final_values = self.actor_critic.forward(
                    self._state_to_tensor(np.stack([final_states[w] for w in truncated_workers])),
                    self.experience_buffer.get_last_actions()[worker_idx],
                )
            rewards[worker_idx] += self.gae_estimator.gamma * final_values

        self.experience_buffer.buffer_rewards(rewards)
        self.experience_buffer.buffer_dones(torch.tensor(outcome.dones, device=self.device))

    def save(self, save_iter: int) -> None:
        for name, module in self._checkpointed.items():
            mlflow.pytorch.log_state_dict(module.state_dict(), f"{name}-{save_iter:04d}")

    def load(self, run_id: str, load_iter: int) -> None:
        # mlflow.pytorch.load_state_dict would leave each download behind in a temporary directory
        with tempfile.TemporaryDirectory() as directory:
            for name, module in self._checkpointed.items():
                path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/{name}-{load_iter:04d}", dst_path=directory)
                module.load_state_dict(torch.load(Path(path) / "state_dict.pth", weights_only=True))

    def _get_ext_values(self, v: Float[torch.Tensor, " worker"]) -> Float[torch.Tensor, "time worker"]:
        values = self.experience_buffer.get_value_buffer()
        return torch.concat([values, v.unsqueeze(0)])

    @staticmethod
    def _reshape_tensor(tensor: Shaped[torch.Tensor, " buffer worker *rest"]) -> Shaped[torch.Tensor, " batch *rest"]:
        total_samples = tensor.size(0) * tensor.size(1)
        return tensor.reshape((total_samples, *tensor.shape[2:]))

    def _get_advantages_returns(
        self, next_states: np.ndarray
    ) -> tuple[Float[torch.Tensor, " batch"], Float[torch.Tensor, " batch"]]:
        with torch.no_grad():
            # Bootstrap with the value of the state reached after the last buffered step
            _, v = self.actor_critic.forward(
                self._state_to_tensor(next_states), self.experience_buffer.get_last_actions()
            )
            values_ext = self._get_ext_values(v)
            advantages, returns = self.gae_estimator.get_advantage_returns(
                self.experience_buffer.get_reward_buffer(),
                values_ext,
                self.experience_buffer.get_dones_buffer(),
            )
            return self._reshape_tensor(advantages), self._reshape_tensor(returns)

    def _get_reshaped_buffers(
        self,
    ) -> tuple[
        Float[torch.Tensor, "batch channel height width"],
        Float[torch.Tensor, " batch"],
        Float[torch.Tensor, " batch"],
        Int64[torch.Tensor, " batch"],
        Int64[torch.Tensor, " batch"],
    ]:
        states = self.experience_buffer.get_state_buffer()
        log_probs = self.experience_buffer.get_log_prob_buffer()
        values = self.experience_buffer.get_value_buffer()
        actions = self.experience_buffer.get_action_buffer()
        prev_actions = self.experience_buffer.get_prev_action_buffer()

        states = self._reshape_tensor(states)
        log_probs = self._reshape_tensor(log_probs)
        values = self._reshape_tensor(values)
        actions = self._reshape_tensor(actions)
        prev_actions = self._reshape_tensor(prev_actions)

        return states, log_probs, values, actions, prev_actions

    def _create_dataset_from_buffers(self, next_states: np.ndarray) -> TensorDataset:
        advantages, returns = self._get_advantages_returns(next_states)
        states, log_probs, values, actions, prev_actions = self._get_reshaped_buffers()
        return TensorDataset(states, log_probs, values, actions, prev_actions, advantages, returns)

    def _calculate_actor_loss(
        self,
        new_logits: Float[torch.Tensor, "minibatch action"],
        b_actions: Int64[torch.Tensor, " minibatch"],
        b_log_probs: Float[torch.Tensor, " minibatch"],
        b_advantages: Float[torch.Tensor, " minibatch"],
    ) -> tuple[Float[torch.Tensor, ""], torch.distributions.Categorical]:
        new_dist = torch.distributions.Categorical(logits=new_logits)

        if self.normalize_advantages and b_advantages.numel() > 1:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        new_log_probs = new_dist.log_prob(b_actions)

        ratio = torch.exp(new_log_probs - b_log_probs)
        act_loss_1 = b_advantages * ratio
        act_loss_2 = b_advantages * torch.clip(ratio, min=1 - self.clip_param, max=1 + self.clip_param)

        act_loss = -torch.mean(torch.minimum(act_loss_1, act_loss_2))
        return act_loss, new_dist

    def _calculate_critic_loss(
        self,
        new_values: Float[torch.Tensor, " minibatch"],
        b_returns: Float[torch.Tensor, " minibatch"],
        b_values: Float[torch.Tensor, " minibatch"],
    ) -> Float[torch.Tensor, ""]:
        crit_loss = torch.mul(torch.nn.functional.mse_loss(b_returns, new_values, reduction="none"), 0.5)
        if self.clip_value:
            v_clip = torch.clip(
                new_values,
                min=b_values - self.clip_param,
                max=b_values + self.clip_param,
            )
            crit_loss_2 = torch.mul(torch.nn.functional.mse_loss(b_returns, v_clip, reduction="none"), 0.5)
            crit_loss = torch.maximum(crit_loss, crit_loss_2)
        crit_loss = torch.mean(crit_loss)

        return crit_loss

    def _optimizer_step(self, loss: Float[torch.Tensor, ""]) -> None:
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.actor_critic.parameters(), max_norm=self.grad_clip_norm)
        self.opt.step()

    def _current_entropy_loss_weight(self) -> float:
        # Linear decay from the initial to the final weight over the course of training
        progress = min(self.update_step / self.total_updates, 1.0)
        return self.initial_entropy_loss_weight + progress * (
            self.final_entropy_loss_weight - self.initial_entropy_loss_weight
        )

    def update(self, next_states: np.ndarray, episode_reward: float | None = None) -> dict[str, float]:
        losses = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
        }

        dataset = self._create_dataset_from_buffers(next_states)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for e in range(1, self.epochs_per_update + 1):
            total_epoch_loss = 0
            total_epoch_batches = 0
            for (
                b_states,
                b_log_probs,
                b_values,
                b_actions,
                b_prev_actions,
                b_advantages,
                b_returns,
            ) in dataloader:
                logits, v = self.actor_critic.forward(b_states, b_prev_actions)

                # Actor loss
                act_loss, action_dist = self._calculate_actor_loss(logits, b_actions, b_log_probs, b_advantages)
                losses["actor_loss"] += act_loss.item()

                # Critic loss
                crit_loss = self._calculate_critic_loss(v, b_returns, b_values)
                losses["critic_loss"] += crit_loss.item()

                # Entropy loss
                entropy = torch.mean(action_dist.entropy())
                losses["entropy_loss"] += entropy.item()

                # Total
                loss = torch.add(
                    act_loss,
                    torch.sub(
                        torch.mul(self.critic_loss_weight, crit_loss),
                        torch.mul(self._current_entropy_loss_weight(), entropy),
                    ),
                )
                losses["total_loss"] += loss.item()
                total_epoch_loss += loss.item()
                total_epoch_batches += 1

                # Optimizer
                self._optimizer_step(loss)

            avg_loss = total_epoch_loss / total_epoch_batches
            log.debug(f"Finished epoch {e} with avg loss {avg_loss}")

        for key in losses:
            losses[key] /= len(dataloader) * self.epochs_per_update
        log.debug(f"Update finished. Losses: {losses}")
        metrics = {
            **losses,
            "entropy_loss_weight": self._current_entropy_loss_weight(),
            "learning_rate": float(self.scheduler.get_last_lr()[0]),
        }
        if self.self_imitation is not None:
            metrics.update(self._self_imitation_update())
        self.scheduler.step()
        if episode_reward is not None:
            self.sampling_strategy.update(episode_reward)
        metrics.update(self.sampling_strategy.get_metrics())

        self.update_step += 1
        return metrics

    def _self_imitation_update(self) -> dict[str, float]:
        """
        Imitate the stored best episodes where their return exceeds the current value estimate, weighted by how much
        it does, and raise the value estimate towards that return.
        """
        assert self.self_imitation is not None
        losses, positive_shares = [], []
        for _ in range(self.config.sil_updates):
            batch = self.self_imitation.sample(self.config.sil_batch_size)
            if batch is None:
                break
            states, prev_actions, actions, returns = (tensor.to(self.device) for tensor in batch)
            logits, values = self.actor_critic.forward(states.to(torch.float32), prev_actions)
            advantages = torch.clamp(returns - values, min=0)
            weights = advantages.detach()
            positive_shares.append(float((weights > 0).float().mean()))
            if weights.sum() <= 0:
                continue
            log_probs = torch.log_softmax(logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            policy_loss = -(weights * log_probs).sum() / weights.sum()
            value_loss = 0.5 * torch.mean(advantages**2)
            loss = self.sil_loss_weight * policy_loss + self.config.sil_value_loss_weight * value_loss
            self._optimizer_step(loss)
            losses.append(loss.item())
        metrics = self.self_imitation.get_metrics()
        if positive_shares:
            metrics["sil_positive_share"] = sum(positive_shares) / len(positive_shares)
        if losses:
            metrics["sil_loss"] = sum(losses) / len(losses)
        return metrics

    def _state_to_tensor(self, state: np.ndarray) -> Float[torch.Tensor, "worker channel height width"]:
        # State has shape (<num_workers>, <num_stack_frames>, <height>, <width>, <num_channels>)
        # Convert to shape (<num_workers>, <num_stack_frames> * <num_channels>, <height>, <width>)
        state_tensor: Shaped[torch.Tensor, "worker stack_frame height width color_channel"] = torch.tensor(
            np.array(state), device=self.device
        )
        state_tensor = torch.concat(torch.unbind(state_tensor, dim=1), dim=-1)
        state_tensor = torch.moveaxis(state_tensor, -1, 1)
        return state_tensor.to(torch.float32)

    def feed_observation(self, state: np.ndarray) -> None:
        self.experience_buffer.buffer_states(self._state_to_tensor(state))
