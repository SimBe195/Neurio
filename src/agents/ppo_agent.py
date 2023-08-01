import logging

import mlflow
import torchinfo

from src.config.agent import PPOAgentConfig

log = logging.getLogger(__name__)
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from torch.utils.data import DataLoader, TensorDataset

from src.models import get_model

from .advantage import GaeEstimator
from .agent import Agent
from .experience_buffer import ExperienceBuffer


class PPOAgent(Agent):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.config, PPOAgentConfig)

        self.gae_estimator = GaeEstimator(
            gamma=self.config.gamma,
            tau=self.config.tau,
        )
        self.critic_loss_weight = self.config.critic_loss_weight
        self.max_entropy_loss_weight = self.config.max_entropy_loss_weight
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
            log.info(f"Using GPU {device_name}")
        else:
            self.device = self.cpu
            device_name = torch.cuda.get_device_name(self.cpu)
            log.info(f"Using CPU {device_name}")

        self.experience_buffer = ExperienceBuffer(self.num_workers, device=self.device)

        self.actor_critic = get_model(
            config=self.config.model, env_info=self.env_info
        ).to(self.device)

        mlflow.log_text(
            str(
                torchinfo.summary(
                    self.actor_critic,
                    input_size=[
                        (
                            self.batch_size,
                            self.env_info.total_channel_dim,
                            self.env_info.width,
                            self.env_info.height,
                        ),
                        (self.batch_size,),
                    ],
                    dtypes=[torch.float32, torch.int32],
                    col_names=[
                        "output_size",
                        "kernel_size",
                        "num_params",
                        "params_percent",
                    ],
                )
            ),
            artifact_file="torchinfo_summary.txt",
        )

        self.opt = torch.optim.Adam(
            params=self.actor_critic.parameters(),
            lr=lr,
        )

        self.scheduler = self.config.learning_rate.create_scheduler(self.opt)

        self.update_step = 0

    def get_state_dicts(self) -> Dict[str, dict]:
        return {
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def _compute_probs_values(self) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self.experience_buffer.get_last_states()
        action = self.experience_buffer.get_last_actions()
        probs, values = self.actor_critic.forward(state, action)

        return probs, values

    def next_actions(self, train: bool = True) -> Tuple[List[int], List[float]]:
        with torch.no_grad():
            probs, values = self._compute_probs_values()

            action_dist = torch.distributions.Categorical(probs=probs)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)
        self.experience_buffer.buffer_values(values)
        self.experience_buffer.buffer_log_probs(log_probs)
        self.experience_buffer.buffer_actions(actions)
        return [int(action) for action in actions.to(self.cpu, copy=True)], [
            float(lp) for lp in log_probs.to(self.cpu, copy=True)
        ]

    def give_reward(self, reward: List[float], done: List[bool]) -> None:
        self.experience_buffer.buffer_rewards(torch.tensor(reward, device=self.device))
        self.experience_buffer.buffer_dones(torch.tensor(done, device=self.device))

    def reset(self) -> None:
        self.experience_buffer.reset()

    def save(self, iter: int) -> None:
        for name, module in [
            ("actor_critic", self.actor_critic),
            ("optimizer", self.opt),
            ("scheduler", self.scheduler),
        ]:
            mlflow.pytorch.log_state_dict(module.state_dict(), f"{name}-{iter:04d}")

    def load(self, iter: int) -> None:
        for name, module in [
            ("actor_critic", self.actor_critic),
            ("optimizer", self.opt),
            ("scheduler", self.scheduler),
        ]:
            artifact_uri = mlflow.get_artifact_uri(f"{name}-{iter:04d}")
            state_dict = mlflow.pytorch.load_state_dict(artifact_uri)
            module.load_state_dict(state_dict)

    def _get_ext_values(self, v: torch.Tensor) -> torch.Tensor:
        values = self.experience_buffer.get_value_buffer()
        return torch.concat([values, v.unsqueeze(0)])

    @staticmethod
    def _reshape_tensor(tensor: torch.Tensor) -> torch.Tensor:
        total_samples = tensor.size(0) * tensor.size(1)
        return tensor.reshape((total_samples, *tensor.shape[2:]))

    def _get_advantages_returns(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            _, v = self._compute_probs_values()
            values_ext = self._get_ext_values(v)
            advantages, returns = self.gae_estimator.get_advantage_returns(
                self.experience_buffer.get_reward_buffer(),
                values_ext,
                self.experience_buffer.get_dones_buffer(),
            )
            advantages = self._reshape_tensor(advantages)
            returns = self._reshape_tensor(returns)

            return advantages, returns

    def _get_reshaped_buffers(self) -> Tuple[torch.Tensor, ...]:
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

    def _create_dataset_from_buffers(self) -> TensorDataset:
        advantages, returns = self._get_advantages_returns()
        states, log_probs, values, actions, prev_actions = self._get_reshaped_buffers()
        return TensorDataset(
            states, log_probs, values, actions, prev_actions, advantages, returns
        )

    def _calculate_actor_loss(
        self,
        new_probs: torch.Tensor,
        b_actions: torch.Tensor,
        b_log_probs: torch.Tensor,
        b_advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.distributions.Categorical]:
        new_dist = torch.distributions.Categorical(probs=new_probs)

        new_log_probs = new_dist.log_prob(b_actions)

        ratio = torch.exp(new_log_probs - b_log_probs)
        act_loss_1 = b_advantages * ratio
        act_loss_2 = b_advantages * torch.clip(
            ratio, min=1 - self.clip_param, max=1 + self.clip_param
        )

        act_loss = -torch.mean(torch.minimum(act_loss_1, act_loss_2))
        return act_loss, new_dist

    def _calculate_critic_loss(
        self,
        new_values: torch.Tensor,
        b_returns: torch.Tensor,
        b_values: torch.Tensor,
    ) -> torch.Tensor:
        crit_loss = 0.5 * torch.nn.functional.mse_loss(
            b_returns, new_values, reduce=False
        )
        if self.clip_value:
            v_clip = torch.clip(
                new_values,
                min=b_values - self.clip_param,
                max=b_values + self.clip_param,
            )
            crit_loss_2 = 0.5 * torch.nn.functional.mse_loss(
                b_returns, v_clip, reduce=False
            )
            crit_loss = torch.maximum(crit_loss, crit_loss_2)
        crit_loss = torch.mean(crit_loss)

        return crit_loss

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        self.opt.zero_grad()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad.clip_grad_norm_(
            self.actor_critic.parameters(), max_norm=self.grad_clip_norm
        )
        self.opt.step()

    def _current_entropy_loss_weight(self) -> float:
        if self.update_step <= self.total_updates // 2:
            return (
                self.max_entropy_loss_weight
                * (2 * self.update_step)
                / self.total_updates
            )
        else:
            return (
                self.max_entropy_loss_weight
                * (2 * (self.total_updates - self.update_step))
                / self.total_updates
            )

    def update(self) -> None:
        losses = {
            "actor": 0.0,
            "critic": 0.0,
            "entropy": 0.0,
            "total": 0.0,
        }

        dataset = self._create_dataset_from_buffers()
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
                probs, v = self.actor_critic.forward(b_states, b_prev_actions)

                # Actor loss
                act_loss, action_dist = self._calculate_actor_loss(
                    probs, b_actions, b_log_probs, b_advantages
                )
                losses["actor"] += act_loss.item()

                # Critic loss
                crit_loss = self._calculate_critic_loss(v, b_returns, b_values)
                losses["critic"] += crit_loss.item()

                # Entropy loss
                entropy = torch.mean(action_dist.entropy())
                losses["entropy"] += entropy.item()

                # Total
                loss = (
                    act_loss
                    + self.critic_loss_weight * crit_loss
                    - self._current_entropy_loss_weight() * entropy
                )
                losses["total"] += loss.item()
                total_epoch_loss += loss.item()
                total_epoch_batches += 1

                # Optimizer
                self._optimizer_step(loss)

            avg_loss = total_epoch_loss / total_epoch_batches
            log.debug(f"Finished epoch {e} with avg loss {avg_loss}")

        for key in losses:
            losses[key] /= len(dataloader) * self.epochs_per_update
        log.debug(f"Update finished. Losses: {losses}")
        self.scheduler.step()

        mlflow.log_metrics(losses, self.update_step)
        self.update_step += 1

        self.experience_buffer.reset()

    def feed_observation(self, state: npt.NDArray) -> None:
        # State has shape (<num_workers>, <num_stack_frames>, <height>, <width>, <num_channels>)
        # Convert to shape (<num_workers>, <num_stack_frames> * <num_channels>, <height>, <width>)
        state_tensor = torch.tensor(np.array(state), device=self.device)
        state_tensor = torch.concat(torch.unbind(state_tensor, dim=1), dim=-1)
        state_tensor = torch.moveaxis(state_tensor, -1, 1)
        self.experience_buffer.buffer_states(state_tensor)
