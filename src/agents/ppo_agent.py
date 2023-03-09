import logging

log = logging.getLogger(__name__)
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from omegaconf import OmegaConf

from src.agents import Agent
from src.agents.advantage import gae_advantage_estimate
from src.agents.experience_buffer import ExperienceBuffer
from src.models import ActorCritic


class PPOAgent(Agent):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gamma: float = self.config.gamma
        self.tau: float = self.config.tau

        self.epochs_per_update: int = self.config.epochs_per_update
        self.batch_size: int = self.config.batch_size

        self.clip_param: float = self.config.clip_param
        self.clip_value: float = self.config.clip_value
        self.critic_loss_weight: float = self.config.critic_loss_weight
        self.entropy_loss_weight: float = self.config.entropy_loss_weight
        self.grad_clip_norm: float = self.config.grad_clip_norm

        self.experience_buffer = ExperienceBuffer(self.num_workers)

        self.cpu = torch.device("cpu")
        if torch.cuda.is_available():
            self.gpu = torch.device("cuda")
            device_name = torch.cuda.get_device_name(self.gpu)
            log.info(f"Using GPU {device_name}")
        else:
            self.gpu = self.cpu
            device_name = torch.cuda.get_device_name(self.cpu)
            log.info(f"Using CPU {device_name}")

        self.actor_critic = ActorCritic(
            in_width=self.in_width,
            in_height=self.in_height,
            in_channels=self.in_stack_frames * self.in_channels,
            num_actions=self.num_actions,
            config=self.config.model,
        ).to(self.gpu)
        self.opt = torch.optim.Adam(
            params=self.actor_critic.parameters(),
            lr=self.config.learning_rate.learning_rate,
            eps=self.config.optimizer_epsilon,
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = getattr(
            torch.optim.lr_scheduler, self.config.learning_rate.class_name
        )(
            self.opt,
            **OmegaConf.to_container(self.config.learning_rate.config, resolve=True),
        )

    def set_num_workers(self, num_workers: int) -> None:
        super().set_num_workers(num_workers)
        self.experience_buffer = ExperienceBuffer(num_workers)

    def next_actions(self, train: bool = True) -> Tuple[List[int], List[float]]:
        in_states = self.experience_buffer.get_last_states().to(self.gpu)
        prev_actions = self.experience_buffer.get_last_actions().to(self.gpu)
        with torch.no_grad():
            logits, values = self.actor_critic.forward(
                in_states, prev_actions, training=train
            )
            logits = logits.to(self.cpu)
            values = values.to(self.cpu)

            action_dist = torch.distributions.Categorical(logits=logits)
            if train:
                actions = action_dist.sample()
            else:
                actions = torch.argmax(logits, dim=-1)
            log_probs = action_dist.log_prob(actions)
        self.experience_buffer.buffer_values(values)
        self.experience_buffer.buffer_log_probs(log_probs)
        self.experience_buffer.buffer_actions(actions)
        return [int(action) for action in actions], [float(lp) for lp in log_probs]

    def give_reward(self, reward: List[float], done: List[bool]) -> None:
        self.experience_buffer.buffer_rewards(torch.tensor(reward))
        self.experience_buffer.buffer_dones(torch.tensor(done))

    def reset(self) -> None:
        self.experience_buffer.reset()

    def save(self, path: Path) -> None:
        torch.save(self.actor_critic.state_dict(), path / "model.pt")
        torch.save(self.opt.state_dict(), path / "opt.pt")
        torch.save(self.scheduler.state_dict(), path / "scheduler.pt")

    def load(self, path: Optional[Path] = None) -> None:
        if not path:
            return
        self.actor_critic.load_state_dict(torch.load(path / "model.pt"))
        self.opt.load_state_dict(torch.load(path / "opt.pt"))
        self.scheduler.load_state_dict(torch.load(path / "scheduler.pt"))

    def update(self) -> None:
        state = self.experience_buffer.get_last_states().to(self.gpu)
        action = self.experience_buffer.get_last_actions().to(self.gpu)
        with torch.no_grad():
            _, v = self.actor_critic.forward(state, action, training=True)
        v = v.to(self.cpu)

        values = self.experience_buffer.get_value_buffer()
        values_ext = torch.concat([values, v.unsqueeze(0)])
        advantages, returns = gae_advantage_estimate(
            self.experience_buffer.get_reward_buffer(),
            values_ext,
            self.experience_buffer.get_dones_buffer(),
            self.gamma,
            self.tau,
        )

        states = self.experience_buffer.get_state_buffer()
        actions = self.experience_buffer.get_action_buffer()
        prev_actions = self.experience_buffer.get_prev_action_buffer()
        log_probs = self.experience_buffer.get_log_prob_buffer()

        total_samples = states.size(0) * states.size(1)

        states = states.reshape((total_samples, *states.shape[2:]))
        log_probs = log_probs.reshape((total_samples, *log_probs.shape[2:]))
        actions = actions.reshape((total_samples, *actions.shape[2:]))
        values = values.reshape((total_samples, *values.shape[2:]))
        prev_actions = prev_actions.reshape((total_samples, *prev_actions.shape[2:]))
        advantages = advantages.reshape((total_samples, *advantages.shape[2:]))
        returns = returns.reshape((total_samples, *returns.shape[2:]))

        losses = {
            "actor": 0.0,
            "critic": 0.0,
            "entropy": 0.0,
            "total": 0.0,
        }

        num_batches = -(-total_samples // self.batch_size)  # ceiling division

        for e in range(self.epochs_per_update):
            indices = torch.randperm(total_samples)
            for b in range(num_batches):
                batch_indices = indices[
                    b * self.batch_size : min(total_samples, (b + 1) * self.batch_size)
                ]
                batch_states = states[batch_indices].detach().to(self.gpu)
                batch_log_probs = log_probs[batch_indices].detach().to(self.gpu)
                batch_actions = actions[batch_indices].detach().to(self.gpu)
                batch_values = values[batch_indices].detach().to(self.gpu)
                batch_prev_actions = prev_actions[batch_indices].detach().to(self.gpu)
                batch_advantages = advantages[batch_indices].detach().to(self.gpu)
                batch_returns = returns[batch_indices].detach().to(self.gpu)

                logits, v = self.actor_critic.forward(
                    batch_states, batch_prev_actions, training=True
                )
                policy = torch.softmax(logits, dim=-1)

                # Actor loss
                new_log_probs = torch.gather(
                    torch.log(policy), dim=-1, index=batch_actions.unsqueeze(-1)
                )

                ratio = torch.exp(new_log_probs - batch_log_probs)
                act_loss_1 = batch_advantages * ratio
                act_loss_2 = batch_advantages * torch.clip(
                    ratio, min=1 - self.clip_param, max=1 + self.clip_param
                )

                act_loss = -torch.mean(torch.minimum(act_loss_1, act_loss_2))
                losses["actor"] += act_loss.item()

                # Critic loss
                crit_loss = 0.5 * torch.square(batch_returns - v)
                if self.clip_value:
                    v_clip = torch.clip(
                        v,
                        min=batch_values - self.clip_param,
                        max=batch_values + self.clip_param,
                    )
                    crit_loss_2 = 0.5 * torch.square(batch_returns - v_clip)
                    crit_loss = torch.maximum(crit_loss, crit_loss_2)
                crit_loss = torch.mean(crit_loss)
                losses["critic"] += crit_loss.item()

                # Entropy loss
                policy_clip = torch.clip(policy, min=1e-05, max=1.0)
                entropy = -torch.mean(
                    torch.sum(policy_clip * torch.log(policy_clip), dim=-1)
                )
                losses["entropy"] += entropy.item()

                # Total
                loss = (
                    act_loss
                    + self.critic_loss_weight * crit_loss
                    - self.entropy_loss_weight * entropy
                )
                losses["total"] += loss.item()

                # Optimizer
                self.opt.zero_grad()
                loss.backward(retain_graph=True)
                # torch.nn.utils.clip_grad.clip_grad_norm(
                #     self.actor_critic.parameters(), max_norm=self.grad_clip_norm
                # )
                self.opt.step()

            log.debug(f"Finished epoch {e}.")

        for key in losses:
            losses[key] /= num_batches * self.epochs_per_update
        log.debug(f"Update finished. Losses: {losses}")
        if self.summary is not None:
            for key, loss in losses.items():
                self.summary.log_update_stat(loss, f"loss/{key}")
            self.summary.log_update_stat(
                self.scheduler.get_last_lr()[-1], "learning_rate"
            )
            self.summary.next_update()
        self.scheduler.step()

        self.experience_buffer.reset()

    def feed_observation(self, state: npt.NDArray) -> None:
        # State has shape (<num_workers>, <num_stack_frames>, <height>, <width>, <num_channels>)
        # Convert to shape (<num_workers>, <num_stack_frames> * <num_channels>, <height>, <width>)
        state_tensor = torch.tensor(np.array(state))
        state_tensor = torch.concat(torch.unbind(state_tensor, dim=1), dim=-1)
        state_tensor = torch.moveaxis(state_tensor, -1, 1)
        self.experience_buffer.buffer_states(state_tensor)
