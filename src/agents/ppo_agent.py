import logging
from typing import List

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from src.agents import Agent
from src.agents.advantage import gae_advantage_estimate
from src.agents.experience_buffer import ExperienceBuffer
from src.models import ActorCritic


class PPOAgent(Agent):
    def __init__(
        self,
        config: DictConfig,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config, *args, **kwargs)
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda

        self.epochs_per_update = config.epochs_per_update
        self.batch_size = config.batch_size

        self.clip_param = config.clip_param
        self.critic_loss_weight = config.critic_loss_weight
        self.entropy_loss_weight = config.entropy_loss_weight
        self.grad_clip_norm = config.grad_clip_norm

        self.experience_buffer = ExperienceBuffer(self.num_workers)

        self.actor_critic = ActorCritic(
            config=config.model,
            num_actions=self.num_actions,
        )
        self.opt = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.deserialize(
                OmegaConf.to_container(config.learning_rate, resolve=True)
            ),
            epsilon=config.optimizer_epsilon,
        )

    def set_num_workers(self, num_workers: int) -> None:
        super().set_num_workers(num_workers)
        self.experience_buffer = ExperienceBuffer(num_workers)

    def next_actions(self, train: bool = True) -> List[int]:
        in_states = self.experience_buffer.get_last_states()
        prev_actions = self.experience_buffer.get_last_actions()
        logits, values = self.actor_critic(in_states, prev_actions, training=train)
        self.experience_buffer.buffer_values(values[:, 0].numpy())
        if train:
            actions = tf.random.categorical(logits, 1, dtype=tf.int32)[:, 0].numpy()
        else:
            actions = tf.argmax(logits, axis=-1, output_type=tf.int32).numpy()
        log_probs = tf.gather(
            tf.nn.log_softmax(logits), actions, axis=-1, batch_dims=1
        ).numpy()
        self.experience_buffer.buffer_log_probs(log_probs)
        self.experience_buffer.buffer_actions(actions)
        return actions.tolist(), log_probs.tolist()

    def give_reward(self, reward: List[float], done: List[bool]) -> None:
        self.experience_buffer.buffer_rewards(reward)
        self.experience_buffer.buffer_dones(done)

    def reset(self) -> None:
        self.experience_buffer.reset()

    def save(self, path: str) -> None:
        self.actor_critic.save_weights(path)

    def load(self, path: str) -> None:
        self.actor_critic.load_weights(path)

    def update(self) -> None:
        state = self.experience_buffer.get_last_states()
        action = self.experience_buffer.get_last_actions()
        _, v = self.actor_critic(state, action, training=True)
        self.experience_buffer.buffer_values(v[:, 0].numpy())

        advantages, returns = gae_advantage_estimate(
            self.experience_buffer.get_reward_buffer(),
            self.experience_buffer.get_value_buffer(),
            self.experience_buffer.get_dones_buffer(),
            self.gamma,
            self.gae_lambda,
        )

        states = self.experience_buffer.get_state_buffer()
        actions = self.experience_buffer.get_action_buffer()
        prev_actions = self.experience_buffer.get_prev_action_buffer()
        log_probs = self.experience_buffer.get_log_prob_buffer()

        advantages = tf.reshape(advantages, (-1,))
        returns = tf.reshape(returns, (-1,))
        states = tf.reshape(states, (-1, *(states.shape[2:])))
        actions = tf.reshape(actions, (-1,))
        prev_actions = tf.reshape(prev_actions, (-1,))
        log_probs = tf.reshape(log_probs, (-1,))

        num_steps = states.shape[0]
        losses = {"actor": 0, "critic": 0, "entropy": 0, "total": 0}
        for e in range(self.epochs_per_update):
            indices = tf.random.shuffle(tf.range(num_steps))
            num_batches = -(-num_steps // self.batch_size)
            for b in range(num_batches):
                batch_indices = indices[
                    b * self.batch_size : min(num_steps, (b + 1) * self.batch_size)
                ]
                batch_states = tf.gather(states, batch_indices)
                batch_log_probs = tf.gather(log_probs, batch_indices)
                batch_actions = tf.gather(actions, batch_indices)
                batch_prev_actions = tf.gather(prev_actions, batch_indices)
                batch_advantages = tf.gather(advantages, batch_indices)
                batch_returns = tf.gather(returns, batch_indices)

                with tf.GradientTape() as tape:
                    logits, v = self.actor_critic(
                        batch_states, batch_prev_actions, training=True
                    )
                    policy = tf.nn.softmax(logits)

                    # Actor loss
                    new_log_probs = tf.gather(
                        tf.math.log(policy), batch_actions, axis=1, batch_dims=1
                    )
                    ratio = tf.math.exp(new_log_probs - batch_log_probs)
                    act_loss_1 = batch_advantages * ratio
                    act_loss_2 = batch_advantages * tf.clip_by_value(
                        ratio, 1 - self.clip_param, 1 + self.clip_param
                    )

                    act_loss = tf.math.negative(
                        tf.reduce_mean(tf.math.minimum(act_loss_1, act_loss_2))
                    )
                    losses["actor"] += act_loss

                    # Critic loss
                    crit_loss = tf.keras.losses.Huber()(batch_returns, v[:, 0])
                    losses["critic"] += crit_loss

                    # Entropy loss
                    clipped_policy = tf.clip_by_value(policy, 1e-10, 1)
                    entropy = tf.reduce_mean(
                        tf.reduce_sum(clipped_policy * tf.math.log(clipped_policy))
                    )
                    losses["entropy"] += entropy

                    loss = (
                        act_loss
                        + self.critic_loss_weight * crit_loss
                        + self.entropy_loss_weight * entropy
                    )
                    losses["total"] += loss

                grads = tape.gradient(loss, self.actor_critic.trainable_variables)
                grads = [tf.clip_by_norm(grad, self.grad_clip_norm) for grad in grads]
                self.opt.apply_gradients(
                    zip(grads, self.actor_critic.trainable_variables)
                )
        for key in losses:
            losses[key] /= num_batches * self.epochs_per_update
        logging.info(f"Update finished. Losses: {losses}")
        for key, loss in losses.items():
            self.summary.log_update_stat(loss, f"{key}_loss")
        self.summary.log_update_stat(self.opt._decayed_lr(tf.float32), "learning_rate")
        self.summary.next_update()

        self.experience_buffer.reset()

    def feed_observation(self, state: np.array) -> None:
        # State has shape (<num_workers>, <num_stack_frames>, <height>, <width>, <num_channels>)
        # Stack in channel dimension instead
        state = np.concatenate(np.moveaxis(state, 1, 0), axis=-1)
        self.experience_buffer.buffer_states(state)
