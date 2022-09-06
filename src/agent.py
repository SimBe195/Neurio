import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import logging
from abc import ABC, abstractmethod

from environment import Environment
from models import ActorCritic


class Agent(ABC):
    def __init__(self, environment: Environment, train_mode: bool = True) -> None:
        self.env = environment
        self.train_mode = train_mode

    def feed_state(self, state: np.array) -> None:
        pass

    def update(self) -> None:
        pass

    @abstractmethod
    def next_action(self) -> int:
        pass

    def reset(self) -> None:
        pass

    def give_reward(self, reward: float, done: bool) -> None:
        pass


class RandomAgent(Agent):
    def __init__(self, **kwargs) -> None:
        return super().__init__(**kwargs)

    def next_action(self) -> int:
        return self.env.action_space.sample()


class PPOAgent(Agent):
    def __init__(self, gamma: float = 0.9, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma
        self.clip_param = 0.2

        self.actor_critic = ActorCritic(num_actions=self.env.action_space.n)
        self.opt = tf.keras.optimizers.Adam(learning_rate=1e-05)

        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.log_probs = []

    def next_action(self) -> int:
        if len(self.states) == 1:
            num_channels = tf.shape(self.states[-1])[-1]
            state_hists = tf.pad(self.states[-1], [[0, 0], [0, 0], [0, 2 * num_channels]])
        elif len(self.states) == 2:
            num_channels = tf.shape(self.states[-1])[-1]
            state_hists = tf.concat([self.states[-1], self.states[-2]], axis=-1)
            state_hists = tf.pad(state_hists, [[0, 0], [0, 0], [0, num_channels]])
        else:
            state_hists = tf.concat([self.states[-1], self.states[-2], self.states[-3]], axis=-1)
        in_state = state_hists[None, ...]
        logits, value = self.actor_critic(in_state, np.array(self.actions[-1])[None, ...], training=self.train_mode)
        if self.train_mode:
            logging.debug(f"State value: {value}")
            self.values.append(value.numpy()[0, 0])
            action = tf.random.categorical(logits, 1)[0, 0]
            self.log_probs.append(tf.nn.log_softmax(logits)[0, action])
        else:
            action = tf.argmax(logits, axis=1).numpy()
        self.actions.append(action)
        logging.debug(f"Action: {action}")
        return int(action)

    def give_reward(self, reward: float, done: bool) -> None:
        self.rewards.append(reward)
        self.dones.append(float(done))

    def reset(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.actions.append(0)
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()

    def update(self) -> None:
        if not self.train_mode:
            return
        num_steps = len(self.states)
        if num_steps < 3:
            return

        state_hist = tf.concat([self.states[-1], self.states[-2], self.states[-3]], axis=-1)

        _, v = self.actor_critic(state_hist[None, ...], np.array(self.actions[-1])[None, ...], training=True)
        self.values.append(v)

        gae = 0
        lmbd = 1.0
        advantage_buffer = []
        returns = []

        for reward, value, next_value, done in zip(
            self.rewards, self.values[:-1], self.values[1:], self.dones
        ):
            delta = reward + self.gamma * next_value * (1 - done) - value
            gae = delta + self.gamma * lmbd * (1 - done) * gae
            advantage_buffer.append(gae)
            returns.append(gae + value)

        advantages = np.array(advantage_buffer, dtype=np.float32)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages))

        states = np.stack(self.states, axis=0)
        states_pad_1 = tf.pad(states, [[1, 0], [0, 0], [0, 0], [0, 0]])[:-1]
        states_pad_2 = tf.pad(states, [[2, 0], [0, 0], [0, 0], [0, 0]])[:-2]
        state_hists = tf.concat([states, states_pad_1, states_pad_2], axis=-1)
        actions = np.array(self.actions, dtype=np.int32)
        prev_actions = tf.pad(actions, [[1, 0]])[:-1]
        returns = np.array(returns, dtype=np.float32)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)

        for e in range(10):
            indices = tf.random.shuffle(tf.range(num_steps))
            batch_size = 32
            num_batches = -(-num_steps // batch_size)
            losses = []
            for b in range(num_batches):
                batch_indices = indices[
                    b * batch_size : min(num_steps, (b + 1) * batch_size)
                ]
                batch_state_hists = tf.gather(state_hists, batch_indices)
                batch_old_log_probs = tf.gather(old_log_probs, batch_indices)
                batch_actions = tf.gather(actions, batch_indices)
                batch_prev_actions = tf.gather(prev_actions, batch_indices)
                batch_advantages = tf.gather(advantages, batch_indices)
                batch_returns = tf.gather(returns, batch_indices)

                with tf.GradientTape() as tape:
                    logits, v = self.actor_critic(batch_state_hists, batch_prev_actions, training=True)
                    policy = tf.nn.softmax(logits)

                    # Actor loss
                    new_log_probs = tf.gather(
                        tf.math.log(policy), batch_actions, axis=-1
                    )
                    ratio = tf.math.exp(new_log_probs - batch_old_log_probs)
                    act_loss_1 = batch_advantages * ratio
                    act_loss_2 = batch_advantages * tf.clip_by_value(
                        ratio, 1 - self.clip_param, 1 + self.clip_param
                    )

                    act_loss = tf.math.negative(
                        tf.reduce_mean(tf.math.minimum(act_loss_1, act_loss_2))
                    )

                    # Critic loss
                    crit_loss = tf.keras.losses.Huber()(batch_returns, v[:, 0])

                    # Entropy loss
                    entropy = tf.reduce_mean(-tf.reduce_sum(policy * tf.math.log(tf.clip_by_value(policy, 1e-10, 1))))

                    loss = act_loss + 0.5 * crit_loss - 0.01 * entropy
                    losses.append(loss)

                    grads = tape.gradient(loss, self.actor_critic.trainable_variables)
                    grads = [tf.clip_by_norm(grad, 0.5) for grad in grads]
                    self.opt.apply_gradients(
                        zip(grads, self.actor_critic.trainable_variables)
                    )
        logging.info(f" Updated Agent. Average loss: {sum(losses) / len(losses)}")

    def feed_state(self, state: np.array) -> None:
        self.states.append(state)
