from typing import Tuple

import tensorflow as tf
from omegaconf import DictConfig


class ConvEncoder(tf.keras.Sequential):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        initializer = tf.keras.initializers.Orthogonal(
            gain=1.41421356
        )  # gain for RELU activation
        for _ in range(config.num_conv_layers):
            self.add(
                tf.keras.layers.Conv2D(
                    filters=config.num_filters,
                    kernel_size=config.kernel_size,
                    strides=(config.stride, config.stride),
                    activation="relu",
                    kernel_initializer=initializer,
                )
            )
        self.add(tf.keras.layers.Flatten())
        for _ in range(config.num_linear_layers):
            self.add(
                tf.keras.layers.Dense(
                    config.linear_layer_size,
                    activation="relu",
                    kernel_initializer=initializer,
                )
            )


class ActorCritic(tf.keras.Model):
    def __init__(self, config: DictConfig, num_actions: int) -> None:
        super().__init__()
        initializer = tf.keras.initializers.Orthogonal(
            gain=1.41421356
        )  # gain for RELU activation
        self.num_actions = num_actions

        self.use_action_history = config.use_action_history

        self.encoder = ConvEncoder(config.encoder)

        self.actor = tf.keras.Sequential()
        for _ in range(config.num_actor_layers):
            self.actor.add(
                tf.keras.layers.Dense(
                    config.actor_layer_size,
                    activation="relu",
                    kernel_initializer=initializer,
                )
            )
        self.actor.add(
            tf.keras.layers.Dense(
                num_actions, activation=None, kernel_initializer=initializer
            )
        )

        self.critic = tf.keras.Sequential()
        for _ in range(config.num_critic_layers):
            self.critic.add(
                tf.keras.layers.Dense(
                    config.actor_layer_size,
                    activation="relu",
                    kernel_initializer=initializer,
                )
            )
        self.critic.add(
            tf.keras.layers.Dense(1, activation=None, kernel_initializer=initializer)
        )

        self.action_embed = tf.keras.Sequential()
        if self.use_action_history:
            self.action_embed.add(
                tf.keras.layers.Dense(
                    config.action_embedding_size,
                    activation=None,
                    kernel_initializer=initializer,
                )
            )

    def call(
        self, inputs: tf.Tensor, prev_actions: tf.Tensor, training: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.encoder(inputs)
        if self.use_action_history:
            a = tf.one_hot(prev_actions, self.num_actions)
            a = self.action_embed(a)
            x = tf.concat([x, a], axis=-1)
        act = self.actor(x)
        if training:
            crit = self.critic(x)
        else:
            crit = 0
        return act, crit
