from typing import Tuple

import tensorflow as tf
from omegaconf import DictConfig, ListConfig


class ConvEncoder(tf.keras.Sequential):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        initializer = tf.keras.initializers.Orthogonal(
            gain=1.41421356
        )  # gain for RELU activation
        self.add(tf.keras.layers.Rescaling(scale=1.0 / 255.0))

        if isinstance(config.num_filters, ListConfig):
            num_filters = config.num_filters
        else:
            num_filters = [config.num_filters] * config.num_conv_layers

        if isinstance(config.kernel_size, ListConfig):
            kernel_size = config.kernel_size
        else:
            kernel_size = [config.kernel_size] * config.num_conv_layers

        if isinstance(config.stride, ListConfig):
            stride = config.stride
        else:
            stride = [config.stride] * config.num_conv_layers

        for n_f, k_s, strd in zip(num_filters, kernel_size, stride):
            self.add(
                tf.keras.layers.Conv2D(
                    filters=n_f,
                    kernel_size=k_s,
                    strides=(strd, strd),
                    activation="relu",
                    kernel_initializer=initializer,
                    padding="same",
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

        self.encoder = ConvEncoder(config.encoder)

        self.common_linear = tf.keras.Sequential()
        for _ in range(config.num_common_layers):
            self.common_linear.add(
                tf.keras.layers.Dense(
                    config.common_layer_size,
                    activation="relu",
                    kernel_initializer=initializer,
                )
            )

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
                num_actions,
                activation=None,
                kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01),
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
            tf.keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1),
            )
        )

        self.action_embed = tf.keras.Sequential()
        self.action_embed.add(
            tf.keras.layers.Dense(
                config.action_embedding_size,
                activation="relu",
                kernel_initializer=initializer,
            )
        )

    def call(
        self, inputs: tf.Tensor, prev_actions: tf.Tensor, training: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.encoder(inputs)
        a = tf.one_hot(prev_actions, self.num_actions)
        a = self.action_embed(a)
        x = tf.concat([x, a], axis=-1)
        x = self.common_linear(x)
        act = self.actor(x)
        if training:
            crit = self.critic(x)
        else:
            crit = tf.zeros((tf.shape(x)[0], 1), dtype=tf.float32)
        return act, crit
