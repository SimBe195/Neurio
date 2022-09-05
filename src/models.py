from typing import Tuple
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import logging


class ConvEncoder(tf.keras.Sequential):
    def __init__(self) -> None:
        super().__init__()
        self.add(
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation="relu"
            )
        )
        self.add(
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation="relu"
            )
        )
        self.add(
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation="relu"
            )
        )
        self.add(
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation="relu"
            )
        )
        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(512, activation="relu"))


class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.encoder = ConvEncoder()

        self.actor = tf.keras.Sequential()
        self.actor.add(tf.keras.layers.Dense(num_actions, activation=None))

        self.critic = tf.keras.Sequential()
        self.critic.add(tf.keras.layers.Dense(1, activation=None))

    def call(
        self, inputs: tf.Tensor, training: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.encoder(inputs)
        act = self.actor(x)
        if training:
            crit = self.critic(x)
        else:
            crit = 0
        return act, crit
