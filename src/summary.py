from collections import deque
from datetime import datetime
from typing import Union

import numpy as np
import tensorflow as tf
from hydra.utils import get_original_cwd


class Summary:
    def __init__(self, exp_name) -> None:
        log_dir = (
            f"{get_original_cwd()}/logs/{exp_name}_{datetime.now():%Y-%m-%d_%H:%M}"
        )
        self.file_writer = tf.summary.create_file_writer(log_dir)

        self.episode_histories = {}

        self.update_step = 0
        self.episode_step = 0

    def log_episode_stat(self, stat: Union[int, float], name) -> None:
        if name not in self.episode_histories:
            self.episode_histories[name] = deque(maxlen=5)
        self.episode_histories[name].append(stat)

        hist_array = np.array(self.episode_histories[name])

        min = np.min(hist_array)
        max = np.max(hist_array)
        mean = np.mean(hist_array)

        with self.file_writer.as_default():
            tf.summary.scalar(f"episode_{name}/min", min, self.episode_step)
            tf.summary.scalar(f"episode_{name}/max", max, self.episode_step)
            tf.summary.scalar(f"episode_{name}/mean", mean, self.episode_step)
            tf.summary.scalar(f"episode_{name}/current", stat, self.episode_step)

    def log_update_stat(self, stat: Union[int, float], name) -> None:
        with self.file_writer.as_default():
            tf.summary.scalar(f"update_{name}", stat, self.update_step)

    def next_episode(self) -> None:
        self.episode_step += 1

    def next_update(self) -> None:
        self.update_step += 1
