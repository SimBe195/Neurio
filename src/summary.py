from typing import Union

import tensorflow as tf
from hydra.utils import get_original_cwd


class Summary:
    def __init__(self, exp_name) -> None:
        log_dir = f"{get_original_cwd()}/logs/{exp_name}"
        self.file_writer = tf.summary.create_file_writer(log_dir)

        self.update_step = 0
        self.episode_step = 0

    def log_episode_stat(self, stat: Union[int, float], name) -> None:
        with self.file_writer.as_default():
            tf.summary.scalar(f"episode_{name}", stat, self.episode_step)

    def log_update_stat(self, stat: Union[int, float], name) -> None:
        with self.file_writer.as_default():
            tf.summary.scalar(f"update_{name}", stat, self.update_step)

    def next_episode(self) -> None:
        self.episode_step += 1

    def next_update(self) -> None:
        self.update_step += 1
