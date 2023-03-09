import logging
from pathlib import Path
from typing import Optional, Union

from hydra.utils import get_original_cwd
from torch.utils.tensorboard.writer import SummaryWriter

log = logging.getLogger(__name__)


class Summary:
    def __init__(self, exp_name: str) -> None:
        log_dir = Path(get_original_cwd()) / "logs" / exp_name
        self.file_writer = SummaryWriter(log_dir=log_dir.as_posix())

        self.update_step = 0
        self.episode_step = 0

    def save(self, save_dir: Path):
        with open(save_dir / "summary_state.txt", "w") as f:
            log.info(f"Save update step {self.update_step}")
            log.info(f"Save episode step {self.episode_step}")
            f.write(f"{self.update_step}\n")
            f.write(f"{self.episode_step}\n")

    def load(self, load_dir: Optional[Path] = None):
        if not load_dir:
            return
        with open(load_dir / "summary_state.txt", "r") as f:
            self.update_step, self.episode_step = map(int, f.read().split("\n")[:2])
            log.info(f"Load update step {self.update_step}")
            log.info(f"Load episode step {self.episode_step}")

    def log_episode_stat(self, stat: Union[int, float], name) -> None:
        self.file_writer.add_scalar(f"episode_{name}", stat, self.episode_step)

    def log_update_stat(self, stat: Union[int, float], name) -> None:
        self.file_writer.add_scalar(f"update_{name}", stat, self.update_step)

    def next_episode(self) -> None:
        self.episode_step += 1

    def next_update(self) -> None:
        self.update_step += 1
