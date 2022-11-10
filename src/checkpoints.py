import glob
import re
from typing import Tuple

from hydra.utils import get_original_cwd


class CheckpointHandler:
    def __init__(self, model_name, level):
        self.base_checkpoint_path = (
            f"{get_original_cwd()}/models/{model_name}/{level}/checkpoint."
        )

    def get_save_path(self, epoch: int) -> str:
        return self.base_checkpoint_path + f"{epoch:03d}"

    def checkpoints_exist(self) -> bool:
        return bool(glob.glob(self.base_checkpoint_path + "*"))

    def find_max_saved_epoch(self) -> int:
        def extract_epoch(filename: str) -> Tuple[int, str]:
            search = re.findall("\d+\.", filename)
            return int(search[0]) if search else -1, filename

        return max(glob.glob(self.base_checkpoint_path + "*"), key=extract_epoch)
