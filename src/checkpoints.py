from pathlib import Path

from hydra.utils import get_original_cwd


class CheckpointHandler:
    def __init__(self, model_name: str, level: str):
        self.base_checkpoint_path = (
            Path(get_original_cwd()) / "models" / model_name / level
        )

    def get_save_path(self, iter: int) -> Path:
        return self.base_checkpoint_path / f"iter-{iter:05d}"

    def checkpoints_exist(self) -> bool:
        return bool(list(self.base_checkpoint_path.glob("*")))

    def find_max_saved_iter(self) -> int:
        def extract_iter(filename: Path) -> int:
            return int(filename.name[-5:])

        return max(
            [extract_iter(filename) for filename in self.base_checkpoint_path.glob("*")]
        )
