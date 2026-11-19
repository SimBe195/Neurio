from collections import defaultdict

import torch


class ExperienceBuffer:
    """The steps of all workers since the last update, as one tensor per step and quantity, e.g. "rewards"."""

    def __init__(self) -> None:
        self._steps: defaultdict[str, list[torch.Tensor]] = defaultdict(list)

    def add(self, **tensors: torch.Tensor) -> None:
        for name, tensor in tensors.items():
            self._steps[name].append(tensor)

    def stack(self, name: str) -> torch.Tensor:
        """:return: The quantity of all buffered steps, with shape (step, worker, ...)"""
        return torch.stack(self._steps[name])

    def __len__(self) -> int:
        return len(self._steps["rewards"])

    def clear(self) -> None:
        self._steps.clear()
