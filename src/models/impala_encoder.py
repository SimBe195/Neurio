import torch
import torch.nn as nn
from jaxtyping import Float

from environment import EnvironmentInfo


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self, x: Float[torch.Tensor, "batch channel height width"]
    ) -> Float[torch.Tensor, "batch channel height width"]:
        return x + self.conv2(nn.functional.relu(self.conv1(nn.functional.relu(x))))


class ImpalaEncoder(nn.Module):
    """
    Residual encoder of IMPALA (Espeholt et al., 2018), which generalizes to unseen levels much better than the DQN
    encoder (Cobbe et al., 2020). Each stage halves the resolution with a convolution and max pooling, followed by
    residual blocks.
    """

    def __init__(self, env_info: EnvironmentInfo, channels: list[int], blocks_per_stage: int, fc_size: int) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        in_channels, height, width = env_info.total_channel_dim, env_info.height, env_info.width
        for out_channels in channels:
            stages += [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                *(ResidualBlock(out_channels) for _ in range(blocks_per_stage)),
            ]
            in_channels, height, width = out_channels, (height + 1) // 2, (width + 1) // 2
        self.stages = nn.Sequential(*stages)
        self.fc_layer = nn.Linear(in_channels * height * width, fc_size)
        self.out_size = fc_size

    def forward(self, x: Float[torch.Tensor, "batch channel height width"]) -> Float[torch.Tensor, "batch feature"]:
        x = self.stages((x / 127.5) - 1)
        x = nn.functional.relu(x).flatten(1)
        return nn.functional.relu(self.fc_layer(x))
