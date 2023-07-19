from typing import List, Optional

import optuna
import torch
import torch.nn as nn

from src.environment import EnvironmentInfo


class ConvEncoder(nn.Module):
    def __init__(
        self,
        env_info: EnvironmentInfo,
        num_filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        fc_size: int,
    ) -> None:
        super().__init__()

        self.conv_layers = nn.ModuleList()
        prev_filters = env_info.total_channel_dim
        conv_width = env_info.width
        conv_height = env_info.height
        for n_f, k_s, strd in zip(num_filters, kernel_sizes, strides):
            self.conv_layers.append(nn.Conv2d(prev_filters, n_f, k_s, strd))
            self.conv_layers.append(nn.ReLU())
            prev_filters = n_f
            conv_width = (conv_width - k_s) // strd + 1
            conv_height = (conv_height - k_s) // strd + 1

        self.fc_layer = nn.Linear(conv_width * conv_height * prev_filters, fc_size)

        self.out_size = fc_size

    def get_out_size(self) -> int:
        return self.out_size

    def forward(self, x: torch.Tensor):
        x = x / 255.0
        for conv in self.conv_layers:
            x = conv(x)
        x = x.reshape((x.size(0), -1))
        x = self.fc_layer(x)
        x = nn.functional.relu(x)
        return x
