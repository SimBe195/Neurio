from typing import List

import torch
import torch.nn as nn

from environment import EnvironmentInfo


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
        for n_f, k_s, stride in zip(num_filters, kernel_sizes, strides):
            padding = (k_s - 1) // 2
            self.conv_layers.append(nn.Conv2d(prev_filters, n_f, k_s, stride, padding=padding))
            self.conv_layers.append(nn.ReLU())
            prev_filters = n_f
            conv_width = self.calculate_out_dim(conv_width, k_s, stride, padding)
            conv_height = self.calculate_out_dim(conv_height, k_s, stride, padding)

        self.fc_layer = nn.Linear(conv_width * conv_height * prev_filters, fc_size)

        self.out_size = fc_size

    @staticmethod
    def calculate_out_dim(in_dim: int, kernel_size: int, stride: int, padding: int) -> int:
        return -(-(in_dim + 2 * padding - kernel_size + 1) // stride)

    def get_out_size(self) -> int:
        return self.out_size

    def forward(self, x: torch.Tensor):
        x = x / 255.0
        x = (x / 127.5) - 1
        for conv in self.conv_layers:
            x = conv(x)
        x = x.reshape((x.size(0), -1))
        x = self.fc_layer(x)
        x = nn.functional.relu(x)
        return x
