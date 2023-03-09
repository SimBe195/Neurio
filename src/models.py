from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_width: int,
        in_height: int,
        in_channels: int,
        config: DictConfig,
    ) -> None:
        super().__init__()

        self.scale_factor = 1.0 / 255.0

        if isinstance(config.num_filters, ListConfig):
            num_filters = config.num_filters
        else:
            num_filters = [config.num_filters]

        if isinstance(config.kernel_size, ListConfig):
            kernel_size = config.kernel_size
        else:
            kernel_size = [config.kernel_size]

        if isinstance(config.stride, ListConfig):
            stride = config.stride
        else:
            stride = [config.stride]

        if isinstance(config.fc_size, ListConfig):
            fc_sizes = config.fc_size
        else:
            fc_sizes = [config.fc_size]

        assert len(num_filters) == len(kernel_size) == len(stride)

        self.conv_layers = nn.ModuleList()
        prev_filters = in_channels
        conv_width = in_width
        conv_height = in_height
        for n_f, k_s, strd in zip(num_filters, kernel_size, stride):
            self.conv_layers.append(nn.Conv2d(prev_filters, n_f, k_s, strd))
            self.conv_layers.append(nn.ReLU())
            prev_filters = n_f
            conv_width = (conv_width - k_s) // strd + 1
            conv_height = (conv_height - k_s) // strd + 1

        self.fc_layers = nn.ModuleList()
        in_size = conv_width * conv_height * prev_filters
        for size in fc_sizes:
            self.fc_layers.append(nn.Linear(in_size, size))
            self.fc_layers.append(nn.ReLU())
            in_size = size

        self.out_size = in_size

    def get_out_size(self) -> int:
        return self.out_size

    def forward(self, x: torch.Tensor):
        x = x * self.scale_factor
        for conv in self.conv_layers:
            x = conv(x)
        x = x.reshape((x.size(0), -1))
        for fc in self.fc_layers:
            x = fc(x)
        return x


class ActorCritic(nn.Module):
    def __init__(
        self,
        in_width: int,
        in_height: int,
        in_channels: int,
        num_actions: int,
        config: DictConfig,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions

        self.encoder = ConvEncoder(in_width, in_height, in_channels, config.encoder)
        enc_size = self.encoder.out_size

        self.action_embed = nn.Embedding(num_actions, config.action_embedding_size)

        if isinstance(config.actor_fc_size, ListConfig):
            actor_fc_size = config.actor_fc_size
        else:
            actor_fc_size = [config.actor_fc_size]

        self.actor_layers = nn.ModuleList()
        prev_size = enc_size + config.action_embedding_size
        for size in actor_fc_size:
            self.actor_layers.append(nn.Linear(prev_size, size))
            self.actor_layers.append(nn.ReLU())
            prev_size = size
        self.actor_layers.append(nn.Linear(prev_size, num_actions))

        if isinstance(config.critic_fc_size, ListConfig):
            critic_fc_size = config.critic_fc_size
        else:
            critic_fc_size = [config.critic_fc_size]

        self.critic_layers = nn.ModuleList()
        prev_size = enc_size + config.action_embedding_size
        for size in critic_fc_size:
            self.critic_layers.append(nn.Linear(prev_size, size))
            self.critic_layers.append(nn.ReLU())
            prev_size = size
        self.critic_layers.append(nn.Linear(prev_size, 1))

    def forward(
        self, x: torch.Tensor, prev_actions: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        a = self.action_embed.forward(prev_actions)

        x_a = torch.concat([x, a], dim=1)

        act = x_a
        for layer in self.actor_layers:
            act = layer.forward(act)

        if training:
            crit = x_a
            for layer in self.critic_layers:
                crit = layer.forward(crit)
            crit = torch.squeeze(crit, -1)
        else:
            crit = torch.zeros((x.size(0),), dtype=torch.float32)

        return act, crit
