from typing import Tuple

import torch
import torch.nn as nn

from config.model import ActorCriticConfig
from .conv_encoder import ConvEncoder
from .model import Model


class ActorCritic(Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.config, ActorCriticConfig)

        self.encoder = ConvEncoder(
            self.env_info,
            num_filters=self.config.num_filters,
            kernel_sizes=self.config.kernel_sizes,
            strides=[kernel_size // 2 for kernel_size in self.config.kernel_sizes],
            fc_size=self.config.fc_size,
        )
        enc_size = self.encoder.out_size

        action_embed_size = self.config.action_embedding_size
        self.action_embed = nn.Embedding(self.env_info.num_actions, action_embed_size)

        prev_size = enc_size + action_embed_size
        self.actor_layer = nn.Linear(prev_size, self.env_info.num_actions)
        self.actor_softmax = nn.Softmax(dim=1)
        self.critic_layer = nn.Linear(prev_size, 1)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain("relu"))  # type: ignore
                nn.init.constant_(module.bias, 0)  # type: ignore

    def forward(self, x: torch.Tensor, prev_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        a = self.action_embed(prev_actions)

        x_a = torch.concat([x, a], dim=1)

        act = self.actor_layer(x_a)
        act = self.actor_softmax(act)

        crit = self.critic_layer(x_a)
        crit = torch.squeeze(crit, -1)

        return act, crit
