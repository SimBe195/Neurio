from typing import Optional, Tuple

import optuna
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.environment import EnvironmentInfo

from .conv_encoder import ConvEncoder


class ActorCritic(nn.Module):
    def __init__(
        self,
        env_info: EnvironmentInfo,
        config: DictConfig,
        trial: Optional[optuna.Trial] = None,
    ) -> None:
        super().__init__()

        self.encoder = ConvEncoder(env_info, config.encoder, trial)
        enc_size = self.encoder.out_size

        action_embed_size = config.action_embedding_size
        self.action_embed = nn.Embedding(env_info.num_actions, action_embed_size)

        prev_size = enc_size + action_embed_size
        self.actor_layer = nn.Linear(prev_size, env_info.num_actions)
        self.critic_layer = nn.Linear(prev_size, 1)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain("relu"))  # type: ignore
                nn.init.constant_(module.bias, 0)  # type: ignore

    def forward(
        self, x: torch.Tensor, prev_actions: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        a = self.action_embed.forward(prev_actions)

        x_a = torch.concat([x, a], dim=1)

        act = self.actor_layer.forward(x_a)

        if training:
            crit = self.critic_layer.forward(x_a)
            crit = torch.squeeze(crit, -1)
        else:
            crit = torch.zeros((x.size(0),), dtype=torch.float32)

        return act, crit