import torch
import torch.nn as nn
from jaxtyping import Float, Int64

from config.model import ImpalaActorCriticConfig
from environment import EnvironmentInfo

from .impala_encoder import ImpalaEncoder, ResidualBlock


class ActorCritic(nn.Module):
    """Action logits and state value from the stacked frames and the previous action."""

    def __init__(self, config: ImpalaActorCriticConfig, env_info: EnvironmentInfo) -> None:
        super().__init__()
        self.encoder = ImpalaEncoder(env_info, config.channels, config.blocks_per_stage, config.fc_size)
        self.action_embed = nn.Embedding(env_info.num_actions, config.action_embedding_size)
        features = self.encoder.out_size + config.action_embedding_size
        self.actor_layer = nn.Linear(features, env_info.num_actions)
        self.critic_layer = nn.Linear(features, 1)

        self._initialize_weights()
        # Faster convolutions on the GPU, together with bfloat16 in forward()
        self.to(memory_format=torch.channels_last)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # Residual blocks start as the identity, so that activations do not grow with depth
        for module in self.modules():
            if isinstance(module, ResidualBlock):
                nn.init.zeros_(module.conv2.weight)
        # A small policy head keeps the initial action distribution close to uniform
        nn.init.orthogonal_(self.actor_layer.weight, 0.01)
        nn.init.orthogonal_(self.critic_layer.weight, 1.0)

    def forward(
        self, x: Float[torch.Tensor, "*batch channel height width"], prev_actions: Int64[torch.Tensor, "*batch"]
    ) -> tuple[Float[torch.Tensor, "*batch action"], Float[torch.Tensor, "*batch"]]:
        """
        :return: Action logits and state values
        """
        # The encoder runs in bfloat16 on the GPU, the heads in full precision for exact action probabilities
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            x = self.encoder(x.contiguous(memory_format=torch.channels_last))
        x = torch.concat([x.float(), self.action_embed(prev_actions)], dim=1)
        return self.actor_layer(x), self.critic_layer(x).squeeze(-1)
