import torch
import torch.nn as nn
from jaxtyping import Float, Int64

from config.model import ImpalaActorCriticConfig, NatureActorCriticConfig

from .impala_encoder import ImpalaEncoder, ResidualBlock
from .model import Model
from .nature_encoder import NatureEncoder


class ActorCritic(Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder: NatureEncoder | ImpalaEncoder
        if isinstance(self.config, NatureActorCriticConfig):
            self.encoder = NatureEncoder(
                self.env_info,
                num_filters=self.config.num_filters,
                kernel_sizes=self.config.kernel_sizes,
                strides=self.config.strides,
                fc_size=self.config.fc_size,
            )
        else:
            assert isinstance(self.config, ImpalaActorCriticConfig)
            self.encoder = ImpalaEncoder(
                self.env_info, self.config.channels, self.config.blocks_per_stage, self.config.fc_size
            )
        enc_size = self.encoder.out_size

        action_embed_size = self.config.action_embedding_size
        self.action_embed = nn.Embedding(self.env_info.num_actions, action_embed_size)

        prev_size = enc_size + action_embed_size
        self.actor_layer = nn.Linear(prev_size, self.env_info.num_actions)
        self.critic_layer = nn.Linear(prev_size, 1)

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
        x = x.float()
        a = self.action_embed(prev_actions)

        x_a = torch.concat([x, a], dim=1)

        # Unnormalized log-probabilities of the actions
        act = self.actor_layer(x_a)

        crit = self.critic_layer(x_a)
        crit = torch.squeeze(crit, -1)

        return act, crit
