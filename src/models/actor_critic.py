import torch
import torch.nn as nn
from jaxtyping import Float, Int64

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
            strides=self.config.strides,
            fc_size=self.config.fc_size,
        )
        enc_size = self.encoder.out_size

        action_embed_size = self.config.action_embedding_size
        self.action_embed = nn.Embedding(self.env_info.num_actions, action_embed_size)

        prev_size = enc_size + action_embed_size
        self.actor_layer = nn.Linear(prev_size, self.env_info.num_actions)
        self.critic_layer = nn.Linear(prev_size, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # A small policy head keeps the initial action distribution close to uniform
        nn.init.orthogonal_(self.actor_layer.weight, 0.01)
        nn.init.orthogonal_(self.critic_layer.weight, 1.0)

    def forward(
        self, x: Float[torch.Tensor, "*batch channel height width"], prev_actions: Int64[torch.Tensor, "*batch"]
    ) -> tuple[Float[torch.Tensor, "*batch action"], Float[torch.Tensor, "*batch"]]:
        """
        :return: Action logits and state values
        """
        x = self.encoder(x)
        a = self.action_embed(prev_actions)

        x_a = torch.concat([x, a], dim=1)

        # Unnormalized log-probabilities of the actions
        act = self.actor_layer(x_a)

        crit = self.critic_layer(x_a)
        crit = torch.squeeze(crit, -1)

        return act, crit
