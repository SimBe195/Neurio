from config.model import ActorCriticConfig, ModelConfig
from environment import EnvironmentInfo

from .actor_critic import ActorCritic
from .conv_encoder import ConvEncoder
from .model import Model

__all__ = ["ActorCritic", "ConvEncoder", "Model", "get_model"]


def get_model(config: ModelConfig, env_info: EnvironmentInfo) -> Model:
    if isinstance(config, ActorCriticConfig):
        return ActorCritic(config, env_info)
    raise NotImplementedError
