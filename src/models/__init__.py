from config.model import ImpalaActorCriticConfig, ModelConfig, NatureActorCriticConfig
from environment import EnvironmentInfo

from .actor_critic import ActorCritic
from .impala_encoder import ImpalaEncoder
from .model import Model
from .nature_encoder import NatureEncoder

__all__ = ["ActorCritic", "ImpalaEncoder", "Model", "NatureEncoder", "get_model"]


def get_model(config: ModelConfig, env_info: EnvironmentInfo) -> Model:
    if isinstance(config, NatureActorCriticConfig | ImpalaActorCriticConfig):
        return ActorCritic(config, env_info)
    raise NotImplementedError
