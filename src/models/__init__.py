from .actor_critic import *
from .actor_critic import ActorCritic
from .conv_encoder import *
from .model import *


def get_model(config: ModelConfig, env_info: EnvironmentInfo) -> Model:
    if isinstance(config, ActorCriticConfig):
        return ActorCritic(config, env_info)
    raise NotImplementedError
