from src.config.agent import AgentConfig, PPOAgentConfig, RandomAgentConfig
from src.environment import EnvironmentInfo

from .advantage import *
from .agent import *
from .experience_buffer import *
from .ppo_agent import *
from .random_agent import *


def get_agent(config: AgentConfig, env_info: EnvironmentInfo) -> Agent:
    if isinstance(config, RandomAgentConfig):
        return RandomAgent(config, env_info)
    if isinstance(config, PPOAgentConfig):
        return PPOAgent(config, env_info)
    raise NotImplementedError
