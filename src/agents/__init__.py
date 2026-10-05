from config.agent import AgentConfig, PPOAgentConfig, RandomAgentConfig
from environment import EnvironmentInfo

from .agent import Agent, StepOutcome
from .ppo_agent import PPOAgent
from .random_agent import RandomAgent

__all__ = ["Agent", "PPOAgent", "RandomAgent", "StepOutcome", "get_agent"]


def get_agent(config: AgentConfig, env_info: EnvironmentInfo) -> Agent:
    if isinstance(config, RandomAgentConfig):
        return RandomAgent(config, env_info)
    if isinstance(config, PPOAgentConfig):
        return PPOAgent(config, env_info)
    raise NotImplementedError
