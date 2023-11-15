from config.agent import RandomAgentConfig, PPOAgentConfig, AgentConfig
from environment import EnvironmentInfo
from .agent import Agent
from .ppo_agent import PPOAgent
from .random_agent import RandomAgent


def get_agent(config: AgentConfig, env_info: EnvironmentInfo) -> Agent:
    if isinstance(config, RandomAgentConfig):
        return RandomAgent(config, env_info)
    if isinstance(config, PPOAgentConfig):
        return PPOAgent(config, env_info)
    raise NotImplementedError
