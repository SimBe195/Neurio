from omegaconf import DictConfig

from .agent import Agent
from .ppo_agent import PPOAgent
from .random_agent import RandomAgent


def get_agent(config: DictConfig, *args, **kwargs) -> Agent:
    if config.name == "random":
        return RandomAgent(config, *args, **kwargs)
    elif config.name == "ppo":
        return PPOAgent(config, *args, **kwargs)
    else:
        raise KeyError(config.name)
