from typing import Type

from omegaconf import DictConfig

from .agent import Agent
from .ppo_agent import PPOAgent
from .random_agent import RandomAgent


def get_agent_class(config: DictConfig) -> Type[Agent]:
    return {
        "random": RandomAgent,
        "ppo": PPOAgent,
    }[config.name]
