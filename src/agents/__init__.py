from config.agent import AgentConfig, DreamerAgentConfig, PPOAgentConfig
from environment import EnvironmentInfo

from .agent import CHECKPOINT_PREFIX, Agent, Play, StepOutcome, checkpoint_name
from .dreamer import DreamerAgent
from .ppo import PPOAgent

__all__ = ["CHECKPOINT_PREFIX", "Agent", "Play", "StepOutcome", "checkpoint_name", "get_agent"]


def get_agent(config: AgentConfig, env_info: EnvironmentInfo) -> Agent:
    if isinstance(config, PPOAgentConfig):
        return PPOAgent(config, env_info)
    if isinstance(config, DreamerAgentConfig):
        return DreamerAgent(config, env_info)
    raise NotImplementedError(type(config).__name__)
