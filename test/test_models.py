from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from agents.ppo.actor_critic import ActorCritic
from agents.ppo.impala_encoder import ImpalaEncoder
from agents.ppo.nature_encoder import NatureEncoder
from config.model import ImpalaActorCriticConfig, NatureActorCriticConfig
from environment import EnvironmentInfo


@pytest.mark.parametrize("in_dim", [7, 10, 94, 114])
@pytest.mark.parametrize("kernel_size,stride", [(3, 1), (5, 2), (9, 4), (4, 3)])
def test_conv_out_dim_matches_pytorch(in_dim, kernel_size, stride):
    padding = (kernel_size - 1) // 2
    conv = torch.nn.Conv2d(1, 1, kernel_size, stride, padding=padding)
    out = conv(torch.zeros(1, 1, in_dim, in_dim))
    assert NatureEncoder.calculate_out_dim(in_dim, kernel_size, stride, padding) == out.shape[-1]


def test_actor_critic_initialization():
    env_info = EnvironmentInfo(width=20, height=16, stack_frames=1, image_channels=1, num_actions=6, num_workers=4)
    config = NatureActorCriticConfig(
        action_embedding_size=4, num_filters=[8], kernel_sizes=[3], strides=[1], fc_size=16
    )
    torch.manual_seed(0)
    model = ActorCritic(config, env_info)

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            assert torch.all(module.bias == 0)
    # Orthogonal weights: rows of the (wide) policy head are orthogonal with norm 0.01
    weight = model.actor_layer.weight
    assert torch.allclose(weight @ weight.T, 1e-4 * torch.eye(6), atol=1e-7)

    logits, _ = model(torch.rand(4, 1, 16, 20) * 255, torch.zeros(4, dtype=torch.int64))
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs, torch.full_like(probs, 1 / 6), atol=1e-2)


def test_actor_critic_forward_on_non_square_input():
    env_info = EnvironmentInfo(width=114, height=94, stack_frames=2, image_channels=1, num_actions=5, num_workers=3)
    config = NatureActorCriticConfig(
        action_embedding_size=4, num_filters=[8, 4, 4], kernel_sizes=[8, 4, 3], strides=[4, 2, 1], fc_size=16
    )
    model = ActorCritic(config, env_info)

    states = torch.rand(3, env_info.total_channel_dim, env_info.height, env_info.width) * 255
    logits, values = model(states, torch.tensor([0, 1, 4]))

    assert logits.shape == (3, 5)
    assert values.shape == (3,)


@pytest.mark.parametrize("model", ["nature", "impala"])
def test_configured_models_fit_the_configured_frames(model):
    with initialize_config_dir(str(Path(__file__).parents[1] / "configs"), version_base="1.3"):
        config = instantiate(compose("neurio_config", overrides=["agent=ppo", f"agent/model={model}"]), _convert_="all")
    env = config.environment
    env_info = EnvironmentInfo(env.width, env.height, config.agent.stack_frames, 1, 12, 2)
    states = torch.rand(2, env_info.total_channel_dim, env.height, env.width) * 255
    logits, values = ActorCritic(config.agent.model, env_info)(states, torch.tensor([0, 1]))
    assert logits.shape == (2, 12) and values.shape == (2,)


def test_impala_actor_critic_on_full_observation():
    with initialize_config_dir(str(Path(__file__).parents[1] / "configs"), version_base="1.3"):
        config = instantiate(compose("neurio_config", overrides=["agent/model=impala"]), _convert_="all").agent.model
    assert isinstance(config, ImpalaActorCriticConfig)
    env_info = EnvironmentInfo(width=112, height=96, stack_frames=4, image_channels=1, num_actions=12, num_workers=3)
    model = ActorCritic(config, env_info)

    assert isinstance(model.encoder, ImpalaEncoder)
    states = torch.rand(3, env_info.total_channel_dim, env_info.height, env_info.width) * 255
    logits, values = model(states, torch.tensor([0, 1, 11]))
    assert logits.shape == (3, 12) and values.shape == (3,)
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs, torch.full_like(probs, 1 / 12), atol=1e-2)
