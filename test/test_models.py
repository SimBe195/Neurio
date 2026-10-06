import pytest
import torch

from config.model import NatureActorCriticConfig
from environment import EnvironmentInfo
from models import ActorCritic, NatureEncoder


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


def test_nature_config_covers_whole_screen():
    """The configured encoder on the configured input must have a receptive field spanning the observation."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    with initialize_config_dir(str(Path(__file__).parents[1] / "configs"), version_base="1.3"):
        config = instantiate(compose("neurio_config", overrides=["agent/model=nature"]), _convert_="all")
    model_config, env_config = config.agent.model, config.environment
    height = int((240 - env_config.clip_top - env_config.clip_bot) / env_config.subsampling_factor)
    width = int((256 - env_config.clip_left - env_config.clip_right) / env_config.subsampling_factor)

    receptive_field, jump = 1, 1
    for kernel_size, stride in zip(model_config.kernel_sizes, model_config.strides, strict=True):
        receptive_field += (kernel_size - 1) * jump
        jump *= stride
    env_info = EnvironmentInfo(width, height, env_config.num_stack_frames, 1, 12, 1)
    encoder = NatureEncoder(env_info, model_config.num_filters, model_config.kernel_sizes, model_config.strides, 16)

    assert (height, width) == (94, 114)
    # Each conv output sees a local patch; the fully connected layer combines all of them, i.e. the whole screen
    assert encoder.fc_layer.in_features == model_config.num_filters[-1] * 12 * 14
    assert receptive_field == 36


def test_impala_actor_critic_on_full_observation():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from config.model import ImpalaActorCriticConfig
    from models import ImpalaEncoder

    with initialize_config_dir(str(Path(__file__).parents[1] / "configs"), version_base="1.3"):
        config = instantiate(compose("neurio_config", overrides=["agent/model=impala"]), _convert_="all").agent.model
    assert isinstance(config, ImpalaActorCriticConfig)
    env_info = EnvironmentInfo(width=114, height=94, stack_frames=4, image_channels=1, num_actions=12, num_workers=3)
    model = ActorCritic(config, env_info)

    # Three stages halve 94x114 to 12x15
    assert isinstance(model.encoder, ImpalaEncoder)
    assert model.encoder.fc_layer.in_features == config.channels[-1] * 12 * 15
    states = torch.rand(3, env_info.total_channel_dim, env_info.height, env_info.width) * 255
    logits, values = model(states, torch.tensor([0, 1, 11]))
    assert logits.shape == (3, 12) and values.shape == (3,)
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs, torch.full_like(probs, 1 / 12), atol=1e-2)
