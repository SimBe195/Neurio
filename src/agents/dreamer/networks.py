"""Building blocks of DreamerV3's networks: layers with its initialization, normalization and output heads."""

import math

import torch
import torch.nn as nn
from jaxtyping import Float

from .distributions import Bernoulli, Categorical, TwoHot, twohot_bins

# 1 / the standard deviation of a unit normal truncated at ±2
TRUNCATED_NORMAL_STD_CORRECTION = 1.1368


def init_weight(weight: torch.Tensor, fan_in: int, scale: float = 1.0) -> None:
    """Truncated normal with the variance of a unit normal scaled by 1 / fan_in, as in DreamerV3."""
    with torch.no_grad():
        nn.init.trunc_normal_(weight, std=1.0, a=-2.0, b=2.0)
        weight.mul_(TRUNCATED_NORMAL_STD_CORRECTION * math.sqrt(1 / fan_in) * scale)


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, outscale: float = 1.0) -> None:
        super().__init__(in_features, out_features)
        init_weight(self.weight, in_features, outscale)
        nn.init.zeros_(self.bias)


class BlockLinear(nn.Module):
    """A linear layer of independent blocks, i.e. a block-diagonal weight matrix, as in DreamerV3's recurrent core."""

    def __init__(self, in_features: int, out_features: int, blocks: int) -> None:
        super().__init__()
        assert in_features % blocks == 0 and out_features % blocks == 0
        self.blocks = blocks
        self.weight = nn.Parameter(torch.empty(blocks, in_features // blocks, out_features // blocks))
        self.bias = nn.Parameter(torch.zeros(out_features))
        init_weight(self.weight, in_features)

    def forward(self, x: Float[torch.Tensor, "*batch features"]) -> Float[torch.Tensor, "*batch out"]:
        x = x.unflatten(-1, (self.blocks, -1))
        x = torch.einsum("...ki,kio->...ko", x, self.weight)
        return x.flatten(-2) + self.bias


class ChannelNorm(nn.Module):
    """RMS normalization over the channels of an image of shape (batch, channel, height, width)."""

    def __init__(self, channels: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(1, keepdim=True) + self.eps)
        return (x * scale * self.weight[:, None, None]).to(x.dtype)


def norm(features: int) -> nn.RMSNorm:
    return nn.RMSNorm(features, eps=1e-4)


class MLP(nn.Sequential):
    """Linear layers, each followed by normalization and SiLU."""

    def __init__(self, in_features: int, layers: int, units: int) -> None:
        modules: list[nn.Module] = []
        for i in range(layers):
            modules += [Linear(in_features if i == 0 else units, units), norm(units), nn.SiLU()]
        super().__init__(*modules)


class TwoHotHead(nn.Module):
    """A scalar of any scale, e.g. reward or value, see TwoHot. Starts at 0 for all inputs."""

    def __init__(self, in_features: int, layers: int, units: int, bins: int) -> None:
        super().__init__()
        self.mlp = MLP(in_features, layers, units)
        self.out = Linear(units if layers else in_features, bins, outscale=0.0)
        self.bins: torch.Tensor
        self.register_buffer("bins", twohot_bins(bins), persistent=False)

    def forward(self, x: torch.Tensor) -> TwoHot:
        return TwoHot(self.out(self.mlp(x)), self.bins)


class BernoulliHead(nn.Module):
    def __init__(self, in_features: int, layers: int, units: int) -> None:
        super().__init__()
        self.mlp = MLP(in_features, layers, units)
        self.out = Linear(units if layers else in_features, 1)

    def forward(self, x: torch.Tensor) -> Bernoulli:
        return Bernoulli(self.out(self.mlp(x)).squeeze(-1))


class PolicyHead(nn.Module):
    """Categorical distribution over the actions, starting close to uniform."""

    def __init__(self, in_features: int, layers: int, units: int, num_actions: int, unimix: float) -> None:
        super().__init__()
        self.unimix = unimix
        self.mlp = MLP(in_features, layers, units)
        self.out = Linear(units if layers else in_features, num_actions, outscale=0.01)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """The logits before the mixture with the uniform distribution."""
        return self.out(self.mlp(x))

    def forward(self, x: torch.Tensor) -> Categorical:
        return Categorical(self.logits(x), self.unimix)


class Ensemble(nn.Module):
    """Independent heads that each predict the same target, so that their disagreement shows what is not learned yet."""

    def __init__(self, in_features: int, heads: int, layers: int, units: int, out_features: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(MLP(in_features, layers, units), Linear(units if layers else in_features, out_features))
            for _ in range(heads)
        )

    def forward(self, x: Float[torch.Tensor, "*batch features"]) -> Float[torch.Tensor, "heads *batch out"]:
        return torch.stack([head(x) for head in self.heads])


def conv(in_channels: int, out_channels: int, kernel: int, outscale: float = 1.0) -> nn.Conv2d:
    layer = nn.Conv2d(in_channels, out_channels, kernel, padding="same")
    init_weight(layer.weight, in_channels * kernel * kernel, outscale)
    assert layer.bias is not None
    nn.init.zeros_(layer.bias)
    return layer
