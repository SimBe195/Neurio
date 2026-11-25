"""DreamerV3's world model: an encoder of the frames, a recurrent state-space model (RSSM) and a decoder."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, UInt8

from config.agent import DreamerAgentConfig

from .distributions import OneHotLatent
from .networks import BlockLinear, ChannelNorm, Linear, conv, norm


@dataclass
class State:
    """The model state: deterministic recurrent part and stochastic part, sampled from its logits."""

    deter: Float[torch.Tensor, "*batch deter"]
    stoch: Float[torch.Tensor, "*batch stoch classes"]
    logits: Float[torch.Tensor, "*batch stoch classes"] | None = None

    @property
    def features(self) -> Float[torch.Tensor, "*batch features"]:
        """Input of the heads: both parts concatenated."""
        return torch.cat([self.deter, self.stoch.flatten(-2)], -1)

    def map(self, fn) -> "State":
        return State(fn(self.deter), fn(self.stoch), None if self.logits is None else fn(self.logits))

    def detach(self) -> "State":
        return self.map(torch.Tensor.detach)

    def without_logits(self) -> "State":
        return State(self.deter, self.stoch)

    @staticmethod
    def stack(states: list["State"], dim: int) -> "State":
        """The states stacked, with their logits if the first one has them."""
        logits = [s.logits for s in states if s.logits is not None]
        assert len(logits) in (0, len(states)), "Either all states have logits or none"
        return State(
            torch.stack([s.deter for s in states], dim),
            torch.stack([s.stoch for s in states], dim),
            torch.stack(logits, dim) if logits else None,
        )


class Encoder(nn.Module):
    """Convolutions with max pooling, each halving the resolution, flattened to one vector per frame."""

    def __init__(self, config: DreamerAgentConfig, channels: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for depth in (config.depth * mult for mult in config.mults):
            layers += [conv(channels, depth, config.kernel), nn.MaxPool2d(2), ChannelNorm(depth), nn.SiLU()]
            channels = depth
        self.layers = nn.Sequential(*layers)

    def forward(self, images: UInt8[torch.Tensor, "*batch height width channel"]) -> Float[torch.Tensor, "*batch out"]:
        batch = images.shape[:-3]
        x = images.flatten(0, -4).permute(0, 3, 1, 2).float() / 255 - 0.5
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            x = self.layers(x.contiguous(memory_format=torch.channels_last))
        return x.float().flatten(1).unflatten(0, batch)


class Decoder(nn.Module):
    """
    Reconstructs the frame from the model state: the deterministic part through a block-diagonal layer and the
    stochastic part through an MLP to a low-resolution image, then upsampling convolutions.
    """

    def __init__(self, config: DreamerAgentConfig, height: int, width: int, channels: int) -> None:
        super().__init__()
        depths = [config.depth * mult for mult in config.mults]
        factor = 2 ** len(depths)
        assert height % factor == 0 and width % factor == 0, f"Frames must be divisible by {factor}."
        self.low_res = (height // factor, width // factor)
        self.groups = config.decoder_blocks
        self.top = depths[-1]
        size = self.low_res[0] * self.low_res[1] * self.top
        self.from_deter = BlockLinear(config.deter, size, self.groups)
        self.from_stoch = nn.Sequential(
            Linear(config.stoch * config.classes, 2 * config.units),
            norm(2 * config.units),
            nn.SiLU(),
            Linear(2 * config.units, size),
        )
        self.low_res_norm = ChannelNorm(self.top)
        layers: list[nn.Module] = []
        for depth, next_depth in zip(depths[::-1], depths[-2::-1], strict=False):
            layers += [nn.Upsample(scale_factor=2), conv(depth, next_depth, config.kernel)]
            layers += [ChannelNorm(next_depth), nn.SiLU()]
        layers += [nn.Upsample(scale_factor=2), conv(depths[0], channels, config.kernel)]
        self.layers = nn.Sequential(*layers)

    def forward(self, state: State) -> Float[torch.Tensor, "*batch height width channel"]:
        """:return: The reconstructed frame, with pixel values from 0 to 1"""
        batch = state.deter.shape[:-1]
        height, width = self.low_res
        deter, stoch = state.deter.flatten(0, -2), state.stoch.flatten(0, -3).flatten(1)
        # Each block of the deterministic part produces a group of channels of the whole low resolution image
        x0 = self.from_deter(deter).unflatten(-1, (self.groups, height, width, self.top // self.groups))
        x0 = x0.permute(0, 1, 4, 2, 3).flatten(1, 2)
        x1 = self.from_stoch(stoch).unflatten(-1, (height, width, self.top)).permute(0, 3, 1, 2)
        x = F.silu(self.low_res_norm(x0 + x1))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            x = self.layers(x.contiguous(memory_format=torch.channels_last))
        return torch.sigmoid(x.float()).permute(0, 2, 3, 1).unflatten(0, batch)


class RSSM(nn.Module):
    """
    Recurrent state-space model. Its deterministic state follows a GRU of independent blocks, the stochastic state is
    either inferred from the frame (posterior) or predicted without it (prior), which lets it imagine the future.
    """

    def __init__(self, config: DreamerAgentConfig, token_size: int, num_actions: int) -> None:
        super().__init__()
        self.config = config
        deter, hidden, blocks = config.deter, config.hidden, config.blocks
        stoch_size = config.stoch * config.classes

        def projection(in_features: int) -> nn.Module:
            return nn.Sequential(Linear(in_features, hidden), norm(hidden), nn.SiLU())

        self.in_deter, self.in_stoch, self.in_action = (
            projection(deter),
            projection(stoch_size),
            projection(num_actions),
        )
        self.hidden = nn.Sequential(BlockLinear(deter + 3 * hidden * blocks, deter, blocks), norm(deter), nn.SiLU())
        self.gru = BlockLinear(deter, 3 * deter, blocks)
        self.posterior = nn.Sequential(projection(deter + token_size), Linear(hidden, stoch_size))
        self.prior = nn.Sequential(projection(deter), projection(hidden), Linear(hidden, stoch_size))

    def initial(self, batch_size: int, device: torch.device) -> State:
        config = self.config
        return State(
            torch.zeros(batch_size, config.deter, device=device),
            torch.zeros(batch_size, config.stoch, config.classes, device=device),
        )

    def _state(self, deter: torch.Tensor, flat_logits: torch.Tensor) -> State:
        """The state with the stochastic part sampled from the logits of all its variables."""
        logits = flat_logits.unflatten(-1, (self.config.stoch, self.config.classes))
        return State(deter, OneHotLatent(logits, self.config.unimix).sample(), logits)

    def _core(self, state: State, action: Float[torch.Tensor, "batch action"]) -> Float[torch.Tensor, "batch deter"]:
        blocks = self.config.blocks
        x = torch.cat([self.in_deter(state.deter), self.in_stoch(state.stoch.flatten(1)), self.in_action(action)], -1)
        # Each block sees its part of the deterministic state and all inputs
        x = torch.cat([state.deter.unflatten(-1, (blocks, -1)), x.unsqueeze(1).expand(-1, blocks, -1)], -1)
        x = self.gru(self.hidden(x.flatten(1))).unflatten(-1, (blocks, 3, -1))
        reset, candidate, update = (x[:, :, i].flatten(1) for i in range(3))
        candidate = torch.tanh(torch.sigmoid(reset) * candidate)
        # Biased towards keeping the state, as in DreamerV3
        update = torch.sigmoid(update - 1)
        return update * candidate + (1 - update) * state.deter

    def observe_step(
        self,
        state: State,
        action: Float[torch.Tensor, "batch action"],
        tokens: Float[torch.Tensor, "batch token"],
        is_first: Bool[torch.Tensor, " batch"],
    ) -> State:
        """The next state, given the action that led to the frame and the frame's tokens. Episodes start from zeros."""
        keep = (~is_first).float().unsqueeze(-1)
        state = State(state.deter * keep, state.stoch * keep.unsqueeze(-1))
        deter = self._core(state, action * keep)
        return self._state(deter, self.posterior(torch.cat([deter, tokens], -1)))

    def observe(
        self,
        start: State,
        prev_actions: Float[torch.Tensor, "batch time action"],
        tokens: Float[torch.Tensor, "batch time token"],
        is_first: Bool[torch.Tensor, "batch time"],
    ) -> State:
        """The states of sequences of frames, each given the action that led to it, from the state before them."""
        state = start
        states = []
        for t in range(tokens.shape[1]):
            state = self.observe_step(state, prev_actions[:, t], tokens[:, t], is_first[:, t])
            states.append(state)
        return State.stack(states, 1)

    def imagine_step(self, state: State, action: Float[torch.Tensor, "batch action"]) -> State:
        """The next state after the action, predicted without seeing the frame."""
        deter = self._core(state, action)
        return self._state(deter, self.prior(deter))

    def prior_logits(self, deter: Float[torch.Tensor, "*batch deter"]) -> Float[torch.Tensor, "*batch stoch classes"]:
        return self.prior(deter).unflatten(-1, (self.config.stoch, self.config.classes))

    def kl_losses(self, posterior: State) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :return: The dynamics loss, which trains the prior towards the posterior, and the representation loss, which
            keeps the posterior predictable. Both are at least `free_nats`, so that the model does not collapse the
            latent to make them 0.
        """
        assert posterior.logits is not None
        unimix, free = self.config.unimix, self.config.free_nats
        prior_logits = self.prior_logits(posterior.deter)
        post, post_detached = OneHotLatent(posterior.logits, unimix), OneHotLatent(posterior.logits.detach(), unimix)
        prior, prior_detached = OneHotLatent(prior_logits, unimix), OneHotLatent(prior_logits.detach(), unimix)
        dynamics = post_detached.kl(prior).clamp(min=free)
        representation = post.kl(prior_detached).clamp(min=free)
        return dynamics, representation
