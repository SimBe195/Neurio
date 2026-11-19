"""The output distributions of DreamerV3's networks."""

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int64


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def sample_index(log_probs: Float[torch.Tensor, "*batch classes"]) -> Int64[torch.Tensor, "*batch"]:
    """Samples a class by the Gumbel-max trick, which unlike torch.distributions compiles without graph breaks."""
    uniform = torch.rand_like(log_probs).clamp(min=1e-20)
    return (log_probs - torch.log(-torch.log(uniform))).argmax(-1)


def mix_uniform(logits: Float[torch.Tensor, "*batch classes"], unimix: float) -> Float[torch.Tensor, "*batch classes"]:
    """Log-probabilities of a mixture with the uniform distribution, so that no class becomes impossible."""
    probs = torch.softmax(logits.float(), -1)
    if unimix:
        probs = (1 - unimix) * probs + unimix / probs.shape[-1]
    return torch.log(probs)


class Categorical:
    def __init__(self, logits: Float[torch.Tensor, "*batch classes"], unimix: float = 0.0) -> None:
        self.logits = mix_uniform(logits, unimix) if unimix else logits.float()

    @property
    def log_probs(self) -> Float[torch.Tensor, "*batch classes"]:
        return torch.log_softmax(self.logits, -1)

    def mode(self) -> Int64[torch.Tensor, "*batch"]:
        return self.logits.argmax(-1)

    def sample(self) -> Int64[torch.Tensor, "*batch"]:
        return sample_index(self.log_probs)

    def log_prob(self, index: Int64[torch.Tensor, "*batch"]) -> Float[torch.Tensor, "*batch"]:
        return self.log_probs.gather(-1, index.unsqueeze(-1)).squeeze(-1)

    def entropy(self) -> Float[torch.Tensor, "*batch"]:
        log_probs = self.log_probs
        return -(log_probs.exp() * log_probs).sum(-1)


class OneHotLatent:
    """
    Independent categorical variables, e.g. 32 with 16 classes each, sampled as one-hot vectors with straight-through
    gradients. Log-probabilities, entropies and KL divergences sum over the variables.
    """

    def __init__(self, logits: Float[torch.Tensor, "*batch variable classes"], unimix: float) -> None:
        self.log_probs = mix_uniform(logits, unimix)

    def sample(self) -> Float[torch.Tensor, "*batch variable classes"]:
        probs = self.log_probs.exp()
        onehot = F.one_hot(sample_index(self.log_probs.detach()), probs.shape[-1]).to(probs.dtype)
        return onehot + probs - probs.detach()

    def entropy(self) -> Float[torch.Tensor, "*batch"]:
        return -(self.log_probs.exp() * self.log_probs).sum((-2, -1))

    def kl(self, other: "OneHotLatent") -> Float[torch.Tensor, "*batch"]:
        return (self.log_probs.exp() * (self.log_probs - other.log_probs)).sum((-2, -1))


def twohot_bins(num_bins: int, device: torch.device | None = None) -> Float[torch.Tensor, " bins"]:
    """Bins spaced evenly in symlog space from -20 to 20, symmetric around an exact 0."""
    assert num_bins % 2 == 1
    half = symexp(torch.linspace(-20, 0, (num_bins - 1) // 2 + 1, device=device))
    return torch.cat([half, -half[:-1].flip(0)])


class TwoHot:
    """
    A scalar as a distribution over bins: predictions are the expected bin, targets the two neighbouring bins weighted
    by their distance. Learns values of any scale with the same loss, e.g. rewards and returns.
    """

    def __init__(self, logits: Float[torch.Tensor, "*batch bins"], bins: Float[torch.Tensor, " bins"]) -> None:
        self.logits = logits.float()
        self.bins = bins

    def mean(self) -> Float[torch.Tensor, "*batch"]:
        # Summed from both ends towards the middle, so that symmetric probabilities give exactly 0
        probs = torch.softmax(self.logits, -1)
        middle = (len(self.bins) - 1) // 2
        weighted = probs * self.bins
        return weighted[..., middle] + (weighted[..., :middle].flip(-1) + weighted[..., middle + 1 :]).sum(-1)

    def loss(self, target: Float[torch.Tensor, "*batch"]) -> Float[torch.Tensor, "*batch"]:
        target = target.detach().float()
        below = ((self.bins <= target.unsqueeze(-1)).sum(-1) - 1).clamp(0, len(self.bins) - 1)
        above = (len(self.bins) - (self.bins > target.unsqueeze(-1)).sum(-1)).clamp(0, len(self.bins) - 1)
        equal = below == above
        to_below = torch.where(equal, 1.0, torch.abs(self.bins[below] - target))
        to_above = torch.where(equal, 1.0, torch.abs(self.bins[above] - target))
        total = to_below + to_above
        weights = F.one_hot(below, len(self.bins)) * (to_above / total).unsqueeze(-1) + F.one_hot(
            above, len(self.bins)
        ) * (to_below / total).unsqueeze(-1)
        return -(weights * torch.log_softmax(self.logits, -1)).sum(-1)


class Bernoulli:
    def __init__(self, logits: Float[torch.Tensor, "*batch"]) -> None:
        self.logits = logits.float()

    def prob(self) -> Float[torch.Tensor, "*batch"]:
        return torch.sigmoid(self.logits)

    def loss(self, target: Float[torch.Tensor, "*batch"]) -> Float[torch.Tensor, "*batch"]:
        return F.binary_cross_entropy_with_logits(self.logits, target.detach().float(), reduction="none")
