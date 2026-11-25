from collections.abc import Callable

import torch


class LaProp(torch.optim.Optimizer):
    """
    DreamerV3's optimizer: adaptive gradient clipping, then normalization by the running RMS of the gradients (as in
    Adam), then momentum on the normalized updates (LaProp, Ziyin et al., 2020), with a linear warmup of the learning
    rate.

    Adaptive gradient clipping (Brock et al., 2021) limits the norm of each parameter's gradient to `agc` times the
    norm of the parameter, at least 1e-3.
    """

    def __init__(
        self,
        params,
        lr: float,
        agc: float,
        warmup: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-20,
    ) -> None:
        super().__init__(params, {"lr": lr, "agc": agc, "beta1": beta1, "beta2": beta2, "eps": eps, "warmup": warmup})

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # ty: ignore[invalid-method-override]
        assert closure is None
        # All parameters are updated at once by the _foreach operations, which run as few GPU kernels
        for group in self.param_groups:
            params = [param for param in group["params"] if param.grad is not None]
            if not params:
                continue
            grads = [param.grad for param in params]
            if group["agc"]:
                limits = torch._foreach_clamp_min(torch._foreach_norm(params), 1e-3)
                torch._foreach_mul_(limits, group["agc"])
                ratios = torch._foreach_div(torch._foreach_norm(grads), limits)
                grads = torch._foreach_div(grads, torch._foreach_clamp_min(ratios, 1.0))
            states = [self.state[param] for param in params]
            for param, state in zip(params, states, strict=True):
                if not state:
                    state["rms"] = torch.zeros_like(param)
                    state["momentum"] = torch.zeros_like(param)
            rms, momentum = [state["rms"] for state in states], [state["momentum"] for state in states]
            group["step"] = step = group.get("step", 0) + 1
            beta1, beta2 = group["beta1"], group["beta2"]
            torch._foreach_mul_(rms, beta2)
            torch._foreach_addcmul_(rms, grads, grads, value=1 - beta2)
            denominators = torch._foreach_div(rms, 1 - beta2**step)
            torch._foreach_sqrt_(denominators)
            torch._foreach_add_(denominators, group["eps"])
            torch._foreach_mul_(momentum, beta1)
            torch._foreach_add_(momentum, torch._foreach_div(grads, denominators), alpha=1 - beta1)
            lr = group["lr"] * min(1.0, step / group["warmup"]) if group["warmup"] else group["lr"]
            torch._foreach_add_(params, momentum, alpha=-lr / (1 - beta1**step))
        return None
