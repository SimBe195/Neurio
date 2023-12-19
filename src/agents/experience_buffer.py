from collections import deque

from beartype import beartype

from jaxtyping import Float, Bool, Int, Int64, jaxtyped

import torch


class ExperienceBuffer:
    def __init__(self, num_workers: int, device: torch.device, size: int | None = None) -> None:
        self.num_workers = num_workers

        self.states: deque[Float[torch.Tensor, "worker channels height width"]] = deque(maxlen=size)
        self.actions: deque[Int64[torch.Tensor, "worker"]] = deque(maxlen=size)
        self.prev_actions: deque[Int64[torch.Tensor, "worker"]] = deque(maxlen=size + 1)
        self.prev_actions.append(torch.zeros(size=(self.num_workers,), dtype=torch.int64, device=device))
        self.values: deque[Float[torch.Tensor, "worker"]] = deque(maxlen=size)
        self.rewards: deque[Float[torch.Tensor, "worker"]] = deque(maxlen=size)
        self.dones: deque[Int64[torch.Tensor, "worker"]] = deque(maxlen=size)
        self.log_probs: deque[Float[torch.Tensor, "worker"]] = deque(maxlen=size)

        self.device = device

    def buffer_states(self, states: Float[torch.Tensor, "worker channels height width"]) -> None:
        self.states.append(states.to(torch.float32))

    def buffer_actions(self, actions: Int[torch.Tensor, "worker"]) -> None:
        actions_cast = actions.to(torch.int64)
        self.prev_actions.append(actions_cast)
        self.actions.append(actions_cast)

    def buffer_values(self, values: Float[torch.Tensor, "worker"]) -> None:
        self.values.append(values.to(torch.float32))

    def buffer_log_probs(self, log_probs: Float[torch.Tensor, "worker"]) -> None:
        self.log_probs.append(log_probs.to(torch.float32))

    def buffer_rewards(self, rewards: Float[torch.Tensor, "worker"]) -> None:
        self.rewards.append(rewards.to(torch.float32))

    def buffer_dones(self, dones: Bool[torch.Tensor, "worker"]) -> None:
        dones_cast = dones.to(torch.int64)
        self.prev_actions.append(torch.multiply(self.prev_actions.pop(), 1 - dones_cast))
        self.dones.append(dones_cast)

    def reset(self, forget_prev_action: bool = False) -> None:
        self.states.clear()
        if forget_prev_action:
            self.prev_actions.clear()
            self.prev_actions.append(torch.zeros(size=(self.num_workers,), dtype=torch.int64, device=self.device))
        else:
            final_action = self.prev_actions[-1]
            self.prev_actions.clear()
            self.prev_actions.append(final_action)
        self.dones.clear()
        self.actions.clear()
        self.values.clear()
        self.rewards.clear()
        self.log_probs.clear()

    def get_last_states(self) -> Float[torch.Tensor, "worker channels height width"]:
        return self.states[-1]

    def get_last_actions(self) -> Float[torch.Tensor, "worker"]:
        return self.prev_actions[-1]

    def get_state_buffer(self) -> Float[torch.Tensor, "buffer worker channels height width"]:
        return torch.stack(list(self.states))

    def get_action_buffer(self) -> Int64[torch.Tensor, "buffer worker"]:
        return torch.stack(list(self.actions))

    def get_prev_action_buffer(self) -> Int64[torch.Tensor, "buffer worker"]:
        return torch.stack(list(self.prev_actions)[:-1])

    def get_value_buffer(self) -> Float[torch.Tensor, "buffer worker"]:
        return torch.stack(list(self.values))

    def get_log_prob_buffer(self) -> Float[torch.Tensor, "buffer worker"]:
        return torch.stack(list(self.log_probs))

    def get_reward_buffer(self) -> Float[torch.Tensor, "buffer worker"]:
        return torch.stack(list(self.rewards))

    def get_dones_buffer(self) -> Int64[torch.Tensor, "buffer worker"]:
        return torch.stack(list(self.dones))
