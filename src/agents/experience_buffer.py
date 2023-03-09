from typing import List

import torch


class ExperienceBuffer:
    def __init__(self, num_workers: int) -> None:
        self.num_workers = num_workers

        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.prev_actions: List[torch.Tensor] = [
            torch.zeros(size=(self.num_workers,), dtype=torch.int64)
        ]
        self.values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

    def buffer_states(self, states: torch.Tensor) -> None:
        assert states.dim() == 4
        assert states.size(0) == self.num_workers
        self.states.append(states.to(torch.float32))

    def buffer_actions(self, actions: torch.Tensor) -> None:
        assert actions.dim() == 1
        assert actions.size(0) == self.num_workers
        actions_cast = actions.to(torch.int64)
        self.prev_actions.append(actions_cast)
        self.actions.append(actions_cast)

    def buffer_values(self, values: torch.Tensor) -> None:
        assert values.dim() == 1
        assert values.size(0) == self.num_workers
        self.values.append(values.to(torch.float32))

    def buffer_log_probs(self, log_probs: torch.Tensor) -> None:
        assert log_probs.dim() == 1
        assert log_probs.size(0) == self.num_workers
        self.log_probs.append(log_probs.to(torch.float32))

    def buffer_rewards(self, rewards: torch.Tensor) -> None:
        assert rewards.dim() == 1
        assert rewards.size(0) == self.num_workers
        self.rewards.append(rewards.to(torch.float32))

    def buffer_dones(self, dones: torch.Tensor) -> None:
        assert dones.dim() == 1
        assert dones.size(0) == self.num_workers
        dones_cast = dones.to(torch.int64)
        self.prev_actions.append(
            torch.multiply(self.prev_actions.pop(-1), 1 - dones_cast)
        )
        self.dones.append(dones_cast)

    def reset(self, forget_prev_action: bool = False) -> None:
        self.states = []
        if forget_prev_action:
            self.prev_actions = [
                torch.zeros(size=(self.num_workers,), dtype=torch.int64)
            ]
        else:
            self.prev_actions = self.prev_actions[-1:]
        self.dones = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.log_probs = []

    def get_last_states(self) -> torch.Tensor:
        return self.states[-1]

    def get_last_actions(self) -> torch.Tensor:
        return self.prev_actions[-1]

    def get_state_buffer(self) -> torch.Tensor:
        return torch.stack(self.states)

    def get_action_buffer(self) -> torch.Tensor:
        return torch.stack(self.actions)

    def get_prev_action_buffer(self) -> torch.Tensor:
        return torch.stack(self.prev_actions[:-1])

    def get_value_buffer(self) -> torch.Tensor:
        return torch.stack(self.values)

    def get_log_prob_buffer(self) -> torch.Tensor:
        return torch.stack(self.log_probs)

    def get_reward_buffer(self) -> torch.Tensor:
        return torch.stack(self.rewards)

    def get_dones_buffer(self) -> torch.Tensor:
        return torch.stack(self.dones)
