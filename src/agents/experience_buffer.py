import numpy as np


class ExperienceBuffer:
    def __init__(self, stack_frames: int) -> None:
        self.states = []
        self.actions = []
        self.prev_actions = [np.zeros(shape=(), dtype=np.int32)]
        self.values = []
        self.rewards = []
        self.dones = []
        self.log_probs = []

        self.stack_frames = stack_frames

    def buffer_state(self, state: np.array) -> None:
        if not len(self.states) or self.dones[-1] == 1:
            # No history available for state, instead duplicate it
            self.states.append(np.concatenate([state] * self.stack_frames, axis=-1))
        else:
            last_stack = self.states[-1]
            last_stack_trimmed = last_stack[..., state.shape[-1] :]
            new_stack = np.concatenate([last_stack_trimmed, state], axis=-1)
            self.states.append(new_stack)

    def buffer_action(self, action: int) -> None:
        np_action = np.array(action, dtype=np.int32)
        self.prev_actions.append(np_action)
        self.actions.append(np_action)

    def buffer_value(self, value: float) -> None:
        self.values.append(np.array(value, dtype=np.float32))

    def buffer_log_prob(self, log_prob: float) -> None:
        self.log_probs.append(np.array(log_prob, dtype=np.float32))

    def buffer_reward(self, reward: float) -> None:
        self.rewards.append(np.array(reward, dtype=np.float32))

    def buffer_done(self, done: bool) -> None:
        self.dones.append(np.array(done, dtype=np.float32))

    def reset(self) -> None:
        if len(self.states) and self.dones[-1] == 0:
            # keep last state to build the next histories
            self.states = self.states[-1:]
            self.prev_actions = self.prev_actions[-1:]
            self.dones = self.dones[-1:]
        else:
            self.states = []
            self.prev_actions = [np.zeros(shape=(), dtype=np.int32)]
            self.dones = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.log_probs = []

    def get_last_state(self) -> np.array:
        return np.array(self.states[-1:])

    def get_last_action(self) -> np.array:
        return self.prev_actions[-1][None, ...]

    def get_state_buffer(self) -> np.array:
        if len(self.states) > len(self.actions):
            # first state has been kept from last update and can be thrown out
            return np.stack(self.states[1:], axis=0)
        return np.stack(self.states, axis=0)

    def get_action_buffer(self) -> np.array:
        return np.stack(self.actions, axis=0)

    def get_prev_action_buffer(self) -> np.array:
        return np.stack(self.prev_actions[:-1], axis=0)

    def get_value_buffer(self) -> np.array:
        return np.stack(self.values, axis=0)

    def get_log_prob_buffer(self) -> np.array:
        return np.stack(self.log_probs, axis=0)

    def get_reward_buffer(self) -> np.array:
        return np.stack(self.rewards, axis=0)

    def get_dones_buffer(self) -> np.array:
        if len(self.dones) > len(self.actions):
            # first done has been kept from last update and can be thrown out
            return np.stack(self.dones[1:], axis=0)
        return np.stack(self.dones, axis=0)
