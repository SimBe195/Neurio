import numpy as np


class ExperienceBuffer:
    def __init__(self, num_workers: int, stack_frames: int) -> None:
        self.states = []
        self.actions = []
        self.prev_actions = [np.zeros(shape=(num_workers,), dtype=np.int32)]
        self.values = []
        self.rewards = []
        self.dones = []
        self.log_probs = []

        self.stack_frames = stack_frames
        self.num_workers = num_workers

    def buffer_states(self, states: np.array) -> None:
        # State shape: (workers, width, height, channels)
        # Buffer shape: [(workers, width, height, stack_frames*channels), ...]

        if not len(self.states):
            # No history available for any state (post reset), instead duplicate it
            self.states.append(
                np.stack(
                    [np.concatenate([s] * self.stack_frames, axis=-1) for s in states],
                    axis=0,
                )
            )
        else:
            state_incl_hist = []
            for prev_done, prev_s, s in zip(self.dones[-1], self.states[-1], states):
                if prev_done == 1:
                    state_incl_hist.append(
                        np.concatenate([s] * self.stack_frames, axis=-1)
                    )
                else:
                    prev_s_trimmed = prev_s[..., s.shape[-1] :]
                    state_incl_hist.append(np.concatenate([prev_s_trimmed, s], axis=-1))

            new_states = np.stack(state_incl_hist, axis=0)
            self.states.append(new_states)

    def buffer_actions(self, actions: np.array) -> None:
        self.prev_actions.append(actions)
        self.actions.append(actions)

    def buffer_values(self, values: np.array) -> None:
        self.values.append(values)

    def buffer_log_probs(self, log_probs: np.array) -> None:
        self.log_probs.append(log_probs)

    def buffer_rewards(self, rewards: np.array) -> None:
        self.rewards.append(rewards)

    def buffer_dones(self, dones: np.array) -> None:
        for w in range(self.num_workers):
            if dones[w] == 1:
                self.prev_actions[-1][w] = 0
        self.dones.append(dones)

    def reset(self) -> None:
        if len(self.states):
            # keep last state to build the next histories
            self.states = self.states[-1:]
            self.prev_actions = self.prev_actions[-1:]
            self.dones = self.dones[-1:]
        self.actions = []
        self.values = []
        self.rewards = []
        self.log_probs = []

    def get_last_states(self) -> np.array:
        return self.states[-1]

    def get_last_actions(self) -> np.array:
        return self.prev_actions[-1]

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
