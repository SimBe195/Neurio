"""
This type stub file was generated by pyright.
"""

from gym import Env, Wrapper

"""An environment wrapper to convert binary to discrete action space."""

class JoypadSpace(Wrapper):
    """An environment wrapper to convert binary to discrete action space."""

    _button_map = ...
    @classmethod
    def buttons(cls) -> list:
        """Return the buttons that can be used as actions."""
        ...
    def __init__(self, env: Env, actions: list) -> None:
        """
        Initialize a new binary to discrete action space wrapper.

        Args:
            env: the environment to wrap
            actions: an ordered list of actions (as lists of buttons).
                The index of each button list is its discrete coded value

        Returns:
            None

        """
        ...
    def step(
        self, action
    ):  # -> Tuple[Unknown, float, bool, bool, dict[Unknown, Unknown]]:
        """
        Take a step using the given action.

        Args:
            action (int): the discrete action to perform

        Returns:
            a tuple of:
            - (numpy.ndarray) the state as a result of the action
            - (float) the reward achieved by taking the action
            - (bool) a flag denoting whether the episode has ended
            - (dict) a dictionary of extra information

        """
        ...
    def reset(self):  # -> Tuple[Unknown, dict[Unknown, Unknown]]:
        """Reset the environment and return the initial observation."""
        ...
    def get_keys_to_action(self):  # -> dict[Unknown, Unknown]:
        """Return the dictionary of keyboard keys to actions."""
        ...
    def get_action_meanings(self):  # -> list[Unknown]:
        """Return a list of actions meanings."""
        ...

__all__ = [JoypadSpace.__name__]
