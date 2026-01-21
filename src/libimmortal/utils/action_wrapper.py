from libimmortal.utils.enums import ActionIndex
from typing import Dict, SupportsFloat, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BasicActionWrapper(gym.ActionWrapper):
    """
    Change action space from MultiDiscrete([2, 2, 2, 2, 2, 2, 2, 2]) to MultiDiscrete([3, 3])
    where each dimension represents (0: no-op, 1: negative direction, 2: positive direction)
    1st dimension: vertical movement (up, down)
    2nd dimension: horizontal movement (left, right)
    """

    VERTICAL_MAP = {1: ActionIndex.MOVE_UP, 2: ActionIndex.MOVE_DOWN}
    HORIZONTAL_MAP = {1: ActionIndex.MOVE_LEFT, 2: ActionIndex.MOVE_RIGHT}

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete([3, 3])

    def step(
        self, action
    ) -> tuple[Dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:

        obs, reward, terminated, truncated, info = self.env.step(self.action(action))

        is_success = float(reward) > 0
        terminated |= is_success
        info["is_success"] = is_success

        return obs, reward, terminated, truncated, info

    def action(self, action: np.ndarray) -> np.ndarray:
        _action = np.zeros(4, dtype=action.dtype)
        vertical, horizontal = action
        if vertical in self.VERTICAL_MAP:
            _action[self.VERTICAL_MAP[vertical]] = 1
        if horizontal in self.HORIZONTAL_MAP:
            _action[self.HORIZONTAL_MAP[horizontal]] = 1
        return np.pad(_action, (0, 4))
