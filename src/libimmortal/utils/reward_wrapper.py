import numpy as np
from typing import Dict
import pickle
from pathlib import Path
import gymnasium as gym
from collections import deque
from libimmortal.utils.aux_func import DEFAULT_ENCODER
from libimmortal.utils.obs_wrapper import PlayerObs
import matplotlib.pyplot as plt

"""
Note that Reward wrappers are applied after DefaultObsWrapper
The observations provided to the reward wrappers are almost raw observations
DefaultObsWrapper just fix some observation issues and add id_map observation
"""


class ImmortalBasicReward(gym.Wrapper):
    REWARD_RANGE = (0.0, 32.0)

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        distance_offset = PlayerObs.GOAL_PLAYER_DIST
        reward = -observation["vector"][distance_offset]
        return observation, reward, terminated, truncated, info


class ImmortalGradReward(gym.Wrapper):
    REWARD_RANGE = (-12.0, 12.0)
    DISTANCE_MAP_PATH = Path("./distance_map.pkl")
    GOAL_REWARD = 10.0
    BAD_REWARD = -10.0
    TIME_PENALTY = 0.01
    STAGNATION_LIMIT = 300
    PLAYER_ID = DEFAULT_ENCODER.name2id["KNIGHT"]
    USE_MOVING_AVG = True
    MOVING_AVG_WINDOW = 10
    DELTA_SCALE = 5.0
    FRACTIONAL_SCALE = 2.0
    DISTANCE_SCALE = 0.02
    BONUS_MAX = 2.0
    BONUS_DIST_START = 23.0
    BONUS_DIST_END = 11.0

    def __init__(
        self,
        env,
        use_moving_avg: bool | None = None,
        moving_avg_window: int | None = None,
    ):
        super().__init__(env)

        assert self.DISTANCE_MAP_PATH.exists()
        with open(self.DISTANCE_MAP_PATH, "rb") as f:
            self.distance_map: np.ndarray = pickle.load(f)

        self.prev_distance: float | None = None
        self.best_distance: float | None = None
        self.steps_since_progress = 0
        self.use_moving_avg = (
            self.USE_MOVING_AVG if use_moving_avg is None else use_moving_avg
        )
        self.moving_avg_window = (
            self.MOVING_AVG_WINDOW
            if moving_avg_window is None
            else max(1, int(moving_avg_window))
        )
        self.distance_window: deque[float] = deque(maxlen=self.moving_avg_window)
        self.prev_smoothed_distance: float | None = None
        self.prev_bonus: float | None = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward, curr_distance = self.reward(observation, reward)
        # If the agent has not made any progress for a prolonged period,
        # treat it as a terminal failure so that vectorized runners
        # (SB3's SubprocVecEnv/DummyVecEnv) force an immediate reset.
        # Using `terminated` instead of only `truncated` avoids cases where
        # downstream code ignores the truncation flag and keeps stepping.
        # if self._update_stagnation(curr_distance):
        #     terminated = True
        #     info = dict(info)
        #     info["stagnation_steps"] = self.steps_since_progress
        #     info["stagnation_terminated"] = True
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.prev_distance = self._get_player_distance(observation)
        self.best_distance = self.prev_distance
        self.steps_since_progress = 0
        self.distance_window.clear()
        if self.prev_distance is not None and self.use_moving_avg:
            self.distance_window.append(self.prev_distance)
            self.prev_smoothed_distance = float(np.mean(self.distance_window))
        else:
            self.prev_smoothed_distance = None
        self.prev_bonus = None
        return observation, info

    def reward(
        self, observation: Dict[str, np.ndarray], original_reward
    ) -> tuple[float, float | None]:
        if float(original_reward) > 0:
            self.prev_distance = None
            self.best_distance = None
            self.steps_since_progress = 0
            self.distance_window.clear()
            self.prev_smoothed_distance = None
            self.prev_bonus = None
            return self.GOAL_REWARD, None

        curr_distance = self._get_vector_distance(observation)
        if curr_distance is None:
            curr_distance = self._get_player_distance(observation)
        if curr_distance is None:
            self.prev_distance = None
            self.distance_window.clear()
            self.prev_smoothed_distance = None
            self.prev_bonus = None
            return self.BAD_REWARD, None
        curr_distance = float(curr_distance)
        curr_distance += (
            self._get_fractional_offset_bilinear(observation, curr_distance)
            * self.FRACTIONAL_SCALE
        )

        if self.use_moving_avg:
            self.distance_window.append(curr_distance)
            smoothed_distance = float(np.mean(self.distance_window))
        else:
            smoothed_distance = curr_distance

        if self.prev_distance is None or (
            self.use_moving_avg and self.prev_smoothed_distance is None
        ):
            shaped_reward = 0.0
        else:
            anchor_prev = (
                self.prev_smoothed_distance
                if self.use_moving_avg
                else self.prev_distance
            )
            shaped_reward = float(anchor_prev - smoothed_distance) * (self.DELTA_SCALE * 2.0)
        shaped_reward -= self.TIME_PENALTY
        shaped_reward -= smoothed_distance * self.DISTANCE_SCALE

        # print(shaped_reward)

        dist = float(observation["vector"][11])
        bonus = self._smooth_bonus(dist)
        if self.prev_bonus is None:
            bonus_delta = 0.0
        else:
            bonus_delta = self.prev_bonus - bonus
        shaped_reward += bonus_delta
        self.prev_bonus = bonus

        if self.use_moving_avg:
            # Track smoothed distance every step so delta rewards don't go stale.
            self.prev_smoothed_distance = smoothed_distance
        self.prev_distance = curr_distance
        return shaped_reward, curr_distance

    def _smooth_bonus(self, dist: float) -> float:
        if dist >= self.BONUS_DIST_START:
            return 0.0
        if dist <= self.BONUS_DIST_END:
            return self.BONUS_MAX

        t = (self.BONUS_DIST_START - dist) / (
            self.BONUS_DIST_START - self.BONUS_DIST_END
        )
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        return self.BONUS_MAX * t

    def _get_player_distance(self, observation: Dict[str, np.ndarray]) -> float | None:
        id_map = observation["id_map"]
        player_pixels = np.argwhere(id_map == self.PLAYER_ID)

        mean_y, mean_x = np.mean(player_pixels, axis=0)
        grid_y = int(round(mean_y))
        grid_x = int(round(mean_x))

        max_y, max_x = self.distance_map.shape
        if not (0 <= grid_x < max_x and 0 <= grid_y < max_y):
            return None

        player_distance = self.distance_map[grid_y, grid_x]

        if player_distance == -1:
            return None

        return float(player_distance)

    @staticmethod
    def _get_vector_distance(observation: Dict[str, np.ndarray]) -> float | None:
        vector = observation.get("vector")
        if vector is None or len(vector) <= PlayerObs.GOAL_PLAYER_DIST:
            return None

        dist = float(vector[PlayerObs.GOAL_PLAYER_DIST])
        if not np.isfinite(dist):
            return None
        return dist

    def _get_fractional_offset_bilinear(
        self, observation: Dict[str, np.ndarray], base_dist: float
    ) -> float:
        id_map = observation.get("id_map")
        if id_map is None:
            return 0.0

        player_pixels = np.argwhere(id_map == self.PLAYER_ID)
        if player_pixels.size == 0:
            return 0.0

        mean_y, mean_x = np.mean(player_pixels, axis=0)
        x0 = int(np.floor(mean_x))
        y0 = int(np.floor(mean_y))
        x1 = x0 + 1
        y1 = y0 + 1

        max_y, max_x = self.distance_map.shape
        if not (0 <= x0 < max_x and 0 <= y0 < max_y):
            return 0.0
        if not (0 <= x1 < max_x and 0 <= y1 < max_y):
            return 0.0

        d00 = self.distance_map[y0, x0]
        d10 = self.distance_map[y0, x1]
        d01 = self.distance_map[y1, x0]
        d11 = self.distance_map[y1, x1]
        if d00 == -1 or d10 == -1 or d01 == -1 or d11 == -1:
            return 0.0

        dx = float(mean_x - x0)
        dy = float(mean_y - y0)
        interp = (
            (1.0 - dx) * (1.0 - dy) * d00
            + dx * (1.0 - dy) * d10
            + (1.0 - dx) * dy * d01
            + dx * dy * d11
        )
        return float(interp - base_dist)

    def _update_stagnation(self, curr_distance: float | None) -> bool:
        if curr_distance is None:
            self.steps_since_progress = 0
            return False

        if self.best_distance is None or curr_distance < self.best_distance:
            self.best_distance = curr_distance
            self.steps_since_progress = 0
            return False

        self.steps_since_progress += 1
        return self.steps_since_progress >= self.STAGNATION_LIMIT

    @staticmethod
    def calculate_distance_map(id_map: np.ndarray) -> np.ndarray:
        id_map = id_map.copy()

        wall_id = DEFAULT_ENCODER.name2id["WALL"]
        goal_id = DEFAULT_ENCODER.name2id["GOAL"]

        id_map[45:53, 114:130] = wall_id
        id_map[75:83, 120:144] = wall_id

        rows, cols = id_map.shape
        dist_map = np.full((rows, cols), -1, dtype=np.int32)
        queue = deque()
        goal_r, goal_c = np.argwhere(id_map == goal_id)[0]

        dist_map[goal_r, goal_c] = 0
        queue.append((goal_r, goal_c))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            curr_r, curr_c = queue.popleft()
            current_dist = dist_map[curr_r, curr_c]

            for dr, dc in directions:
                next_r, next_c = curr_r + dr, curr_c + dc

                if 0 <= next_r < rows and 0 <= next_c < cols:
                    if (
                        dist_map[next_r, next_c] == -1
                        and id_map[next_r, next_c] != wall_id
                    ):
                        dist_map[next_r, next_c] = current_dist + 1
                        queue.append((next_r, next_c))

        return dist_map

    @staticmethod
    def save_distance_map(id_map: np.ndarray, save_image: bool = False):
        distance_map = ImmortalGradReward.calculate_distance_map(id_map)
        with open(ImmortalGradReward.DISTANCE_MAP_PATH, "wb") as f:
            pickle.dump(distance_map, f)
        if save_image:
            plt.imshow(distance_map, cmap="viridis", interpolation="nearest")
            plt.colorbar()
            plt.savefig("distance_map.png", bbox_inches="tight")
            plt.close()


class NormalizedRewardWrapper(gym.Wrapper):
    """
    return a normalized reward in [0, 1] based on the specified reward range.
    """

    def __init__(self, env: gym.Env, reward_range: tuple[float, float] | None = None):
        super().__init__(env)

        if reward_range is not None:
            self.reward_range = reward_range
        elif hasattr(env, "REWARD_RANGE"):
            self.reward_range = getattr(env, "REWARD_RANGE")
        else:
            raise ValueError("Reward range must be specified.")

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        min_reward, max_reward = self.reward_range
        normalized_reward = (float(reward) - min_reward) / (max_reward - min_reward)
        return observation, normalized_reward, terminated, truncated, info
