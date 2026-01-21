import numpy as np
from typing import Dict
import pickle
from pathlib import Path
import gymnasium as gym
from collections import deque
from libimmortal.utils.aux_func import DEFAULT_ENCODER
import matplotlib.pyplot as plt


class ImmortalGradReward(gym.Wrapper):
    DISTANCE_MAP_PATH = Path("./distance_map.pkl")
    GRID_MAPPING_PATH = Path("./log.txt")
    GOAL_REWARD = 100.0
    BAD_REWARD = -500.0

    def __init__(self, env):
        super().__init__(env)

        assert self.DISTANCE_MAP_PATH.exists()
        with open(self.DISTANCE_MAP_PATH, "rb") as f:
            self.distance_map: np.ndarray = pickle.load(f)

        assert self.GRID_MAPPING_PATH.exists()
        data = np.loadtxt(self.GRID_MAPPING_PATH, delimiter=",")
        slope_x, intercept_x = np.polyfit(data[:, 0], data[:, 2], 1)
        slope_y, intercept_y = np.polyfit(data[:, 1], data[:, 3], 1)
        self.grid_pos_data = (slope_x, intercept_x, slope_y, intercept_y)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = self.reward(observation, reward)
        return observation, reward, terminated, truncated, info

    def reward(self, observation: Dict[str, np.ndarray], original_reward):
        if float(original_reward) > 0:
            return self.GOAL_REWARD

        vector_obs = observation["vector"]
        player_x = vector_obs[0]
        player_y = -vector_obs[1]  # mapping had inverted y-axis
        grid_x, grid_y = self.get_grid_pos(player_x, player_y)

        max_y, max_x = self.distance_map.shape  # note: (y,x) order
        if not (0 <= grid_x < max_x and 0 <= grid_y < max_y):
            return self.BAD_REWARD

        player_distance = self.distance_map[grid_y, grid_x]  # note: (y,x) order

        if player_distance == -1:
            return self.BAD_REWARD

        return -player_distance

    def get_grid_pos(self, player_x, player_y) -> tuple[int, int]:
        slope_x, intercept_x, slope_y, intercept_y = self.grid_pos_data
        gx = int(slope_x * player_x + intercept_x)
        gy = int(slope_y * player_y + intercept_y)
        return gx, gy

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
            plt.imshow(distance_map, cmap="hot", interpolation="nearest")
            plt.colorbar()
            plt.savefig("distance_map.png", bbox_inches="tight")
            plt.close()
