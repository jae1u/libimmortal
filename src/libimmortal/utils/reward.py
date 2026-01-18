import numpy as np
from libimmortal.utils.aux_func import calculate_distance_map, get_grid_pos
from typing import Dict
import pickle
from pathlib import Path


class ImmortalRewardShaper:
    GOAL_REWARD = 100.0
    BAD_REWARD = -500.0

    def __init__(self):
        self.distance_map_path = Path("./distance_map.pkl")
        if self.distance_map_path.exists():
            with open(self.distance_map_path, "rb") as f:
                self.distance_map = pickle.load(f)
        else:
            self.distance_map = None

    def compute_reward(self, observation: Dict[str, np.ndarray], original_reward):
        if self.distance_map is None:
            id_map = observation["id_map"]
            self.distance_map = calculate_distance_map(id_map)
            with open(self.distance_map_path, "wb") as f:
                pickle.dump(self.distance_map, f)

        if float(original_reward) > 0:
            return self.GOAL_REWARD

        vector_obs = observation["vector"]
        player_x = vector_obs[0]
        player_y = -vector_obs[1]
        grid_x, grid_y = get_grid_pos(player_x, player_y)

        if not (
            0 <= grid_y < self.distance_map.shape[0]
            and 0 <= grid_x < self.distance_map.shape[1]
        ):
            return self.BAD_REWARD
        player_distance = float(self.distance_map[grid_y, grid_x])

        if player_distance == 0:
            reward = self.BAD_REWARD
        else:
            reward = -player_distance

        return reward
