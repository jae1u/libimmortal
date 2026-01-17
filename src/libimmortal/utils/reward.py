import numpy as np
from libimmortal.utils import colormap_to_ids_and_onehot
from libimmortal.utils.aux_func import calculate_distance_map, get_grid_pos


class ImmortalRewardShaper:
    def __init__(self, goal_reward=100.0):
        self.goal_reward = goal_reward
        self.distance_map = None
        self.prev_grid_distance = None

    def reset(self, vector_obs, graphic_obs=None):
        if graphic_obs is not None:
            id_map, _ = colormap_to_ids_and_onehot(graphic_obs)
            self.distance_map = calculate_distance_map(id_map)

            player_x = vector_obs[0]
            player_y = -vector_obs[1]
            grid_x, grid_y = get_grid_pos(player_x, player_y)

            if (
                0 <= grid_y < self.distance_map.shape[0]
                and 0 <= grid_x < self.distance_map.shape[1]
            ):
                self.prev_grid_distance = float(self.distance_map[grid_y, grid_x])
            else:
                self.prev_grid_distance = None
        else:
            self.distance_map = None
            self.prev_grid_distance = None

    def compute_reward(
        self, vector_obs, original_reward, done, truncated, graphic_obs=None
    ):
        # Goal reached
        if original_reward > 0:
            return self.goal_reward

        if self.distance_map is not None:
            # Use grid-based distance
            player_x = vector_obs[0]
            player_y = -vector_obs[1]
            grid_x, grid_y = get_grid_pos(player_x, player_y)

            if not (
                0 <= grid_y < self.distance_map.shape[0]
                and 0 <= grid_x < self.distance_map.shape[1]
            ):
                return -500.0

            player_distance = float(self.distance_map[grid_y, grid_x])
        else:
            # Use vector_obs distance
            player_distance = float(vector_obs[11])

        if player_distance == 0:
            reward = -500.0
        else:
            reward = -player_distance

        return reward
