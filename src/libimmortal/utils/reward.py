import numpy as np
from libimmortal.utils import colormap_to_ids_and_onehot
from libimmortal.utils.aux_func import calculate_distance_map, get_grid_pos


class ImmortalRewardShaper:
    """
    Reward shaper based on grid distance map from env.py
    Uses BFS-based distance calculation to goal
    """
    def __init__(self, goal_reward=100.0):
        self.goal_reward = goal_reward
        self.distance_map = None
        self.prev_grid_distance = None
    
    def reset(self, vector_obs, graphic_obs=None):
        """Initialize distance map and previous distance"""
        if graphic_obs is not None:
            # Compute distance map using BFS
            id_map, _ = colormap_to_ids_and_onehot(graphic_obs)
            self.distance_map = calculate_distance_map(id_map)
            
            # Get player position and grid distance
            player_x = vector_obs[0]  # PLAYER_POSITION_X
            player_y = -vector_obs[1]  # PLAYER_POSITION_Y (negated)
            grid_x, grid_y = get_grid_pos(player_x, player_y)
            
            # Bounds check
            if 0 <= grid_y < self.distance_map.shape[0] and 0 <= grid_x < self.distance_map.shape[1]:
                self.prev_grid_distance = float(self.distance_map[grid_y, grid_x])
            else:
                self.prev_grid_distance = None
        else:
            # graphic_obs가 없으면 distance map 사용하지 않음
            self.distance_map = None
            self.prev_grid_distance = None
    
    def compute_reward(self, vector_obs, original_reward, done, truncated, graphic_obs=None):
        # Goal reached
        if original_reward > 0:
            return self.goal_reward
        
        if self.distance_map is not None:
            # Use grid-based distance (from graphic_obs)
            player_x = vector_obs[0]
            player_y = -vector_obs[1]
            grid_x, grid_y = get_grid_pos(player_x, player_y)
            
            if not (0 <= grid_y < self.distance_map.shape[0] and 0 <= grid_x < self.distance_map.shape[1]):
                return -500.0
            
            player_distance = float(self.distance_map[grid_y, grid_x])
        else:
            # Use vector_obs distance (when graphic_obs is not available)
            player_distance = float(vector_obs[11])
        
        if player_distance == 0:
            reward = -500.0
        else:
            reward = -player_distance
        
        return reward

