# apply new reward fn
import numpy as np


class ImmortalRewardShaper:
    def __init__(self, 
                 goal_reward=10.0,
                 distance_coef=0.5,
                 kill_reward=0.5,
                 time_penalty=0.001,
                 death_penalty=5.0):
        self.goal_reward = goal_reward
        self.distance_coef = distance_coef
        self.kill_reward = kill_reward
        self.time_penalty = time_penalty
        self.death_penalty = death_penalty
        
        self.prev_goal_distance = None
        self.prev_enemy_count = None
    
    def reset(self, vector_obs):
        self.prev_goal_distance = float(vector_obs[11])
        self.prev_enemy_count = self._count_enemies(vector_obs)
    
    def compute_reward(self, vector_obs, original_reward, done, truncated):
        shaped_reward = 0.0
        
        if original_reward > 0:
            return self.goal_reward
        
        current_distance = float(vector_obs[11])
        if self.prev_goal_distance is not None:
            distance_delta = self.prev_goal_distance - current_distance
            shaped_reward += distance_delta * self.distance_coef
        self.prev_goal_distance = current_distance
        
        current_enemy_count = self._count_enemies(vector_obs)
        if self.prev_enemy_count is not None:
            enemies_killed = self.prev_enemy_count - current_enemy_count
            if enemies_killed > 0:
                shaped_reward += enemies_killed * self.kill_reward
        self.prev_enemy_count = current_enemy_count
        
        shaped_reward -= self.time_penalty
        
        if done and not truncated and original_reward <= 0:
            shaped_reward -= self.death_penalty
        
        return shaped_reward
    
    def _count_enemies(self, vector_obs):
        count = 0
        for i in range(10):
            start_idx = 13 + i * 9
            if start_idx + 7 < len(vector_obs):
                enemy_health = vector_obs[start_idx + 7]
                if enemy_health > 0:
                    count += 1
        return count
