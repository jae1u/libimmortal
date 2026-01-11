import numpy as np
from gym import spaces
from collections import deque

from libimmortal.utils.enums import *
from libimmortal.utils import colormap_to_ids_and_onehot, parse_observation

"""
class GraphicObservationColorMap:
    BLANK = [0, 0, 0]               # Black
    WALL = [0, 0, 255]              # Blue
    PLATFORM = [113, 69, 1]         # Purple
    TURRET = [179, 0, 255]          # Magenta
    BOMBKID = [255, 2, 0]           # Red
    SKELETON = [255, 255, 255]      # White
    ARROW = [255, 255, 0]           # Yellow
    EXPLOSION = [255, 127, 0]       # Orange
    KNIGHT = [4, 255, 210]          # Cyan
    GOAL = [2, 255, 0]              # Green
    KNIGHT_ATTACK = [128, 128, 204] # Light Blue
"""

class BasicObsBuilder:
    def __init__(self):
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(11, 90, 160), dtype=np.float32),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(103,), dtype=np.float32)
        })

    def build(self, raw_obs):
        graphic_obs, vector_obs = parse_observation(raw_obs)
        id_map, onehot = colormap_to_ids_and_onehot(graphic_obs)

        # normalize
        onehot = onehot.astype(np.float32)
        vector_obs = vector_obs.astype(np.float32)

        return { # Dict
            "image": onehot,
            "vector": vector_obs
        }

class ArrowObsBuilder:
    def __init__(self, history_len=2, MAX_ARROWS=3):
        self.observation_space = spaces.Dict({
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(113+10*MAX_ARROWS,), dtype=np.float32)
        })
        self.history_len = history_len
        self.arrow_history = deque(maxlen=history_len)
        self.MAX_ARROWS = MAX_ARROWS

    def build(self, raw_obs):
        graphic_obs, vector_obs = parse_observation(raw_obs)
        id_map, onehot = colormap_to_ids_and_onehot(graphic_obs)

        arrow_obs = np.zeros((self.MAX_ARROWS, 10), dtype=np.float32)

        arrows_y, arrows_x = np.where(id_map == 6)
        arrows = list(zip(arrows_x, arrows_y))
        self.arrow_history.append(arrows)
        for i, arrow in enumerate(arrows):
            if i >= self.MAX_ARROWS:
                break

            if len(self.arrow_history) >= 2:
                prev_arrows = self.arrow_history[-2]
                if i < len(prev_arrows):
                    vx = arrow[0] - prev_arrows[i][0]
                    vy = arrow[1] - prev_arrows[i][1]
                else:
                    vx, vy = 0, 0
            else:
                vx, vy = 0, 0

            arrow_obs[i] = np.array([0, 0, 0, 1, arrow[0], arrow[1], vx, vy, 0, 0])
        self.arrow_history.popleft()

        
        player_obs = vector_obs[0:13]
        enemy_obs = vector_obs[13:103].reshape(10,9)
        enemy_obs = np.insert(enemy_obs, 3, np.full((10, 1), 0), axis=1) # ENEMY_TYPE_ARROW
        vector_obs = np.concatenate([player_obs, enemy_obs.flatten(), arrow_obs.flatten()])


        # normalize
        vector_obs = vector_obs.astype(np.float32)

        return { # Dict
            "vector": vector_obs
        }