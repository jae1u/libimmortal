from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from libimmortal.utils import colormap_to_ids_and_onehot


class PlayerObs:
    POS_X = 0
    POS_Y = 1
    VEL_X = 2
    VEL_Y = 3
    CUL_DAMAGE = 4
    IS_ACTIONABLE = 5
    IS_HITTING = 6
    IS_DOBBLE_JUMP_AVAILABLE = 7
    IS_ATTACKABLE = 8
    GOAL_POS_X = 9
    GOAL_POS_Y = 10
    GOAL_PLAYER_DIST = 11
    TIME_ELAPSED = 12


class EnemyObs:
    TYPE_SKELETON = 0
    TYPE_BOMBKID = 1
    TYPE_TURRET = 2
    POS_X = 3
    POS_Y = 4
    VEL_X = 5
    VEL_Y = 6
    HEALTH = 7
    STATE = 8


class DefaultObsWrapper(gym.ObservationWrapper):
    """
    Fix graphic observation shape
    Fix vector observation - missing enemy turret info
    Add onehot and id_map observations
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=1, shape=(11, 90, 160), dtype=np.float32
                ),
                "vector": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(103,), dtype=np.float32
                ),
                "id_map": spaces.Box(low=0, high=10, shape=(90, 160), dtype=np.int32),
                "raw_graphic": spaces.Box(
                    low=0, high=255, shape=(3, 90, 160), dtype=np.uint8
                ),
            }
        )

    def observation(self, observation: list[np.ndarray]):
        graphic_obs, vector_obs = observation
        graphic_obs = graphic_obs.flatten().reshape(90, 160, 3)  # Fix shape (HWC)

        id_map, onehot = colormap_to_ids_and_onehot(graphic_obs)
        onehot = onehot.astype(np.float32)
        vector_obs = vector_obs.astype(np.float32)
        id_map = id_map.astype(np.int32)
        graphic_obs = np.transpose(graphic_obs, (2, 0, 1))  # (CHW)

        player_obs = vector_obs[:13]
        enemy_obs = []
        for i in range(10):
            base = 13 + i * 9
            skeleton = vector_obs[base + 1]
            bombkid = vector_obs[base + 2]
            turret = np.float32(skeleton == 0.0 and bombkid == 0.0)
            enemy_obs.extend([skeleton, bombkid, turret])
            enemy_obs.extend(vector_obs[base + 3 : base + 9])

        vector_obs = np.concatenate([player_obs, enemy_obs])

        return {
            "image": onehot,
            "vector": vector_obs,
            "id_map": id_map,
            "raw_graphic": graphic_obs,
        }


class NormalizedVecWrapper(gym.ObservationWrapper):
    """
    return a normalized vector observation
    default ranges from -1 to 1
    CUL_DAMAGE, TIME_ELAPSED are just log scaled
    """

    POS_X_MIN = -15.0
    POS_X_MAX = 16.385
    POS_Y_MIN = -11.0
    POS_Y_MAX = 11.0
    MAX_DISTANCE = 32.0
    HEALTH_MIN = 0.0
    SKELETON_HEALTH_MAX = 20.0
    BOMBKID_HEALTH_MAX = 1.0
    TURRET_HEALTH_MAX = 100.0

    VEL_MIN = -45.0  # Assumed
    VEL_MAX = 45.0  # Assumed

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=1, shape=(11, 90, 160), dtype=np.float32
                ),
                "vector": spaces.Box(low=-1, high=1, shape=(99,), dtype=np.float32),
                "id_map": spaces.Box(low=0, high=10, shape=(90, 160), dtype=np.int32),
                "raw_graphic": spaces.Box(
                    low=0, high=255, shape=(3, 90, 160), dtype=np.uint8
                ),
            }
        )

    def observation(self, observation: dict[str, np.ndarray]):
        vector_obs = observation["vector"]

        player_obs = vector_obs[:9]
        player_obs[PlayerObs.POS_X] = self._normalize(
            player_obs[PlayerObs.POS_X], self.POS_X_MIN, self.POS_X_MAX
        )
        player_obs[PlayerObs.POS_Y] = self._normalize(
            player_obs[PlayerObs.POS_Y], self.POS_Y_MIN, self.POS_Y_MAX
        )
        player_obs[PlayerObs.VEL_X] = self._normalize(
            player_obs[PlayerObs.VEL_X], self.VEL_MIN, self.VEL_MAX
        )
        player_obs[PlayerObs.VEL_Y] = self._normalize(
            player_obs[PlayerObs.VEL_Y], self.VEL_MIN, self.VEL_MAX
        )
        player_obs[PlayerObs.CUL_DAMAGE] = self._normalize(
            np.log1p(player_obs[PlayerObs.CUL_DAMAGE]), 0.0, 500.0
        )
        player_obs[PlayerObs.IS_ACTIONABLE] = (
            player_obs[PlayerObs.IS_ACTIONABLE] * 2 - 1
        )
        player_obs[PlayerObs.IS_HITTING] = player_obs[PlayerObs.IS_HITTING] * 2 - 1
        player_obs[PlayerObs.IS_DOBBLE_JUMP_AVAILABLE] = (
            player_obs[PlayerObs.IS_DOBBLE_JUMP_AVAILABLE] * 2 - 1
        )
        player_obs[PlayerObs.IS_ATTACKABLE] = (
            player_obs[PlayerObs.IS_ATTACKABLE] * 2 - 1
        )

        enemy_obs = []
        for i in range(10):
            base = 13 + i * 9
            if vector_obs[base + 0] == 1.0:
                MAX_HEALTH = self.SKELETON_HEALTH_MAX
            elif vector_obs[base + 1] == 1.0:
                MAX_HEALTH = self.BOMBKID_HEALTH_MAX
            elif vector_obs[base + 2] == 1.0:
                MAX_HEALTH = self.TURRET_HEALTH_MAX
            else:
                MAX_HEALTH = 1.0

            enemy_pos_x = self._normalize(
                vector_obs[base + 3], self.POS_X_MIN, self.POS_X_MAX
            )
            enemy_pos_y = self._normalize(
                vector_obs[base + 4], self.POS_Y_MIN, self.POS_Y_MAX
            )
            enemy_vel_x = self._normalize(
                vector_obs[base + 5], self.VEL_MIN, self.VEL_MAX
            )
            enemy_vel_y = self._normalize(
                vector_obs[base + 6], self.VEL_MIN, self.VEL_MAX
            )

            health = self._normalize(vector_obs[base + 7], self.HEALTH_MIN, MAX_HEALTH)

            enemy_obs.extend(
                [
                    vector_obs[base + 0] * 2 - 1,
                    vector_obs[base + 1] * 2 - 1,
                    vector_obs[base + 2] * 2 - 1,
                    enemy_pos_x,
                    enemy_pos_y,
                    enemy_vel_x,
                    enemy_vel_y,
                    health,
                    vector_obs[base + 8] * 2 - 1,
                ]
            )

        observation["vector"] = np.concatenate([player_obs, enemy_obs]).astype(
            np.float32
        )

        return observation

    @staticmethod
    def _normalize(value, min_value, max_value):
        value = np.clip(value, min_value, max_value)
        return (value - min_value) / (max_value - min_value) * 2 - 1


class ArrowObsWrapper(DefaultObsWrapper):
    def __init__(self, env: gym.Env, history_len: int = 2, max_arrows: int = 3):
        super().__init__(env)
        self.history_len = history_len
        self.max_arrows = max_arrows
        self.arrow_history = deque(maxlen=history_len)

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=1, shape=(11, 90, 160), dtype=np.float32
                ),
                "vector": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(103 + 10 * max_arrows,),
                    dtype=np.float32,
                ),
                "id_map": spaces.Box(low=0, high=10, shape=(90, 160), dtype=np.int32),
                "raw_graphic": spaces.Box(
                    low=0, high=255, shape=(3, 90, 160), dtype=np.uint8
                ),
            }
        )

    def reset(self, **kwargs):
        raise NotImplementedError("ArrowObsWrapper isn't ready for use yet.")
        self.arrow_history.clear()
        return super().reset(**kwargs)

    def observation(self, observation: list[np.ndarray]):
        obs_dict = super().observation(observation)

        arrow_obs = self._extract_arrow_info(obs_dict["id_map"])

        obs_dict["vector"] = np.concatenate([obs_dict["vector"], arrow_obs.flatten()])

        return obs_dict

    def _extract_arrow_info(self, id_map: np.ndarray) -> np.ndarray:
        arrow_obs = np.zeros((self.max_arrows, 10), dtype=np.float32)

        arrows_y, arrows_x = np.where(id_map == 6)
        arrows = list(zip(arrows_x, arrows_y))
        self.arrow_history.append(arrows)
        for i, arrow in enumerate(arrows):
            if i >= self.max_arrows:
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

        return arrow_obs
