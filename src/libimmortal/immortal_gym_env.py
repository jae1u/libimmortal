from typing import Dict, SupportsFloat, Any
import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from libimmortal.utils.reward_wrapper import ImmortalGradReward, NormalizedRewardWrapper
from libimmortal.utils.obs_wrapper import DefaultObsWrapper
from libimmortal.utils.enums import ActionIndex
import gymnasium as gym
from gymnasium.wrappers import PassiveEnvChecker, FilterObservation
from gymnasium import spaces
import shimmy

gym.register_envs(shimmy)


class BasicActionWrapper(gym.ActionWrapper):
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


class ImmortalGymEnv(gym.Wrapper):
    DEFAULT_PATH = "../immortal_suffering/immortal_suffering_linux_build.x86_64"

    def __init__(
        self,
        game_path: str = DEFAULT_PATH,
        port: int = 5005,
        time_scale: float = 2.0,
        seed: int = 42,
        max_steps: int = 2000,
        obs_wrapper_class: type[gym.ObservationWrapper] = DefaultObsWrapper,
        no_filter_observation: bool = False,
    ):
        _engine_channel = EngineConfigurationChannel()
        _env_param_channel = EnvironmentParametersChannel()
        env = UnityEnvironment(
            file_name=game_path,
            base_port=port,
            no_graphics=False,
            side_channels=[_engine_channel, _env_param_channel],
        )
        _engine_channel.set_configuration_parameters(
            time_scale=time_scale,
            target_frame_rate=-1,
            capture_frame_rate=0,
            quality_level=0,
        )
        _env_param_channel.set_float_parameter("seed", float(seed))

        env = UnityToGymWrapper(
            env,
            uint8_visual=True,
            allow_multiple_obs=True,
        )
        env = gym.make(
            "GymV21Environment-v0",
            env=env,
            max_episode_steps=max_steps,
            disable_env_checker=True,
        )
        env = DefaultObsWrapper(env)
        env = ImmortalGradReward(env)
        env = NormalizedRewardWrapper(env)
        if not no_filter_observation:
            env = FilterObservation(env, filter_keys=["image", "vector"])
        env = BasicActionWrapper(env)
        env = PassiveEnvChecker(env)
        super().__init__(env)


def save_image(obs: np.ndarray):
    from PIL import Image

    assert "raw_graphic" in obs, "Enable 'no_filter_observation' to save raw image."
    raw_image = obs["raw_graphic"]
    img = Image.fromarray(raw_image.transpose(1, 2, 0))
    img.save("debug_obs.png")


if __name__ == "__main__":
    env = ImmortalGymEnv(no_filter_observation=True)
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)
    env.close()

    save_image(obs)
    ImmortalGradReward.save_distance_map(obs["id_map"], save_image=True)
