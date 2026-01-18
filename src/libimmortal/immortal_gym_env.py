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
from libimmortal.utils.reward import ImmortalRewardShaper
from libimmortal.utils.obs_wrapper import BasicObsWrapper
import gymnasium as gym
from gymnasium.wrappers import PassiveEnvChecker
from gymnasium import spaces
import shimmy

gym.register_envs(shimmy)


class BasicActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2])

    def step(
        self, action
    ) -> tuple[Dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:

        obs, reward, terminated, truncated, info = self.env.step(self.action(action))

        is_success = float(reward) > 0
        terminated |= is_success
        info["is_success"] = is_success

        return obs, reward, terminated, truncated, info

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.pad(action, (0, 4))


class ImmortalGymEnv(gym.Wrapper):
    DEFAULT_PATH = "../immortal_suffering/immortal_suffering_linux_build.x86_64"

    def __init__(
        self,
        game_path: str = DEFAULT_PATH,
        port: int = 5005,
        time_scale: float = 2.0,
        seed: int = 42,
        max_steps: int = 2000,
        obs_wrapper_class: type[gym.ObservationWrapper] = BasicObsWrapper,
        width: int = 720,
        height: int = 480,
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
            width=width,
            height=height,
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
        env = obs_wrapper_class(env)
        env = BasicActionWrapper(env)
        env = PassiveEnvChecker(env)
        super().__init__(env)

        self.reward_shaper = ImmortalRewardShaper()

    def step(self, action) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, original_reward, terminated, truncated, info = super().step(action)
        shaped_reward = self.reward_shaper.compute_reward(obs, original_reward)
        return obs, shaped_reward, terminated, truncated, info


if __name__ == "__main__":
    env = ImmortalGymEnv()
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward)
    env.close()
