import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from libimmortal.utils.action_wrapper import BasicActionWrapper
from libimmortal.utils.reward_wrapper import ImmortalGradReward, NormalizedRewardWrapper
from libimmortal.utils.obs_wrapper import DefaultObsWrapper, NormalizedVecWrapper
import gymnasium as gym
from gymnasium.wrappers import PassiveEnvChecker, FilterObservation
import shimmy

gym.register_envs(shimmy)


class ImmortalGymEnv(gym.Wrapper):
    DEFAULT_PATH = "../immortal_suffering/immortal_suffering_linux_build.x86_64"

    def __init__(
        self,
        game_path: str = DEFAULT_PATH,
        port: int = 5005,
        time_scale: float = 2.0,
        seed: int = 42,
        max_steps: int = 2000,
        no_filter_observation: bool = False,
    ):
        self.game_path = game_path
        self.port = port
        self.time_scale = time_scale
        self.seed = seed
        self.max_steps = max_steps
        self.no_filter_observation = no_filter_observation
        self.env: gym.Env | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.env is not None:
            self.env.close()

        self.seed = seed if seed is not None else self.seed
        env = self._make_env()
        super().__init__(env)

        assert self.env is not None
        return self.env.reset(seed=seed, options=options)

    def _make_env(self) -> gym.Env:
        _engine_channel = EngineConfigurationChannel()
        _env_param_channel = EnvironmentParametersChannel()
        env = UnityEnvironment(
            file_name=self.game_path,
            base_port=self.port,
            no_graphics=False,
            side_channels=[_engine_channel, _env_param_channel],
        )
        _engine_channel.set_configuration_parameters(
            time_scale=self.time_scale,
            target_frame_rate=-1,
            capture_frame_rate=0,
            quality_level=0,
        )
        _env_param_channel.set_float_parameter("seed", float(self.seed))

        env = UnityToGymWrapper(
            env,
            uint8_visual=True,
            allow_multiple_obs=True,
        )
        env = gym.make(
            "GymV21Environment-v0",
            env=env,
            max_episode_steps=self.max_steps,
            disable_env_checker=True,
        )
        env = BasicActionWrapper(env)
        env = DefaultObsWrapper(env)  # Fix and add onehot/id_map

        env = ImmortalGradReward(env)  # Edit reward using almost raw observation
        # env = NormalizedRewardWrapper(env)  # Normalize reward to 0~1

        env = NormalizedVecWrapper(env)  # Normalize vector observation to -1~1

        if not self.no_filter_observation:
            env = FilterObservation(env, filter_keys=["image", "vector"])

        env = PassiveEnvChecker(env)

        return env


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

    # save_image(obs)
    # ImmortalGradReward.save_distance_map(obs["id_map"], save_image=True)
