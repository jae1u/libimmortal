from pathlib import Path
import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

# from libimmortal.env import ImmortalSufferingEnv

from libimmortal.immortal_gym_env import ImmortalGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from libimmortal.utils import find_n_free_tcp_ports

import gymnasium as gym
import argparse

def make_env(**kwargs):
    def _init() -> gym.Env:
        env = ImmortalGymEnv(**kwargs)
        return env

    return _init

def get_env_fns(args: argparse.Namespace, ports):
    env_fns = [
        make_env(
            game_path=args.game_path,
            port=ports[i],
            time_scale=args.time_scale,
            seed=args.seed + i,
            max_steps=args.max_steps,
            no_filter_observation=args.no_filter_observation,
        )
        for i in range(args.n_envs)
    ]
    return env_fns

def init_ports(args: argparse.Namespace) -> list[int]:
    if args.port is None:
        ports = find_n_free_tcp_ports(args.n_envs)
    else:
        ports = [args.port + i for i in range(args.n_envs)]
    return ports

def main():
    import tqdm
    import argparse

    parser = argparse.ArgumentParser(description="Test Immortal Suffering Environment")
    parser.add_argument(
        "--game_path",
        type=str,
        required=False,
        default="/root/immortal_suffering/immortal_suffering_linux_build.x86_64",
        help="Path to the Unity executable",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default=5005,
        help="Port number for the Unity environment and python api to communicate",
    )
    parser.add_argument(
        "--time_scale",
        type=float,
        required=False,
        default=1.0,  # !NOTE: This will be set as 1.0 in assessment
        help="Speed of the simulation, maximum 2.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Seed that controls enemy spawn",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=False,
        default=720,
        help="Visualized game screen width",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=False,
        default=480,
        help="Visualized game screen height",
    )
    parser.add_argument("--verbose", action="store_true", help="Whether to print logs")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=18000,  # !NOTE: This will be set as 18000 (5 minutes in real-time) in assessment
        help="Number of steps to run the environment",
    )

    parser.add_argument("--no-filter-observation", action="store_true")

    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=10000000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=10, help="각 업데이트당 반복 횟수")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--ent_coef", type=float, default=0.02)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_from_20260131")
    parser.add_argument("--save_freq", type=int, default=10000)
    parser.add_argument("--policy", type=str, default="MultiInputPolicy", help="MultiInputPolicy for Dict observations")
    parser.add_argument("--use_transformer", action="store_true", help="Use transformer-based feature extractor")
    parser.add_argument("--transformer_d_model", type=int, default=128)
    parser.add_argument("--transformer_n_heads", type=int, default=4)
    parser.add_argument("--transformer_n_layers", type=int, default=2)
    parser.add_argument("--transformer_ffn_dim", type=int, default=256)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    parser.add_argument("--image_features_dim", type=int, default=256)
    parser.add_argument("--vector_features_dim", type=int, default=128)
    parser.add_argument("--transformer_vector_only", action="store_true", help="Use transformer over vector only (ignore image) for faster experiments")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use WandB logging (default: enabled)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="immortal-suffering-sb3")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default="/root/libimmortal/checkpoints_from_20260131/ppo_immortal_5040000_steps.zip", help="Path to checkpoint to resume from")
    
    ###################################
    """
    You can add more arguments here for your AI agent if needed.
    """
    ###################################
    
    
    args = parser.parse_args()

    # env = ImmortalSufferingEnv(
    #     game_path=args.game_path,
    #     port=args.port,
    #     time_scale=args.time_scale,  # !NOTE: This will be set as 1.0 in assessment
    #     seed=args.seed,  # !NOTE: This will be set as random number in assessment
    #     width=args.width,
    #     height=args.height,
    #     verbose=args.verbose,
    # )
    
    ports = init_ports(args)
    
    env = SubprocVecEnv(get_env_fns(args, ports), start_method="spawn")
    env = VecNormalize(
        env,
        training=True,
        norm_obs=False,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=args.gamma,
        epsilon=1e-8,
    )
    env = VecNormalize(
        env,
        training=True,
        norm_obs=False,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=args.gamma,
        epsilon=1e-8,
    )

    checkpoint_path = Path(args.resume_from)
    timesteps_str = checkpoint_path.stem.split("_")[-2]
    # vec_normalize_path = (
    #     checkpoint_path.parent / f"ppo_immortal_vecnormalize_{timesteps_str}_steps.pkl"
    # )
    # vec_normalize_path = "/root/libimmortal/checkpoints_from_20260131/ppo_immortal_vecnormalize_3760000_steps.pkl"
    vec_normalize_path = "/root/libimmortal/checkpoints_from_20260131/ppo_immortal_vecnormalize_5040000_steps.pkl"
    env = VecNormalize.load(str(vec_normalize_path), env)

    MAX_STEPS = args.max_steps
    obs = env.reset()

    model = PPO.load(
        args.resume_from,
        env=env,
    )

    ###################################
    """
    Import your AI agent here and replace the random action below with your agent's action.
    """
    ###################################

    for _ in tqdm.tqdm(range(MAX_STEPS), desc="Stepping through environment"):
        
        ###################################
        """
        Do whatever you want with the observation here and get action from your AI agent.
        Replace the random action below with your agent's action.
        """
        ###################################
        
        # action = env.env.action_space.sample()  # REPLACE this with your AI agent's action
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, done, info = env.step(action)

    env.close()


if __name__ == "__main__":
    main()

