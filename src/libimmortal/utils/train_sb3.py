import argparse
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

from libimmortal.env import ImmortalSufferingEnv
from libimmortal.utils import find_free_tcp_port, find_n_free_tcp_ports
from libimmortal.utils.reward import ImmortalRewardShaper
from libimmortal.utils.obs_builder import BasicObsBuilder, ArrowObsBuilder


class VecNormalizeCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="rl_model", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            # 모델 저장
            model_path = (
                self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
            )
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")

            # VecNormalize stats 저장
            if hasattr(self.training_env, "save"):
                stats_path = (
                    self.save_path / f"vec_normalize_{self.num_timesteps}_steps.pkl"
                )
                self.training_env.save(stats_path)
                if self.verbose > 0:
                    print(f"Saving VecNormalize stats to {stats_path}")

        return True


class GymnasiumWrapper(gym.Env):
    def __init__(
        self, game_path, port, time_scale=2.0, seed=42, max_steps=2000, obs_builder=None
    ):
        super().__init__()

        self.env = ImmortalSufferingEnv(
            game_path=game_path,
            port=port,
            time_scale=time_scale,
            seed=seed,
            width=720,
            height=480,
            verbose=False,
        )

        self.max_steps = max_steps
        self.current_step = 0
        self.obs_builder = obs_builder

        # MLAgents는 MultiDiscrete([2 2 2 2 2 2 2 2])를 사용
        # 각 액션이 0 또는 1
        # 이를 하나의 Discrete space로 변환: 2^8 = 256 가지
        original_action_space = self.env.env.action_space

        if hasattr(original_action_space, "nvec"):
            self.action_dims = original_action_space.nvec
            self.n_actions = int(np.prod(self.action_dims))
            self.action_space = spaces.Discrete(self.n_actions)
            self.is_multi_discrete = True
        else:
            self.action_space = original_action_space
            self.is_multi_discrete = False

        if obs_builder is not None:
            temp_obs = self.env.reset()
            built_obs = obs_builder.build(temp_obs)

            if isinstance(built_obs, dict):
                obs_spaces = {}
                for key, value in built_obs.items():
                    if key == "image":
                        obs_spaces[key] = spaces.Box(
                            low=0, high=255, shape=value.shape, dtype=value.dtype
                        )
                    elif key == "vector":
                        obs_spaces[key] = spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=value.shape,
                            dtype=value.dtype,
                        )
                self.observation_space = spaces.Dict(obs_spaces)
            else:
                # vector only
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=built_obs.shape,
                    dtype=built_obs.dtype,
                )
        else:
            # raw mode: graphic + vector
            self.observation_space = spaces.Dict(
                {
                    "image": spaces.Box(
                        low=0, high=255, shape=(3, 90, 160), dtype=np.uint8
                    ),
                    "vector": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(103,), dtype=np.float32
                    ),
                }
            )

        self.reward_shaper = ImmortalRewardShaper()

        self.episode_step = 0
        self.episode_reward = 0
        self.goal_reached = False
        self.enemies_killed_this_ep = 0
        self.min_distance_this_ep = float("inf")

    def _discrete_to_multi_discrete(self, action):
        if not self.is_multi_discrete:
            return action
        multi_action = []
        remaining = int(action)
        for dim in reversed(self.action_dims):
            multi_action.append(remaining % dim)
            remaining //= dim
        multi_action.reverse()

        return np.array(multi_action, dtype=np.int32)

    def _count_enemies(self, vector_obs):
        count = 0
        for i in range(10):
            start_idx = 13 + i * 9
            if start_idx + 7 < len(vector_obs):
                enemy_health = vector_obs[start_idx + 7]
                if enemy_health > 0:
                    count += 1
        return count

    def _parse_observation(self, obs):
        if isinstance(obs, dict):
            if "image" in obs:
                return obs["image"], obs.get("vector", np.zeros(103))
            elif "graphic" in obs:
                return obs["graphic"], obs.get("vector", np.zeros(103))
            else:
                return obs.get("vector", np.zeros(103)), obs.get(
                    "vector", np.zeros(103)
                )
        elif isinstance(obs, tuple) and len(obs) == 2:
            return obs
        else:
            return obs[0], obs[1]

    def reset(self, seed=None, options=None):
        if self.episode_step > 0:
            print(
                f"[EP] steps={self.episode_step} reward={self.episode_reward:.2f} "
                f"goal={self.goal_reached} "
                f"min_dist={self.min_distance_this_ep:.1f}"
            )

        raw_obs = self.env.reset()

        if self.obs_builder:
            obs = self.obs_builder.build(raw_obs)

            if isinstance(self.obs_builder, BasicObsBuilder):
                # BasicObsBuilder
                raw_graphic_obs, raw_vector_obs = self._parse_observation(raw_obs)
                self.reward_shaper.reset(raw_vector_obs, raw_graphic_obs)
                self.prev_enemies = self._count_enemies(raw_vector_obs)
            else:
                # ArrowObsBuilder
                _, raw_vector_obs = self._parse_observation(raw_obs)
                self.reward_shaper.reset(raw_vector_obs, None)
                self.prev_enemies = self._count_enemies(raw_vector_obs)

            vector_obs = obs.get("vector", obs.get("image", np.zeros(103)))
        else:
            graphic_obs, vector_obs = self._parse_observation(raw_obs)
            self.reward_shaper.reset(vector_obs, graphic_obs)
            obs = {"image": graphic_obs, "vector": vector_obs}

        self.current_step = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.goal_reached = False
        self.min_distance_this_ep = float("inf")

        return obs, {}

    def step(self, action):
        multi_action = self._discrete_to_multi_discrete(action)
        raw_obs, reward, done, info = self.env.step(multi_action)

        self.current_step += 1
        self.episode_step += 1

        if self.obs_builder:
            obs_dict = self.obs_builder.build(raw_obs)

            if isinstance(self.obs_builder, BasicObsBuilder):
                raw_graphic_obs, raw_vector_obs = self._parse_observation(raw_obs)
                vector_for_reward = raw_vector_obs
                graphic_obs = raw_graphic_obs
            else:
                _, raw_vector_obs = self._parse_observation(raw_obs)
                vector_for_reward = raw_vector_obs
                graphic_obs = None
        else:
            graphic_obs, vector_obs = self._parse_observation(raw_obs)
            vector_for_reward = vector_obs
            obs_dict = {"image": graphic_obs, "vector": vector_obs}

        current_distance = float(vector_for_reward[11])
        self.min_distance_this_ep = min(self.min_distance_this_ep, current_distance)

        if reward > 0:
            self.goal_reached = True

        truncated = self.current_step >= self.max_steps
        if truncated and not done:
            info["TimeLimit.truncated"] = True

        shaped_reward = self.reward_shaper.compute_reward(
            vector_for_reward, reward, done, truncated, graphic_obs
        )
        self.episode_reward += shaped_reward

        info["goal_distance"] = current_distance
        info["goal_reached"] = self.goal_reached

        return obs_dict, shaped_reward, done, truncated, info

    def close(self):
        self.env.close()


def make_env(
    game_path, port, time_scale=2.0, seed=42, max_steps=2000, obs_builder=None
):
    def _init():
        env = GymnasiumWrapper(
            game_path, port, time_scale, seed, max_steps, obs_builder
        )
        env = Monitor(env)
        return env

    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="SB3 PPO 학습")

    parser.add_argument(
        "--game_path",
        type=str,
        default="/root/immortal_suffering/immortal_suffering_linux_build.x86_64",
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--time_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_timesteps", type=int, default=10000000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument(
        "--n_epochs", type=int, default=20, help="각 업데이트당 반복 횟수"
    )
    parser.add_argument(
        "--max_steps", type=int, default=2000, help="에피소드 최대 스텝 (truncate)"
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_freq", type=int, default=10000)
    parser.add_argument(
        "--policy",
        type=str,
        default="MultiInputPolicy",
        help="MultiInputPolicy for Dict observations",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use WandB logging")
    parser.add_argument("--wandb_project", type=str, default="immortal-suffering-sb3")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--obs_type", type=str, default="basic", choices=["basic", "arrow", "raw"]
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.obs_type == "basic":
        obs_builder = BasicObsBuilder()
    elif args.obs_type == "arrow":
        obs_builder = ArrowObsBuilder()
    else:
        obs_builder = None

    # 포트 자동 할당 (n_envs 개수만큼)
    if args.port is None:
        ports = find_n_free_tcp_ports(args.n_envs)
        print(f"할당된 포트: {ports}")
    else:
        ports = [args.port + i for i in range(args.n_envs)]

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.use_wandb:
        run_name = (
            args.wandb_run_name
            or f"ppo_{args.obs_type}_{args.seed}_{args.learning_rate}"
        )

        # 체크포인트 재개 시 기존 run에 이어서 로깅
        if args.resume_from and Path(args.resume_from).exists():
            # run_id 파일이 있으면 같은 run 사용
            run_id_file = Path(args.checkpoint_dir) / "wandb_run_id.txt"
            if run_id_file.exists():
                run_id = run_id_file.read_text().strip()
                print(f"WandB run 재개: {run_id}")
                wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    id=run_id,
                    resume="allow",
                    config=vars(args),
                    monitor_gym=True,
                    save_code=True,
                )
            else:
                # run_id 파일이 없으면 새로 생성
                wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config=vars(args),
                    monitor_gym=True,
                    save_code=True,
                )
                run_id_file.write_text(wandb.run.id)
        else:
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                monitor_gym=True,
                save_code=True,
            )
            run_id_file = Path(args.checkpoint_dir) / "wandb_run_id.txt"
            run_id_file.parent.mkdir(parents=True, exist_ok=True)
            run_id_file.write_text(wandb.run.id)

        print(f"WandB 프로젝트: {args.wandb_project}, Run: {run_name}")
        print(
            f"WandB URL: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}"
        )

    # 병렬
    env_fns = [
        make_env(
            args.game_path,
            ports[i],
            args.time_scale,
            args.seed + i,
            args.max_steps,
            obs_builder,
        )
        for i in range(args.n_envs)
    ]

    env = SubprocVecEnv(env_fns)

    # VecNormalize. observation과 reward 정규화
    env = VecNormalize(
        env,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=args.gamma,
        epsilon=1e-8,
    )

    # Callbacks
    callbacks = [
        VecNormalizeCheckpointCallback(
            save_freq=args.save_freq,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_immortal",
            verbose=1,
        )
    ]

    if args.use_wandb:
        callbacks.append(
            WandbCallback(
                model_save_freq=args.save_freq,
                model_save_path=str(checkpoint_dir),
                verbose=2,
            )
        )

    print(f"\n[Training Config]")
    print(
        f"Envs: {args.n_envs} | Steps: {args.n_steps} | Batch: {args.batch_size} | Epochs: {args.n_epochs}"
    )

    # GPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = "cuda"
        print(f"Device: cuda | GPUs: {n_gpus} | Total: {args.total_timesteps:,}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = "cpu"
        print(f"Device: cpu | Total: {args.total_timesteps:,}")

    if args.resume_from and Path(args.resume_from).exists():
        print(f"\nresume from: {args.resume_from}")

        # VecNormalize 통계 복원
        checkpoint_path = Path(args.resume_from)
        if "steps" in checkpoint_path.stem:
            timesteps_str = checkpoint_path.stem.split("_")[-2]
            vec_normalize_path = (
                checkpoint_path.parent / f"vec_normalize_{timesteps_str}_steps.pkl"
            )
            if vec_normalize_path.exists():
                print(f"VecNormalize 통계 복원: {vec_normalize_path}")
                env = VecNormalize.load(str(vec_normalize_path), env)
            else:
                print(f"VecNormalize 파일 없음, 기존 env 사용: {vec_normalize_path}")

        model = PPO.load(
            args.resume_from,
            env=env,
            device=device,
        )
    else:
        print(f"\n새로운 모델 생성")
        model = PPO(
            args.policy,
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            verbose=2,
            device=device,
        )

    print(f"\nStarting training...")
    print("=" * 60)

    # 체크포인트에서 재개 시 timesteps 이어서 진행
    reset_timesteps = not (args.resume_from and Path(args.resume_from).exists())

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=False,
            reset_num_timesteps=reset_timesteps,
        )

        final_path = checkpoint_dir / "ppo_immortal_final.zip"
        model.save(str(final_path))
        print(f"\nsave path: {final_path}")

        vec_normalize_path = checkpoint_dir / "vec_normalize_final.pkl"
        env.save(str(vec_normalize_path))
        print(f"VecNormalize 통계 저장: {vec_normalize_path}")

    finally:
        print("\nexit env...")
        env.close()

        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
