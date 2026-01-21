import argparse
from pathlib import Path
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

from libimmortal.immortal_gym_env import ImmortalGymEnv
from libimmortal.utils import find_n_free_tcp_ports


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
                getattr(self.training_env, "save")(stats_path)
                if self.verbose > 0:
                    print(f"Saving VecNormalize stats to {stats_path}")

        return True


def make_env(**kwargs):
    def _init() -> gym.Env:
        env = ImmortalGymEnv(**kwargs)
        env = Monitor(env)
        return env

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SB3 PPO 학습")

    # fmt: off
    parser.add_argument("--game_path", type=str, default="../immortal_suffering/immortal_suffering_linux_build.x86_64")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--time_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=2000, help="에피소드 최대 스텝 (truncate)")
    parser.add_argument("--no-filter-observation", action="store_true")

    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--total_timesteps", type=int, default=10000000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=20, help="각 업데이트당 반복 횟수")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_freq", type=int, default=10000)
    parser.add_argument("--policy", type=str, default="MultiInputPolicy", help="MultiInputPolicy for Dict observations")
    parser.add_argument("--use_wandb", action="store_true", help="Use WandB logging")
    parser.add_argument("--wandb_project", type=str, default="immortal-suffering-sb3")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    # fmt: on

    return parser.parse_args()


def init_ports(args: argparse.Namespace) -> list[int]:
    if args.port is None:
        ports = find_n_free_tcp_ports(args.n_envs)
    else:
        ports = [args.port + i for i in range(args.n_envs)]
    return ports


def init_checkpoint_dir(args: argparse.Namespace) -> Path:
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def wandb_init(args: argparse.Namespace, checkpoint_dir: Path):
    if not args.use_wandb:
        return
    default_name = f"ppo_{args.seed}_{args.learning_rate}"
    run_name = args.wandb_run_name or default_name
    run_id_file = checkpoint_dir / "wandb_run_id.txt"

    run_id = None
    resume_mode = None
    if args.resume_from and Path(args.resume_from).exists() and run_id_file.exists():
        run_id = run_id_file.read_text().strip()
        resume_mode = "allow"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        id=run_id,
        resume=resume_mode,
        config=vars(args),
        monitor_gym=True,
        save_code=True,
    )
    assert wandb.run is not None

    if run_id is None:
        run_id_file.write_text(wandb.run.id)

    print(f"WandB Project: {args.wandb_project}, Run: {run_name}")
    print(f"WandB URL: {wandb.run.get_url()}")


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


def get_callbacks(args: argparse.Namespace, checkpoint_dir: Path) -> list[BaseCallback]:
    callbacks: list[BaseCallback] = [
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
    return callbacks


def main():
    args = parse_args()

    ports = init_ports(args)
    checkpoint_dir = init_checkpoint_dir(args)
    wandb_init(args, checkpoint_dir)

    env = SubprocVecEnv(get_env_fns(args, ports))
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

    callbacks = get_callbacks(args, checkpoint_dir)

    print(f"\n[Training Config]")
    print(
        f"Envs: {args.n_envs} | Steps: {args.n_steps} | Batch: {args.batch_size} | Epochs: {args.n_epochs}"
    )

    assert torch.cuda.is_available()
    n_gpus = torch.cuda.device_count()
    device = "cuda"
    print(f"Device: cuda | GPUs: {n_gpus} | Total: {args.total_timesteps:,}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if args.resume_from and Path(args.resume_from).exists():
        print(f"resume from: {args.resume_from}")

        checkpoint_path = Path(args.resume_from)
        if "steps" in checkpoint_path.stem:
            timesteps_str = checkpoint_path.stem.split("_")[-2]
            vec_normalize_path = (
                checkpoint_path.parent / f"vec_normalize_{timesteps_str}_steps.pkl"
            )
            assert vec_normalize_path.exists()
            print(f"VecNormalize 통계 복원: {vec_normalize_path}")
            env = VecNormalize.load(str(vec_normalize_path), env)

        model = PPO.load(
            args.resume_from,
            env=env,
            device=device,
        )
    else:
        print(f"새로운 모델 생성")
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

    print(f"Starting training...")
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
        print(f"save path: {final_path}")

        vec_normalize_path = checkpoint_dir / "vec_normalize_final.pkl"
        env.save(str(vec_normalize_path))
        print(f"VecNormalize 통계 저장: {vec_normalize_path}")
    finally:
        print("exit env...")
        env.close()
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
