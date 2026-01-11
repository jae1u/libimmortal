import argparse
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

from libimmortal.env import ImmortalSufferingEnv
from libimmortal.utils import find_free_tcp_port, find_n_free_tcp_ports
from libimmortal.utils.reward import ImmortalRewardShaper
from libimmortal.utils.obs_builder import BasicObsBuilder, ArrowObsBuilder


class GymnasiumWrapper(gym.Env):
    def __init__(self, game_path, port, time_scale=2.0, seed=42, max_steps=2000, obs_builder=None):
        super().__init__()
        
        self.env = ImmortalSufferingEnv(
            game_path=game_path,
            port=port,
            time_scale=time_scale,
            seed=seed,
            width=720,
            height=480,
            verbose=False
        )
        
        self.max_steps = max_steps
        self.current_step = 0
        self.obs_builder = obs_builder
        
        # MLAgents는 MultiDiscrete([2 2 2 2 2 2 2 2])를 사용
        # 각 액션이 0 또는 1 
        # 이를 하나의 Discrete space로 변환: 2^8 = 256 가지
        original_action_space = self.env.env.action_space
        
        if hasattr(original_action_space, 'nvec'):
            self.action_dims = original_action_space.nvec
            self.n_actions = int(np.prod(self.action_dims))
            self.action_space = spaces.Discrete(self.n_actions)
            self.is_multi_discrete = True
        else:
            self.action_space = original_action_space
            self.is_multi_discrete = False
        
        # Observation space
        # obs_builder에 따라 observation space 결정
        if obs_builder is not None:
            # obs_builder의 observation space 사용
            temp_obs = self.env.reset()
            built_obs = obs_builder.build(temp_obs)
            
            if isinstance(built_obs, dict):
                obs_spaces = {}
                for key, value in built_obs.items():
                    if key == 'image':
                        obs_spaces[key] = spaces.Box(
                            low=0, high=255, 
                            shape=value.shape, 
                            dtype=value.dtype
                        )
                    elif key == 'vector':
                        obs_spaces[key] = spaces.Box(
                            low=-np.inf, high=np.inf, 
                            shape=value.shape, 
                            dtype=value.dtype
                        )
                self.observation_space = spaces.Dict(obs_spaces)
            else:
                # vector only
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=built_obs.shape, 
                    dtype=built_obs.dtype
                )
        else:
            # raw mode: graphic + vector
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(3, 90, 160), dtype=np.uint8),
                'vector': spaces.Box(low=-np.inf, high=np.inf, shape=(103,), dtype=np.float32),
            })
        
        self.reward_shaper = ImmortalRewardShaper()
        
        self.episode_step = 0
        self.episode_reward = 0
        self.goal_reached = False
        self.enemies_killed_this_ep = 0
        self.min_distance_this_ep = float('inf')
    
    def _discrete_to_multi_discrete(self, action):
        """Discrete action을 MultiDiscrete로 변환"""
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
        """Count alive enemies from vector observation"""
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
            if 'image' in obs:
                return obs['image'], obs.get('vector', np.zeros(103))
            elif 'graphic' in obs:
                return obs['graphic'], obs.get('vector', np.zeros(103))
            else:
                return obs.get('vector', np.zeros(103)), obs.get('vector', np.zeros(103))
        elif isinstance(obs, tuple) and len(obs) == 2:
            return obs
        else:
            return obs[0], obs[1]
    
    def reset(self, seed=None, options=None):
        if self.episode_step > 0:
            print(f"[EP] steps={self.episode_step} reward={self.episode_reward:.2f} "
                  f"goal={self.goal_reached} kills={self.enemies_killed_this_ep} "
                  f"min_dist={self.min_distance_this_ep:.1f}")
        
        raw_obs = self.env.reset()
        
        if self.obs_builder:
            obs = self.obs_builder.build(raw_obs)
            vector_obs = obs.get('vector', obs.get('image', np.zeros(103)))
            graphic_obs = obs.get('image', None)
            
            # obs_builder 사용 시 graphic_obs는 None (ArrowObsBuilder는 vector만 반환)
            if vector_obs.shape[0] >= 103:
                self.reward_shaper.reset(vector_obs[:103], graphic_obs)
                self.prev_enemies = self._count_enemies(vector_obs[:103])
            else:
                self.reward_shaper.reset(vector_obs, graphic_obs)
                self.prev_enemies = self._count_enemies(vector_obs)
        else:
            graphic_obs, vector_obs = self._parse_observation(raw_obs)
            self.reward_shaper.reset(vector_obs, graphic_obs)
            self.prev_enemies = self._count_enemies(vector_obs)
            obs = {'image': graphic_obs, 'vector': vector_obs}
        
        self.current_step = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.goal_reached = False
        self.enemies_killed_this_ep = 0
        self.min_distance_this_ep = float('inf')
        
        return obs, {}
    
    def step(self, action):
        multi_action = self._discrete_to_multi_discrete(action)
        raw_obs, reward, done, info = self.env.step(multi_action)
        
        self.current_step += 1
        self.episode_step += 1
        
        if self.obs_builder:
            obs_dict = self.obs_builder.build(raw_obs)
            
            # vector 추출
            if 'vector' in obs_dict:
                vector_obs = obs_dict['vector']
            elif 'image' in obs_dict:
                vector_obs = np.zeros(103, dtype=np.float32)
            else:
                vector_obs = np.zeros(103, dtype=np.float32)
            
            # obs_builder 사용 시 graphic_obs는 None (ArrowObsBuilder는 vector만 반환)
            graphic_obs = obs_dict.get('image', None)
            
            if vector_obs.shape[0] >= 103:
                vector_for_reward = vector_obs[:103]
            else:
                vector_for_reward = vector_obs
        else:
            graphic_obs, vector_obs = self._parse_observation(raw_obs)
            vector_for_reward = vector_obs
            obs_dict = {'image': graphic_obs, 'vector': vector_obs}
        
        current_distance = float(vector_for_reward[11])
        self.min_distance_this_ep = min(self.min_distance_this_ep, current_distance)
        
        if reward > 0:
            self.goal_reached = True
        
        # Track enemy kills for logging (simplified without reward shaper dependency)
        current_enemies = self._count_enemies(vector_for_reward)
        if hasattr(self, 'prev_enemies'):
            if self.prev_enemies > current_enemies:
                self.enemies_killed_this_ep += (self.prev_enemies - current_enemies)
        self.prev_enemies = current_enemies
        
        truncated = self.current_step >= self.max_steps
        if truncated and not done:
            info['TimeLimit.truncated'] = True
        
        shaped_reward = self.reward_shaper.compute_reward(vector_for_reward, reward, done, truncated, graphic_obs)
        self.episode_reward += shaped_reward
        
        info['goal_distance'] = current_distance
        info['enemies_alive'] = current_enemies
        info['goal_reached'] = self.goal_reached
        
        return obs_dict, shaped_reward, done, truncated, info
    
    def close(self):
        self.env.close()


def make_env(game_path, port, time_scale=2.0, seed=42, max_steps=2000, obs_builder=None):
    def _init():
        env = GymnasiumWrapper(game_path, port, time_scale, seed, max_steps, obs_builder)
        env = Monitor(env) 
        return env
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="SB3 PPO 학습")
    
    parser.add_argument("--game_path", type=str,
                       default="/root/immortal_suffering/immortal_suffering_linux_build.x86_64")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--time_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_timesteps", type=int, default=10000000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_epochs", type=int, default=20, help="각 업데이트당 반복 횟수")
    parser.add_argument("--max_steps", type=int, default=2000, help="에피소드 최대 스텝 (truncate)")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_freq", type=int, default=10000)
    parser.add_argument("--tensorboard_log", type=str, default="./tensorboard_logs")
    parser.add_argument("--policy", type=str, default="MultiInputPolicy", 
                       help="MultiInputPolicy for Dict observations")
    parser.add_argument("--use_wandb", action="store_true", help="Use WandB logging")
    parser.add_argument("--wandb_project", type=str, default="immortal-suffering-sb3")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--obs_type", type=str, default="basic", choices=["basic", "arrow", "raw"])
    
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
    
    # WandB 초기화
    if args.use_wandb:
        run_name = args.wandb_run_name or f"ppo_{args.seed}_{args.learning_rate}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True, 
            monitor_gym=True,
            save_code=True,
        )
        print(f"WandB 초기화. 프로젝트: {args.wandb_project}, Run: {run_name}")
        print(f"WandB URL: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}")
    
    # 병렬 
    env_fns = [make_env(args.game_path, ports[i], args.time_scale, args.seed + i, args.max_steps, obs_builder) 
               for i in range(args.n_envs)]
    
    env = SubprocVecEnv(env_fns)
    
    # VecNormalize: observation과 reward 정규화
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
    
    print("VecNormalize 적용: obs/reward 정규화 활성화")
    
    # Callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_immortal"
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
    print(f"Envs: {args.n_envs} | Steps: {args.n_steps} | Batch: {args.batch_size} | Epochs: {args.n_epochs}")
    
    # GPU 설정
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = 'cuda'
        print(f"Device: cuda | GPUs: {n_gpus} | Total: {args.total_timesteps:,}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = 'cpu'
        print(f"Device: cpu | Total: {args.total_timesteps:,}")
    
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
        tensorboard_log=args.tensorboard_log,
        verbose=2,  
        device=device
    )
    
    print(f"\nStarting training...")
    print(f"Tensorboard: tensorboard --logdir {args.tensorboard_log}")
    print("="*60)
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=False  
        )
        
        final_path = checkpoint_dir / "ppo_immortal_final.zip"
        model.save(str(final_path))
        print(f"\nsave path: {final_path}")
        
        # VecNormalize 통계 저장
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
