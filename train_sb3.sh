echo "==================================="
echo " Stable-Baselines3 PPO 학습 시작"
echo "==================================="
echo ""

cd /root/libimmortal
source venv/bin/activate

# GPU 선택 (default: 0)
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU: $GPU_ID"

python -m libimmortal.utils.train_sb3 \
    --total_timesteps 10000000 \
    --n_envs 2 \
    --n_steps 2048 \
    --batch_size 4096 \
    --n_epochs 20 \
    --max_steps 2000 \
    --save_freq 50000 \
    --time_scale 1.0 \
    --seed 42 \
    --obs_type arrow \
    --checkpoint_dir ./checkpoints \
    --use_wandb \
    --wandb_project "immortal-suffering-sb3" \
    --resume_from ./checkpoints/ppo_immortal_600000_steps.zip

echo ""
echo "==================================="
echo " 학습 완료"
echo "==================================="
echo ""
echo "체크포인트: ./checkpoints/"
echo "최종 모델: ./checkpoints/ppo_immortal_final.zip"
echo ""
