echo "==================================="
echo " Stable-Baselines3 PPO 학습 시작"
echo "==================================="
echo ""

cd /root/libimmortal
source venv/bin/activate

# GPU 2개 사용 설정
export CUDA_VISIBLE_DEVICES=0,1

python -m libimmortal.utils.train_sb3 \
    --total_timesteps 10000000 \
    --n_envs 4 \
    --n_steps 2048 \
    --batch_size 8192 \
    --n_epochs 20 \
    --max_steps 2000 \
    --save_freq 100000 \
    --time_scale 1.0 \
    --seed 42 \
    --obs_type arrow \
    --checkpoint_dir ./checkpoints \
    --tensorboard_log ./tensorboard_logs \
    --use_wandb \
    --wandb_project "immortal-suffering-sb3"

echo ""
echo "==================================="
echo " 학습 완료"
echo "==================================="
echo ""
echo "체크포인트: ./checkpoints/"
echo "최종 모델: ./checkpoints/ppo_immortal_final.zip"
echo ""
