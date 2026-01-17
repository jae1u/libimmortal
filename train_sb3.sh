#!/bin/bash

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU ID: $GPU_ID"

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
