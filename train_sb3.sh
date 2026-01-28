#!/bin/bash

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU ID: $GPU_ID"

python -m libimmortal.utils.train_sb3 \
    --use_transformer \
    --transformer_vector_only \
    --n_envs 8 \
    --ent_coef 0.01 \
    --use_wandb \
    --wandb_project "immortal-suffering-sb3" \
    --resume_from  "/root/libimmortal/checkpoints/ppo_immortal_1120000_steps.zip" \