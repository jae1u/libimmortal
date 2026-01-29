#!/bin/bash

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU ID: $GPU_ID"

python -m libimmortal.utils.train_sb3 \
    --use_transformer \
    --transformer_vector_only \
    --n_envs 8 \
    --learning_rate 0.0003 \
    --batch_size 1024 \
    --ent_coef 0.01 \
    --transformer_d_model 128 \
    --transformer_n_heads 4 \
    --use_wandb \
    --wandb_project "immortal-suffering-sb3" \