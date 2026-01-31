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
    --checkpoint_dir "./checkpoints_from_20260131" \
    --use_transformer \
    --image_features_dim 96 \
    --vector_features_dim 128 \
    --transformer_d_model 128 \
    --transformer_n_heads 4 \
    --transformer_n_layers 4 \
    --transformer_ffn_dim 256 \
    --transformer_dropout 0.1 \
    --resume_from "/root/libimmortal/checkpoints_from_20260131/ppo_immortal_5040000_steps.zip"