#!/bin/bash

python -m mpnn.train \
    --path_for_training_data ./datasets/train/pdb_2021aug02_sample \
    --optimizer adamw \
    --scheduler cosine \
    --hidden_dim 128 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --num_neighbors 48 \
    --backbone_noise 0.2 \
    --max_protein_length 10000 \
    --rescut 3.5 \
    --num_workers 6 \
    --batch_size 2000 \
    --num_epochs 200 \
    --learning_rate 3e-3 \
    --weight_decay 1e-2 \
    --gradient_norm 1.0 \
    --dropout 0.1 \
    --exclude_membrane \
    --mixed_precision \
    --force_rerun \
    --wandb \
    --wandb_project mpnn-features \
    --wandb_entity stanford-protein \
    --run_name baseline
