#!/bin/bash

PROJECT_ROOT_DIR=$(git rev-parse --show-toplevel)

python -m mpnn.training \
    --path_for_training_data "$PROJECT_ROOT_DIR"/datasets/pdb_2021aug02_sample/ \
    --optimizer adamw \
    --scheduler cosine \
    --hidden_dim 128 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --num_neighbors 48 \
    --max_protein_length 10000 \
    --backbone_noise 0.2 \
    --rescut 3.5 \
    --num_workers 8 \
    --batch_size 10000 \
    --num_epochs 2 \
    --learning_rate 3e-3 \
    --weight_decay 1e-2 \
    --gradient_norm 1.0 \
    --dropout 0.1 \
    --exclude_membrane \
    --mixed_precision \
    --megascale_split_path "$PROJECT_ROOT_DIR"/datasets/megascale/mega_splits.pkl \
    --megascale_pdb_dir "$PROJECT_ROOT_DIR"/datasets/megascale/AlphaFold_model_PDBs \
    --megascale_csv "$PROJECT_ROOT_DIR"/datasets/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
    --fsd_thermo_csv "$PROJECT_ROOT_DIR"/datasets/FSD/fsd_thermo.csv \
    --fsd_thermo_pdb_dir "$PROJECT_ROOT_DIR"/datasets/FSD/PDBs \
    --fsd_thermo_cache_path "$PROJECT_ROOT_DIR"/datasets/FSD/fsd_thermo.pkl \
    # --wandb \
    # --wandb_project mpnn-features \
    # --wandb_entity stanford-protein
