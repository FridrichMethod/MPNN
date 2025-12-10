#!/usr/bin/bash

#SBATCH --time=48:00:00
#SBATCH -p btrippe,stat,hns,owners
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=180GB
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

ml reset

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/groups/btrippe/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/groups/btrippe/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/groups/btrippe/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/groups/btrippe/miniconda3/bin:$PATH"
    fi  
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate /oak/stanford/groups/btrippe/arthur/conda_envs/mpnn

echo "Job started at: $(date)"

cd /scratch/groups/btrippe/arthur/inverse-folding

# HIDDEN_DIM=256
# NUM_LAYERS=4

# iterate through epochs 1, 2, 4, 8, 16, 32, 64, 128, 256

# for EPOCHS in 1 4 16 64 256; do
#     # iterate through learning rates (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
#     for LEARNING_RATE in 1e-2 3e-3 1e-3 3e-4 1e-4 3e-5 1e-5; do
#         python src/training.py --path_for_training_data /scratch/groups/btrippe/arthur/data/mpnn/raw/pdb_2021aug02/ --wandb --wandb_project "pretraining-scaling" --wandb_entity "stanford-protein" --optimizer "adamw" --scheduler "cosine" --learning_rate $LEARNING_RATE --weight_decay 1e-2 --num_epochs $EPOCHS --gradient_norm 1.0 --dropout 0.1 --backbone_noise 0.2 --rescut 3.5 --hidden_dim $HIDDEN_DIM --num_encoder_layers $NUM_LAYERS --num_decoder_layers $NUM_LAYERS --num_neighbors 48 --batch_size 10000 --max_protein_length 10000 --tags bf16,h$HIDDEN_DIM,l$NUM_LAYERS,e$EPOCHS,lr$LEARNING_RATE
#     done
# done

python src/training.py --path_for_training_data /scratch/groups/btrippe/arthur/data/mpnn/raw/pdb_2021aug02/ --wandb --wandb_project pretraining-scaling --wandb_entity stanford-protein --optimizer adamw --scheduler cosine --learning_rate 3e-3 --weight_decay 1e-2 --num_epochs 256 --gradient_norm 1.0 --dropout 0.1 --backbone_noise 0.2 --rescut 3.5 --hidden_dim 128 --num_encoder_layers 3 --num_decoder_layers 3 --num_neighbors 48 --batch_size 10000 --max_protein_length 10000 --tags baseline,bf16,h128,l3,e200,lr0.003,low_precision --low_precision --num_workers 14

python src/training.py --path_for_training_data /scratch/groups/btrippe/arthur/data/mpnn/raw/pdb_2021aug02/ --wandb --wandb_project pretraining-scaling --wandb_entity stanford-protein --optimizer adamw --scheduler cosine --learning_rate 3e-3 --weight_decay 1e-2 --num_epochs 200 --gradient_norm 1.0 --dropout 0.1 --backbone_noise 0.2 --rescut 3.5 --hidden_dim 128 --num_encoder_layers 3 --num_decoder_layers 3 --num_neighbors 48 --batch_size 10000 --max_protein_length 10000 --tags baseline,fp32,h128,l3,e200,lr0.003 --num_workers 14