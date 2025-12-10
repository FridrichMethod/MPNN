#!/usr/bin/bash

#SBATCH --time=50:00:00
#SBATCH -p btrippe,stat,hns
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=90GB
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

python src/training.py --path_for_training_data /scratch/groups/btrippe/arthur/data/mpnn/raw/pdb_2021aug02/ --wandb --wandb_project "pretraining-scaling" --wandb_entity "stanford-protein" --run_name "sensitivity_lr1e-3" --optimizer "adamw" --scheduler "cosine" --learning_rate 1e-3 --weight_decay 1e-2 --num_epochs 200 --gradient_norm 1.0 --dropout 0.0 --backbone_noise 0.2 --rescut 3.5 --hidden_dim 128 --num_encoder_layers 3 --num_decoder_layers 3 --num_neighbors 48 --batch_size 10000 --max_protein_length 10000


python src/training.py --path_for_training_data /scratch/groups/btrippe/arthur/data/mpnn/raw/pdb_2021aug02/ --wandb --wandb_project "pretraining-scaling" --wandb_entity "stanford-protein" --run_name "sensitivity_lr1e-5" --optimizer "adamw" --scheduler "cosine" --learning_rate 1e-5 --weight_decay 1e-2 --num_epochs 200 --gradient_norm 1.0 --dropout 0.0 --backbone_noise 0.2 --rescut 3.5 --hidden_dim 128 --num_encoder_layers 3 --num_decoder_layers 3 --num_neighbors 48 --batch_size 10000 --max_protein_length 10000
