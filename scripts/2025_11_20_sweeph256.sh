#!/usr/bin/bash

#SBATCH --time=72:00:00
#SBATCH -p btrippe,stat,hns
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=120GB
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --array=0-4

ml reset
ml gcc/14.2

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

HIDDEN_DIM=256
NUM_LAYERS=4
PARTITION_ID=${SLURM_ARRAY_TASK_ID}
# iterate through epochs 1, 2, 4, 8, 16, 32, 64, 128, 256


python -m mpnn.sweep_neighbors \
  --path_for_training_data /scratch/groups/btrippe/arthur/data/mpnn/raw/pdb_2021aug02/ \
  --wandb --wandb_project pretraining-scaling --wandb_entity stanford-protein \
  --optimizer adamw --scheduler cosine --learning_rate 1e-3 --weight_decay 1e-2 \
  --num_epochs 192 --gradient_norm 1.0 --dropout 0.1 --backbone_noise 0.2 \
  --rescut 3.5 --hidden_dim $HIDDEN_DIM \
  --num_encoder_layers $NUM_LAYERS --num_decoder_layers $NUM_LAYERS \
  --num_neighbors 48 --batch_size 10000 --max_protein_length 5000 \
  --seed 0 --exclude_membrane --mixed_precision \
  --tags soluble,max5000,mixed_precision \
  --megascale_pdb_dir /home/groups/btrippe/datasets/megascale/AlphaFold_model_PDBs \
  --megascale_csv /home/groups/btrippe/datasets/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
  --partition $PARTITION_ID