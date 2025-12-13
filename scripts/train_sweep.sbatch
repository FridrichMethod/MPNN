#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=train_sweep_%A_%a.out
#SBATCH --error=train_sweep_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=btrippe,stat,bioe,owners,possu,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_mem:80GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-15

# Define parameter arrays
EDGE_CUTOFFS=(8.0 10.0 15.0 20.0)
USE_VIRTUAL_CENTERS=(false true)
OCCUPANCY_CUTOFFS=(6.0 9.0)

# Calculate indices for the current task
# Total combinations = 4 * 2 * 2 = 16
# Index logic:
# i_edge = task_id % 4
# i_vc = (task_id / 4) % 2
# i_occ = (task_id / 8) % 2

i_edge=$((SLURM_ARRAY_TASK_ID % 4))
i_vc=$(((SLURM_ARRAY_TASK_ID / 4) % 2))
i_occ=$(((SLURM_ARRAY_TASK_ID / 8) % 2))

EDGE_CUTOFF=${EDGE_CUTOFFS[$i_edge]}
USE_VC_BOOL=${USE_VIRTUAL_CENTERS[$i_vc]}
OCCUPANCY_CUTOFF=${OCCUPANCY_CUTOFFS[$i_occ]}

# Construct flags based on boolean
if [ "$USE_VC_BOOL" = true ]; then
    VC_FLAG="--use_virtual_center"
    VC_NAME="vc"
else
    VC_FLAG=""
    VC_NAME="novc"
fi

RUN_NAME="mpnn_edge${EDGE_CUTOFF}_${VC_NAME}_occ${OCCUPANCY_CUTOFF}"

echo "Running task ${SLURM_ARRAY_TASK_ID}: Edge=${EDGE_CUTOFF}, VC=${VC_FLAG}, Occ=${OCCUPANCY_CUTOFF}"
echo "Run name: ${RUN_NAME}"

# Activate environment (adjust if needed based on your system)
source ~/.bashrc
conda activate ./.env/mpnn

python -m mpnn.train \
    --optimizer adamw \
    --scheduler cosine \
    --hidden_dim 128 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --num_neighbors 48 \
    --edge_cutoff "${EDGE_CUTOFF}" \
    ${VC_FLAG} \
    --occupancy_cutoff "${OCCUPANCY_CUTOFF}" \
    --backbone_noise 0.2 \
    --max_protein_length 10000 \
    --rescut 3.5 \
    --num_workers 8 \
    --batch_size 10000 \
    --num_epochs 200 \
    --learning_rate 3e-3 \
    --weight_decay 1e-2 \
    --gradient_norm 1.0 \
    --dropout 0.1 \
    --exclude_membrane \
    --mixed_precision \
    --wandb \
    --wandb_project mpnn-features-sweep \
    --wandb_entity stanford-protein \
    --run_name "${RUN_NAME}"
