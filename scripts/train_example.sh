#!/bin/bash

python src/training.py --path_for_training_data ~/data/mpnn/raw/pdb_2021aug02/ --wandb --wandb_project "pretraining-scaling" --wandb_entity "stanford-protein" --run_name "mpnn_original_implementation"