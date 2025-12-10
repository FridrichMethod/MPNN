# Inverse Folding

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/) [![CI Status](https://github.com/fridrichmethod/ProteinMPNN/workflows/CI/badge.svg)](https://github.com/fridrichmethod/ProteinMPNN/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
conda create -n mpnn python=3.12 -y
conda activate mpnn
pip install uv
uv pip install -e .[dev,mypy]
```

## Download dataset

There are two urls, one is a small subsample (47MB) at `https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz`. The full dataset (64GB uncompressed) is available at `https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz`.

```bash
cd PATH_TO_DATA_DIR
wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz
tar -xvzf pdb_2021aug02.tar.gz
```

## Training

```bash
python src/training.py --path_for_training_data PATH_TO_DATA_DIR/pdb_2021aug02/ --wandb --wandb_project "pretraining-scaling" --wandb_entity "stanford-protein" --run_name "mpnn_original_implementation"
```
