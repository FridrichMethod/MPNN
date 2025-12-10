import argparse
import copy

from mpnn.training import train

LR_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
EPOCHS_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 1024, 2048, 4096]
WD_GRID = [0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 0.4, 0.8]


def get_neighbors_sweep(args):
    current_lr = args.learning_rate
    current_epochs = args.num_epochs
    current_wd = args.weight_decay
    # find the higher and lower than current_lr and current_epochs
    higher_lr = min(LR_GRID, key=lambda x: x if x > current_lr else float("inf"))
    lower_lr = max(LR_GRID, key=lambda x: x if x < current_lr else -float("inf"))
    higher_epochs = min(EPOCHS_GRID, key=lambda x: x if x > current_epochs else float("inf"))
    lower_epochs = max(EPOCHS_GRID, key=lambda x: x if x < current_epochs else -float("inf"))
    higher_wd = min(WD_GRID, key=lambda x: x if x > current_wd else float("inf"))
    lower_wd = max(WD_GRID, key=lambda x: x if x < current_wd else -float("inf"))

    higher_lr_args = copy.deepcopy(args)
    higher_lr_args.learning_rate = higher_lr
    lower_lr_args = copy.deepcopy(args)
    lower_lr_args.learning_rate = lower_lr
    higher_epochs_args = copy.deepcopy(args)
    higher_epochs_args.num_epochs = higher_epochs
    lower_epochs_args = copy.deepcopy(args)
    lower_epochs_args.num_epochs = lower_epochs
    higher_wd_args = copy.deepcopy(args)
    higher_wd_args.weight_decay = higher_wd
    lower_wd_args = copy.deepcopy(args)
    lower_wd_args.weight_decay = lower_wd

    print(f"Current LR: {current_lr}, Higher LR: {higher_lr}, Lower LR: {lower_lr}")
    print(
        f"Current Epochs: {current_epochs}, Higher Epochs: {higher_epochs}, Lower Epochs: {lower_epochs}"
    )
    print(f"Current Weighr Decay: {current_wd}, Higher WD: {higher_wd}, Lower WD: {lower_wd}")
    return (
        higher_lr_args,
        lower_lr_args,
        higher_epochs_args,
        lower_epochs_args,
        higher_wd_args,
        lower_wd_args,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument(
        "--path_for_training_data",
        type=str,
        default="my_path/pdb_2021aug02",
        help="path for loading training data",
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        default="./training_output",
        help="path for logs and model weights",
    )
    argparser.add_argument(
        "--previous_checkpoint",
        type=str,
        default="",
        help="path for previous model weights, e.g. file.pt",
    )
    argparser.add_argument(
        "--num_epochs", type=int, default=200, help="number of epochs to train for"
    )
    argparser.add_argument(
        "--save_model_every_n_epochs",
        type=int,
        default=10,
        help="save model weights every n epochs",
    )
    argparser.add_argument(
        "--num_examples_per_epoch",
        type=int,
        default=1000000,
        help="number of training example to load for one epoch",
    )
    argparser.add_argument("--optimizer", type=str, default="adamw", help="optimizer to use")
    argparser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    argparser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    argparser.add_argument("--scheduler", type=str, default="", help="scheduler to use")
    argparser.add_argument(
        "--batch_size", type=int, default=10000, help="number of tokens for one batch"
    )
    argparser.add_argument(
        "--max_protein_length",
        type=int,
        default=10000,
        help="maximum length of the protein complext",
    )
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument(
        "--num_encoder_layers", type=int, default=3, help="number of encoder layers"
    )
    argparser.add_argument(
        "--num_decoder_layers", type=int, default=3, help="number of decoder layers"
    )
    argparser.add_argument(
        "--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph"
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout"
    )
    argparser.add_argument(
        "--backbone_noise",
        type=float,
        default=0.2,
        help="amount of noise added to backbone during training",
    )
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument(
        "--gradient_norm",
        type=float,
        default=-1.0,
        help="clip gradient norm, set to negative to omit clipping",
    )
    argparser.add_argument(
        "--mixed_precision", action="store_true", help="train with mixed precision"
    )
    argparser.add_argument("--wandb", action="store_true", help="use wandb for logging")
    argparser.add_argument(
        "--wandb_entity", type=str, default="stanford-protein", help="wandb entity"
    )
    argparser.add_argument("--run_name", type=str, default="", help="wandb run name")
    argparser.add_argument(
        "--wandb_project", type=str, default="pretraining-scaling", help="wandb project"
    )
    argparser.add_argument(
        "--num_workers", type=int, default=12, help="number of workers for data loading"
    )
    argparser.add_argument(
        "--reload_data_every_n_epochs",
        type=int,
        default=5,
        help="reload training data every n epochs",
    )
    argparser.add_argument("--code_version", type=str, default="nov20", help="code version")
    argparser.add_argument(
        "--force_rerun",
        action="store_true",
        help="force rerun of the model even if a training run already exists.",
    )
    argparser.add_argument("--seed", type=int, default=42, help="random seed")
    argparser.add_argument(
        "--validate_every_n_epochs", type=int, default=5, help="validate every n epochs"
    )
    argparser.add_argument("--tags", type=lambda x: x.split(","), default=[], help="tags for wandb")
    argparser.add_argument(
        "--exclude_membrane",
        action="store_true",
        help="exclude transmembrane proteins from training (soluble MPNN)",
    )
    argparser.add_argument("--debug", action="store_true", help="debug mode")
    argparser.add_argument(
        "--megascale_pdb_dir",
        type=str,
        default="/data/megascale/AlphaFold_model_PDBs",
        help="path for megascale PDBs",
    )
    argparser.add_argument(
        "--megascale_csv",
        type=str,
        default="/data/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv",
        help="path for megascale CSV",
    )
    argparser.add_argument(
        "--fsd_thermo_csv",
        type=str,
        default="data/FSD/fsd_thermo.csv",
        help="path for FSD thermo CSV",
    )
    argparser.add_argument(
        "--fsd_thermo_pdb_dir", type=str, default="data/FSD/PDBs", help="path for FSD thermo PDBs"
    )
    argparser.add_argument(
        "--fsd_thermo_cache_path",
        type=str,
        default="cache/fsd_thermo.pkl",
        help="path for FSD thermo cache",
    )
    argparser.add_argument(
        "--partition",
        type=int,
        default=0,
        help="0: args, 1: higher_lr, 2: lower_lr, 3: higher_epochs, 4: lower_epochs, 5: higher_wd, 6: lower_ wd",
    )
    args = argparser.parse_args()
    (
        higher_lr_args,
        lower_lr_args,
        higher_epochs_args,
        lower_epochs_args,
        higher_wd_args,
        lower_wd_args,
    ) = get_neighbors_sweep(args)
    if args.partition == 0:
        train(args)
    elif args.partition == 1:
        train(higher_lr_args)
    elif args.partition == 2:
        train(lower_lr_args)
    elif args.partition == 3:
        train(higher_epochs_args)
    elif args.partition == 4:
        train(lower_epochs_args)
    elif args.partition == 5:
        train(higher_wd_args)
    elif args.partition == 6:
        train(lower_wd_args)
