import argparse
import gc
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.types import Device
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from mpnn.data import (
    MegascaleDataset,
    PDBDataset,
    PDBDatasetFlattened,
    StructureDataset,
    StructureLoader,
    ThermoMutDBDataset,
)
from mpnn.data.data_utils import (
    build_training_clusters,
    loader_pdb,
)
from mpnn.env import (
    DEFAULT_TRAIN_DATA_PATH,
    DEFAULT_TRAIN_OUTPUT_DIR,
    EXCLUDED_PDBS_CSV,
    FSD_THERMO_CACHE_PATH,
    FSD_THERMO_CSV,
    FSD_THERMO_PDB_DIR,
    MEGASCALE_CSV,
    MEGASCALE_PDB_DIR,
    MEGASCALE_SPLIT_PATH,
)
from mpnn.finetune import validation_step
from mpnn.models import EnergyMPNN, ProteinMPNN
from mpnn.typing_utils import StrPath
from mpnn.utils import enable_tf32_if_available


def loss_nll(S, log_probs, mask):
    """Negative log probabilities"""
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs, -1)  # [B, L]
    true_false = S_argmaxed == S
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """Negative log probabilities"""
    S_onehot = torch.nn.functional.one_hot(S, 21).to(dtype=log_probs.dtype)

    # Label smoothing
    S_onehot = S_onehot + weight / S_onehot.size(-1)
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0  # fixed
    return loss, loss_av


def worker_init_fn(worker_id):
    np.random.seed(None)
    random.seed(None)
    torch_seed = np.random.randint(0, 2**32 - 1)
    torch.manual_seed(torch_seed)


def get_run_name(args):
    return f"{args.code_version}_h{args.hidden_dim}_l{args.num_encoder_layers}_n{args.num_neighbors}_lr{args.learning_rate}_e{args.num_epochs}_wd{args.weight_decay}_bs{args.batch_size}_bb{args.backbone_noise}_s{args.seed}"


def setup_run(args):
    run_name, tags = None, None
    if args.run_name:
        run_name = args.run_name
        tags = args.tags if args.tags else [args.code_version]
    else:
        run_name = get_run_name(args)
        tags = args.tags
        tags.extend(run_name.split("_"))

    os.makedirs(args.output_dir, exist_ok=True)

    base_folder = os.path.join(args.output_dir, run_name)
    os.makedirs(base_folder, exist_ok=True)

    logfile = os.path.join(base_folder, "log.txt")

    with open(logfile, "w") as f:
        f.write("Epoch\tTrain\tValidation\n")

    return run_name, tags, base_folder, logfile


def load_pdb_data(data_path: StrPath, args: argparse.Namespace):
    params = {
        "LIST": os.path.join(data_path, "list.csv"),
        "VAL": os.path.join(data_path, "valid_clusters.txt"),
        "TEST": os.path.join(data_path, "test_clusters.txt"),
        "DIR": data_path,
        "DATCUT": "2030-Jan-01",
        "RESCUT": args.rescut,  # resolution cutoff for PDBs
        "HOMO": 0.70,  # min seq.id. to detect homo chains
    }

    def collate_passthrough_filter(batch):  # batch_size=1
        if not batch[0]:
            return None
        return batch[0]

    excluded_pdbs = []
    if args.exclude_membrane:
        excluded_pdbs = pd.read_csv(EXCLUDED_PDBS_CSV)["PDB_IDS"].tolist()

    LOAD_PARAM = {
        "batch_size": 1,
        "shuffle": True,
        "pin_memory": False,
        "collate_fn": collate_passthrough_filter,
        "persistent_workers": True,
        "num_workers": min(args.num_workers, os.cpu_count()),
    }

    print("building training clusters")
    train, valid, _ = build_training_clusters(params)

    print("loading datasets")

    train_clusters = PDBDataset(list(train.keys()), loader_pdb, train, params)
    print(f"number of training clusters: {len(train_clusters)}")
    train_cluster_loader = torch.utils.data.DataLoader(
        train_clusters, worker_init_fn=worker_init_fn, **LOAD_PARAM
    )

    valid_clusters = PDBDatasetFlattened(list(valid.keys()), loader_pdb, valid, params)
    print(f"number of validation clusters: {len(valid_clusters)}")
    valid_cluster_loader = torch.utils.data.DataLoader(
        valid_clusters, worker_init_fn=worker_init_fn, **LOAD_PARAM
    )

    pdb_dict_train = []
    skipped_excluded = 0
    for x in tqdm(
        train_cluster_loader, total=len(train_cluster_loader), desc="Loading training data"
    ):
        if x is None:
            continue
        if x["name"].split("_")[0] in excluded_pdbs:
            skipped_excluded += 1
            continue
        pdb_dict_train.append(x)
    print(f"Skipped {skipped_excluded} excluded PDBs")

    pdb_dict_valid = []
    for x in tqdm(
        valid_cluster_loader, total=len(valid_cluster_loader), desc="Loading validation data"
    ):
        if x is not None:
            pdb_dict_valid.append(x)

    if args.max_protein_length > args.batch_size:
        args.max_protein_length = args.batch_size
        print(
            "max_protein_length must be less than batch_size. Reducing max_protein_length to batch_size."
        )

    pdb_dataset_train = StructureDataset(
        pdb_dict_train, truncate=None, max_length=args.max_protein_length
    )
    pdb_dataset_valid = StructureDataset(
        pdb_dict_valid, truncate=None, max_length=args.max_protein_length
    )

    pdb_loader_train = StructureLoader(pdb_dataset_train, batch_size=args.batch_size)
    pdb_loader_valid = StructureLoader(pdb_dataset_valid, batch_size=args.batch_size)

    return pdb_loader_train, pdb_loader_valid, train_cluster_loader, excluded_pdbs


def get_model_and_optimizer(args, device: Device, total_steps):
    model = ProteinMPNN(
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_encoder_layers,
        num_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
    )
    model.to(device)

    print("Total parameters: ", sum(p.numel() for p in model.parameters()))

    epoch = 0
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    warmup_steps = int(0.01 * total_steps)
    warmup = LinearLR(optimizer, start_factor=1e-8, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0.0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    if args.previous_checkpoint:
        checkpoint = torch.load(args.previous_checkpoint)
        epoch = checkpoint["epoch"]  # write epoch from the checkpoint
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return model, optimizer, scheduler, epoch


@torch.inference_mode()
def eval_pretrained_mpnn(
    pretrained_model,
    batch_size=10000,
    device: Device = "cuda",
    mc_samples=20,
    backbone_noise=0.2,
    megascale_split_path=MEGASCALE_SPLIT_PATH,
    megascale_pdb_dir=MEGASCALE_PDB_DIR,
    megascale_csv=MEGASCALE_CSV,
    fsd_thermo_csv=FSD_THERMO_CSV,
    fsd_thermo_pdb_dir=FSD_THERMO_PDB_DIR,
    fsd_thermo_cache_path=FSD_THERMO_CACHE_PATH,
):

    pretrained_model.eval()
    model = EnergyMPNN(
        protein_mpnn=pretrained_model,
        use_antithetic_variates=True,
        noise_level=backbone_noise,
        device=device,
    )
    model.eval()

    megascale_train = MegascaleDataset(
        csv_path=megascale_csv,
        pdb_dir=megascale_pdb_dir,
        split_path=megascale_split_path,
        split="train",
    )
    megascale_valid = MegascaleDataset(
        csv_path=megascale_csv,
        pdb_dir=megascale_pdb_dir,
        split_path=megascale_split_path,
        split="val",
    )
    megascale_test = MegascaleDataset(
        csv_path=megascale_csv,
        pdb_dir=megascale_pdb_dir,
        split_path=megascale_split_path,
        split="test",
    )
    fsd_thermo_train = ThermoMutDBDataset(
        csv_path=fsd_thermo_csv,
        pdb_dir=fsd_thermo_pdb_dir,
        pdb_dict_cache_path=fsd_thermo_cache_path,
        cif=False,
    )

    train_metrics = validation_step(
        model,
        megascale_train,
        batch_size=batch_size,
        name="train",
        device=device,
        mc_samples=mc_samples,
    )
    valid_metrics = validation_step(
        model,
        megascale_valid,
        batch_size=batch_size,
        name="valid",
        device=device,
        mc_samples=mc_samples,
    )
    test_metrics = validation_step(
        model,
        megascale_test,
        batch_size=batch_size,
        name="test",
        device=device,
        mc_samples=mc_samples,
    )
    fsd_thermo_metrics = validation_step(
        model,
        fsd_thermo_train,
        batch_size=batch_size,
        name="fsd_thermo",
        device=device,
        mc_samples=mc_samples,
    )

    # convert all metric values to floating point
    train_metrics = {k: float(v) for k, v in train_metrics.items()}
    valid_metrics = {k: float(v) for k, v in valid_metrics.items()}
    test_metrics = {k: float(v) for k, v in test_metrics.items()}
    fsd_thermo_metrics = {k: -1 * float(v) for k, v in fsd_thermo_metrics.items()}
    return train_metrics, valid_metrics, test_metrics, fsd_thermo_metrics


def train(args):
    if args.seed is not None:
        seed_everything(args.seed)

    if enable_tf32_if_available():
        print(f"TF32 is enabled. Precision: {torch.get_float32_matmul_precision()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name, tags, base_folder, logfile = setup_run(args)
    if (
        os.path.exists(os.path.join(base_folder, "model_weights", "last_epoch.pt"))
        and not args.force_rerun
    ):
        print(f"Training run already exists for {run_name}. Use --force_rerun to rerun the model.")
        return

    pdb_loader_train, pdb_loader_valid, train_cluster_loader, excluded_pdbs = load_pdb_data(
        args.path_for_training_data, args
    )
    model, optimizer, scheduler, epoch = get_model_and_optimizer(
        args, device, total_steps=len(pdb_loader_train) * args.num_epochs
    )

    if args.wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config=args.__dict__,
            tags=tags,
        )
        wandb.log({"Total parameters": sum(p.numel() for p in model.parameters())})

    total_step = 0
    for e in tqdm(range(args.num_epochs), desc="Epoch"):
        t0 = time.time()
        e = epoch + e
        model.train()
        train_sum, train_weights = 0.0, 0.0
        train_acc = 0.0
        skipped_batches = 0

        for batch_idx, data in tqdm(
            enumerate(pdb_loader_train), total=len(pdb_loader_train), desc="Training Batch"
        ):
            try:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                data = data.to(device)  # type: ignore
                mask_for_loss = (data.mask * data.chain_mask_all).unsqueeze(0)
                S = data.chain_seq_label.unsqueeze(0)

                if args.mixed_precision:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        log_probs = model(
                            data.x,
                            data.chain_seq_label,
                            data.mask,
                            data.chain_mask_all,
                            data.residue_idx,
                            data.chain_encoding_all,
                            data.batch,
                        )
                        log_probs_3d = log_probs.unsqueeze(0)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs_3d, mask_for_loss)
                else:
                    log_probs = model(
                        data.x,
                        data.chain_seq_label,
                        data.mask,
                        data.chain_mask_all,
                        data.residue_idx,
                        data.chain_encoding_all,
                        data.batch,
                    )
                    log_probs_3d = log_probs.unsqueeze(0)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs_3d, mask_for_loss)

                loss_av_smoothed.backward()

                if args.gradient_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                optimizer.step()
                scheduler.step()

                loss, _, true_false = loss_nll(S, log_probs_3d, mask_for_loss)

                train_sum += torch.sum(loss * mask_for_loss).float().cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).float().cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).float().cpu().data.numpy()

                total_step += 1

            except RuntimeError as err:
                if "out of memory" in str(err).lower() or "CUDA out of memory" in str(err):
                    skipped_batches += 1
                    print(
                        f"WARNING: OOM at step {total_step}, batch {batch_idx}. Skipping batch. ({skipped_batches} skipped so far)"
                    )
                    # Clear cache and free memory
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    # Re-raise if it's not an OOM error
                    skipped_batches += 1
                    print(
                        f"WARNING: Error at step {total_step}, batch {batch_idx}. Skipping batch. ({skipped_batches} skipped so far)"
                    )
                    print(err)
                    continue

        train_loss = train_sum / train_weights
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)

        train_perplexity_ = np.format_float_positional(
            np.float32(train_perplexity), unique=False, precision=3
        )

        train_accuracy_ = np.format_float_positional(
            np.float32(train_accuracy), unique=False, precision=3
        )

        if (e + 1) == 1 or (e + 1) % args.validate_every_n_epochs == 0 or e == args.num_epochs - 1:
            model.eval()
            validation_sum, validation_weights = 0.0, 0.0
            validation_acc = 0.0
            for _, data in tqdm(
                enumerate(pdb_loader_valid), total=len(pdb_loader_valid), desc="Validation Batch"
            ):
                data = data.to(device)  # type: ignore
                S = data.chain_seq_label.unsqueeze(0)
                mask_for_loss = (data.mask * data.chain_mask_all).unsqueeze(0)
                with torch.inference_mode():
                    log_probs = model(
                        data.x,
                        data.chain_seq_label,
                        data.mask,
                        data.chain_mask_all,
                        data.residue_idx,
                        data.chain_encoding_all,
                        data.batch,
                    )
                    log_probs_3d = log_probs.unsqueeze(0)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs_3d, mask_for_loss)

                loss, _, true_false = loss_nll(S, log_probs_3d, mask_for_loss)

                validation_sum += torch.sum(loss * mask_for_loss).float().cpu().data.numpy()
                validation_acc += torch.sum(true_false * mask_for_loss).float().cpu().data.numpy()
                validation_weights += torch.sum(mask_for_loss).float().cpu().data.numpy()

            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)

            validation_perplexity_ = np.format_float_positional(
                np.float32(validation_perplexity), unique=False, precision=3
            )
            validation_accuracy_ = np.format_float_positional(
                np.float32(validation_accuracy), unique=False, precision=3
            )
            if args.wandb:
                wandb.log(
                    {
                        "validation_perplexity": float(validation_perplexity_),
                        "validation_accuracy": float(validation_accuracy_),
                    },
                    step=total_step,
                )
                if e == args.num_epochs - 1:
                    wandb.log(
                        {
                            "final_validation_perplexity": float(validation_perplexity_),
                            "final_validation_accuracy": float(validation_accuracy_),
                        },
                        step=total_step,
                    )
                    with torch.inference_mode():
                        train_metrics, valid_metrics, test_metrics, fsd_thermo_metrics = (
                            eval_pretrained_mpnn(
                                model,
                                batch_size=10000,
                                device=device,
                                mc_samples=20,
                                backbone_noise=args.backbone_noise,
                                megascale_split_path=args.megascale_split_path,
                                megascale_pdb_dir=args.megascale_pdb_dir,
                                megascale_csv=args.megascale_csv,
                                fsd_thermo_csv=args.fsd_thermo_csv,
                                fsd_thermo_pdb_dir=args.fsd_thermo_pdb_dir,
                                fsd_thermo_cache_path=args.fsd_thermo_cache_path,
                            )
                        )
                    wandb.log(
                        {
                            **train_metrics,
                            **valid_metrics,
                            **test_metrics,
                            **fsd_thermo_metrics,
                        },
                        step=total_step,
                    )

        t1 = time.time()
        dt = np.format_float_positional(np.float32(t1 - t0), unique=False, precision=1)
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        with open(logfile, "a") as f:
            f.write(
                f"epoch: {e + 1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}, skipped_batches: {skipped_batches}\n"
            )
        if args.wandb:
            wandb.log(
                {
                    "epoch": e + 1,
                    "step": total_step,
                    "time": float(dt),
                    "train_perplexity": float(train_perplexity_),
                    "train_accuracy": float(train_accuracy_),
                    "learning_rate": optimizer.param_groups[0]["lr"]
                    if args.optimizer == "adamw"
                    else optimizer.rate(total_step),
                    "skipped_batches": skipped_batches,
                },
                step=total_step,
            )

        print(
            f"epoch: {e + 1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, skipped_batches: {skipped_batches}"
        )

        # save model weights
        optimizer_state_dict = optimizer.state_dict()
        model_info = {
            "epoch": e + 1,
            "step": total_step,
            "num_edges": args.num_neighbors,
            "noise_level": args.backbone_noise,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler.state_dict() if args.scheduler == "cosine" else None,
        }

        if not os.path.exists(os.path.join(base_folder, "model_weights")):
            os.makedirs(os.path.join(base_folder, "model_weights"))
        checkpoint_filename_last = os.path.join(base_folder, "model_weights", "last_epoch.pt")
        torch.save(model_info, checkpoint_filename_last)

        if (e + 1) % args.save_model_every_n_epochs == 0:
            checkpoint_filename = os.path.join(
                base_folder, "model_weights", f"epoch_{e + 1}_step_{total_step}.pt"
            )
            torch.save(model_info, checkpoint_filename)

        if (e + 1) % args.reload_data_every_n_epochs == 0:
            print(f"Reloading training data at epoch {e + 1}...")
            pdb_dict_train = []
            skipped_excluded = 0
            for x in tqdm(
                train_cluster_loader, total=len(train_cluster_loader), desc="Loading training data"
            ):
                if x is None:
                    continue
                if x["name"].split("_")[0] in excluded_pdbs:
                    skipped_excluded += 1
                    continue
                pdb_dict_train.append(x)
            print(f"Skipped {skipped_excluded} excluded PDBs")
            pdb_dataset_train = StructureDataset(
                pdb_dict_train, truncate=None, max_length=args.max_protein_length
            )
            pdb_loader_train = StructureLoader(pdb_dataset_train, batch_size=args.batch_size)

        if e == args.num_epochs - 1:
            print("Training complete. Saving model weights...")
            checkpoint_filename = os.path.join(base_folder, "model_weights", "final_epoch.pt")
            torch.save(model_info, checkpoint_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument(
        "--path_for_training_data",
        type=str,
        default=DEFAULT_TRAIN_DATA_PATH,
        help="path for loading training data",
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_TRAIN_OUTPUT_DIR,
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
        "--megascale_split_path",
        type=str,
        default=MEGASCALE_SPLIT_PATH,
        help="path for megascale split",
    )
    argparser.add_argument(
        "--megascale_pdb_dir",
        type=str,
        default=MEGASCALE_PDB_DIR,
        help="path for megascale PDBs",
    )
    argparser.add_argument(
        "--megascale_csv",
        type=str,
        default=MEGASCALE_CSV,
        help="path for megascale CSV",
    )
    argparser.add_argument(
        "--fsd_thermo_csv",
        type=str,
        default=FSD_THERMO_CSV,
        help="path for FSD thermo CSV",
    )
    argparser.add_argument(
        "--fsd_thermo_pdb_dir",
        type=str,
        default=FSD_THERMO_PDB_DIR,
        help="path for FSD thermo PDBs",
    )
    argparser.add_argument(
        "--fsd_thermo_cache_path",
        type=str,
        default=FSD_THERMO_CACHE_PATH,
        help="path for FSD thermo cache",
    )

    args = argparser.parse_args()
    train(args)
