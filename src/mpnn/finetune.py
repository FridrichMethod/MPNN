import argparse
import os

import numpy as np
import torch
import wandb
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

from mpnn.energy_mpnn import EnergyMPNN
from mpnn.energy_mpnn_dataset import MegascaleDataset, MgnifyDataset, ThermoMutDBDataset
from mpnn.env import (
    FSD_THERMO_CACHE_PATH,
    FSD_THERMO_CSV,
    FSD_THERMO_PDB_DIR,
    MEGASCALE_CSV,
    MEGASCALE_PDB_DIR,
    MEGASCALE_SPLIT_PATH,
    MGNIFY_CACHE_PATH,
    MGNIFY_CSV,
    MGNIFY_PDB_DIR,
)
from mpnn.protein_mpnn import ProteinMPNN


def validation_step(
    model, dataset_valid, batch_size=20000, name="val", mc_samples=20, device="cuda"
):
    val_spearman = []
    val_pearson = []
    all_pred = []
    all_labels = []
    for sample in tqdm(dataset_valid):
        ddG = sample["ddG"].to(device)
        mut_seqs = sample["complex_mut_seqs"]
        N = mut_seqs.shape[0]
        M = (
            batch_size // mut_seqs.shape[1]
        )  # convert number of tokens to number of sequences per batch

        mc_samples_pred = []
        for _ in range(mc_samples):
            sample_pred = []
            # Batching for mutants
            for batch_idx in range(0, N, M):
                B = min(N - batch_idx, M)
                # ddG prediction
                batch_pred = model.folding_ddG(
                    sample["complex"], mut_seqs[batch_idx : batch_idx + B]
                )
                sample_pred.append(batch_pred.detach().cpu())

            mc_samples_pred.append(torch.cat(sample_pred))

        pred = torch.stack(mc_samples_pred).mean(dim=0)

        if pred.numel() > 1:
            sp, _ = spearmanr(pred.cpu().detach().numpy(), ddG.cpu().detach().numpy())
            pr, _ = pearsonr(pred.cpu().detach().numpy(), ddG.cpu().detach().numpy())
            if not np.isnan(sp):
                val_spearman.append(sp)
                val_pearson.append(pr)

        all_pred.append(pred.cpu().detach().numpy())
        all_labels.append(ddG.cpu().detach().numpy())

    sp, _ = spearmanr(np.concatenate(all_pred), np.concatenate(all_labels))
    pr, _ = pearsonr(np.concatenate(all_pred), np.concatenate(all_labels))

    # format to 3 decimal places
    return {
        f"{name}_spearman": np.format_float_positional(
            np.mean(val_spearman), unique=False, precision=3
        ),
        f"{name}_pearson": np.format_float_positional(
            np.mean(val_pearson), unique=False, precision=3
        ),
        f"{name}_all_spearman": np.format_float_positional(sp, unique=False, precision=3),
        f"{name}_all_pearson": np.format_float_positional(pr, unique=False, precision=3),
    }


def train_epoch(
    model, train_dataset, optimizer, ddG_loss_fn, name="megascale", batch_size=10000, device="cuda"
):
    train_sum = []
    all_pred = []
    all_labels = []
    perm = torch.randperm(len(train_dataset))
    shuffled = [train_dataset[i] for i in perm]

    for sample in tqdm(shuffled, desc=f"{name} Epoch -- Complex"):
        ddG = sample["ddG"].to(device)
        mut_seqs = sample["complex_mut_seqs"]
        N = mut_seqs.shape[0]
        M = (
            batch_size // mut_seqs.shape[1]
        )  # convert number of tokens to number of sequences per batch

        # random shuffling
        permutation = torch.randperm(ddG.shape[0])
        ddG = ddG[permutation]
        mut_seqs = mut_seqs[permutation]

        sample_pred = []

        # Batching for mutants
        for batch_idx in range(0, N, M):
            B = min(N - batch_idx, M)
            optimizer.zero_grad()

            # ddG prediction
            pred = model.folding_ddG(sample["complex"], mut_seqs[batch_idx : batch_idx + B])

            ddG_loss = ddG_loss_fn(pred, ddG[batch_idx : batch_idx + B])

            ddG_loss.backward()
            optimizer.step()

            train_sum.append(ddG_loss.item())
            sample_pred.append(pred.detach().cpu())
            if args.single_batch:
                break

        sample_pred = torch.cat(sample_pred)
        all_pred.append(sample_pred.cpu().detach().numpy())
        all_labels.append(ddG.cpu().detach().numpy())

    sp, _ = spearmanr(np.concatenate(all_pred), np.concatenate(all_labels))
    pr, _ = pearsonr(np.concatenate(all_pred), np.concatenate(all_labels))

    return {
        f"{name}_train_loss": np.mean(train_sum),
        f"{name}_train_spearman": sp,
        f"{name}_train_pearson": pr,
    }


def mgnify_train_epoch(
    model, mgnify_dataset, optimizer, ddG_loss_fn, batch_size=10000, device="cuda"
):
    model.train()

    all_pred = []
    all_labels = []
    train_sum = []

    num_seqs = 0
    max_len = 0
    batch_complex_list = []
    batch_mut_seqs_list = []
    ddG_batched_list = []

    # shuffle the dataset
    permutation = torch.randperm(len(mgnify_dataset))
    shuffled = [mgnify_dataset[i] for i in permutation]

    for sample in tqdm(shuffled, desc="Mgnify Epoch -- Complex"):
        if num_seqs * max_len < batch_size:
            batch_complex_list.append(sample["complex"])
            batch_mut_seqs_list.append(sample["complex_mut_seqs"])
            num_seqs += sample["complex_mut_seqs"].shape[0]
            max_len = max(max_len, sample["complex_mut_seqs"].shape[1])
            ddG_batched_list.append(sample["ddG"])
            continue

        try:
            pred = model.folding_ddG_batched(batch_complex_list, batch_mut_seqs_list)

            ddG = torch.cat(ddG_batched_list, dim=0).to(device)

            loss = ddG_loss_fn(pred, ddG)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_pred.append(pred.cpu().detach().numpy())
            all_labels.append(ddG.cpu().detach().numpy().flatten())
            train_sum.append(loss.item())
        except Exception as e:
            print(f"Error in mgnify_train_epoch: {e}")
            continue

        num_seqs = 0
        max_len = 0
        batch_complex_list = []
        batch_mut_seqs_list = []
        ddG_batched_list = []

    sp, _ = spearmanr(np.concatenate(all_pred), np.concatenate(all_labels))
    pr, _ = pearsonr(np.concatenate(all_pred), np.concatenate(all_labels))

    return {
        "mgnify_train_spearman": sp,
        "mgnify_train_pearson": pr,
        "mgnify_train_loss": np.mean(train_sum),
    }


def finetune(
    model,
    megascale_train,
    megascale_valid,
    megascale_test,
    mgnify_train,
    fsd_thermo_train,
    args,
    batch_size=10000,
    device="cuda",
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    ddG_loss_fn = torch.nn.MSELoss()

    if args.wandb:
        print("Running zero-shot validation")
        model.eval()
        with torch.no_grad():
            valid_metrics = validation_step(
                model,
                megascale_valid,
                batch_size=10000,
                name="valid",
                mc_samples=args.mc_samples,
                device=device,
            )
            test_metrics = validation_step(
                model,
                megascale_test,
                batch_size=10000,
                name="test",
                mc_samples=args.mc_samples,
                device=device,
            )

        wandb.log(valid_metrics, step=0)
        wandb.log(test_metrics, step=0)

    for e in tqdm(range(args.num_epochs), desc="Epoch"):
        model.train()
        if args.mgnify_train:
            # Iterate through mgnify dataset
            mgnify_metrics = mgnify_train_epoch(
                model, mgnify_train, optimizer, ddG_loss_fn, batch_size=batch_size, device=device
            )

        # Iterate through megascale dataset
        if args.megascale_train:
            megascale_metrics = train_epoch(
                model,
                megascale_train,
                optimizer,
                ddG_loss_fn,
                name="megascale",
                batch_size=batch_size,
                device=device,
            )

        if args.fsd_thermo_train:
            fsd_thermo_metrics = train_epoch(
                model,
                fsd_thermo_train,
                optimizer,
                ddG_loss_fn,
                name="fsd_thermo",
                batch_size=batch_size,
                device=device,
            )

        if args.wandb:
            if e % args.val_freq == 0:
                model.eval()
                with torch.no_grad():
                    valid_metrics = validation_step(
                        model,
                        megascale_valid,
                        batch_size=10000,
                        name="valid",
                        device=device,
                        mc_samples=args.mc_samples,
                    )
                    test_metrics = validation_step(
                        model,
                        megascale_test,
                        batch_size=10000,
                        name="test",
                        device=device,
                        mc_samples=args.mc_samples,
                    )

                wandb.log(valid_metrics, step=e + 1)
                wandb.log(test_metrics, step=e + 1)

            if args.megascale_train:
                wandb.log(megascale_metrics, step=e + 1)
            if args.mgnify_train:
                wandb.log(mgnify_metrics, step=e + 1)
            if args.fsd_thermo_train:
                wandb.log(fsd_thermo_metrics, step=e + 1)

            wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=e + 1)
        scheduler.step()
        if (e + 1) % args.model_save_freq == 0:
            if not os.path.exists(args.model_save_dir):
                os.makedirs(args.model_save_dir)
            torch.save(
                model.protein_mpnn.state_dict(),
                f"{args.model_save_dir}/{args.run_name}_epoch{e}.pt",
            )

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    torch.save(model.protein_mpnn.state_dict(), f"{args.model_save_dir}/{args.run_name}_final.pt")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--run_name", type=str, default="stability_finetune")
    argparser.add_argument("--checkpoint", type=str, default="checkpoints/proteinmpnn.pt")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--num_epochs", type=int, default=80)
    argparser.add_argument("--batch_size", type=int, default=10000)
    argparser.add_argument("--model_save_freq", type=int, default=10)
    argparser.add_argument("--model_save_dir", type=str, default="checkpoints/stability_finetuned")
    argparser.add_argument("--wandb", action="store_true")

    # Sample only one batch of mutants per domain during training.
    argparser.add_argument("--single_batch", action="store_true")
    argparser.add_argument("--val_freq", type=int, default=10)  # Train validation frequency
    argparser.add_argument("--noise_level", type=float, default=0.1)  # Backbone noise.
    argparser.add_argument("--dropout", type=float, default=0.0)  # Dropout during model training.
    argparser.add_argument("--megascale_train", action="store_true")
    argparser.add_argument("--mgnify_train", action="store_true")
    argparser.add_argument("--fsd_thermo_train", action="store_true")
    argparser.add_argument("--mc_samples", type=int, default=20)

    # Do not permutation order between mutant and wildtype during decoding.
    argparser.add_argument("--no_antithetic_variates", action="store_true")
    argparser.add_argument(
        "--megascale_split_path",
        type=str,
        default=MEGASCALE_SPLIT_PATH,
    )
    argparser.add_argument(
        "--megascale_pdb_dir",
        type=str,
        default=MEGASCALE_PDB_DIR,
    )
    argparser.add_argument(
        "--megascale_csv",
        type=str,
        default=MEGASCALE_CSV,
    )
    argparser.add_argument(
        "--mgnify_pdb_dir",
        type=str,
        default=MGNIFY_PDB_DIR,
    )
    argparser.add_argument(
        "--mgnify_csv",
        type=str,
        default=MGNIFY_CSV,
    )
    argparser.add_argument(
        "--mgnify_cache_path",
        type=str,
        default=MGNIFY_CACHE_PATH,
    )
    argparser.add_argument(
        "--fsd_thermo_csv",
        type=str,
        default=FSD_THERMO_CSV,
    )
    argparser.add_argument(
        "--fsd_thermo_pdb_dir",
        type=str,
        default=FSD_THERMO_PDB_DIR,
    )
    argparser.add_argument(
        "--fsd_thermo_cache_path",
        type=str,
        default=FSD_THERMO_CACHE_PATH,
    )
    argparser.add_argument("--lr", type=float, default=1e-6)
    argparser.add_argument("--weight_decay", type=float, default=1e-2)
    argparser.add_argument("--random_init", action="store_true")
    argparser.add_argument("--embedding_dim", type=int, default=128)
    argparser.add_argument("--num_layers", type=int, default=3)
    argparser.add_argument("--num_neighbors", type=int, default=48)

    args = argparser.parse_args()

    torch.manual_seed(args.seed)

    fsd_thermo_train = None
    mgnify_train = None
    megascale_train = None

    if args.fsd_thermo_train:
        fsd_thermo_train = ThermoMutDBDataset(
            csv_path=args.fsd_thermo_csv,
            pdb_dir=args.fsd_thermo_pdb_dir,
            pdb_dict_cache_path=args.fsd_thermo_cache_path,
            cif=False,
        )

    if args.mgnify_train:
        mgnify_train = MgnifyDataset(
            csv_path=args.mgnify_csv,
            pdb_dir=args.mgnify_pdb_dir,
            pdb_dict_cache_path=args.mgnify_cache_path,
            cif=True,
        )

    # Dataset preprocessing/loading
    if args.megascale_train:
        megascale_train = MegascaleDataset(
            csv_path=args.megascale_csv,
            pdb_dir=args.megascale_pdb_dir,
            split_path=args.megascale_split_path,
            split="train",
        )

    megascale_valid = MegascaleDataset(
        csv_path=args.megascale_csv,
        pdb_dir=args.megascale_pdb_dir,
        split_path=args.megascale_split_path,
        split="val",
    )
    megascale_test = MegascaleDataset(
        csv_path=args.megascale_csv,
        pdb_dir=args.megascale_pdb_dir,
        split_path=args.megascale_split_path,
        split="test",
    )

    device = torch.device("cuda")

    protein_mpnn = ProteinMPNN(
        hidden_dim=args.embedding_dim,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        num_neighbors=args.num_neighbors,
        dropout=0.0,
        augment_eps=0.0,
    )

    mpnn_checkpoint = torch.load(args.checkpoint)
    if "model_state_dict" in mpnn_checkpoint.keys():
        protein_mpnn.load_state_dict(mpnn_checkpoint["model_state_dict"])
    else:
        protein_mpnn.load_state_dict(mpnn_checkpoint)
    print("Successfully loaded model at", args.checkpoint)

    model = EnergyMPNN(
        protein_mpnn=protein_mpnn,
        use_antithetic_variates=not args.no_antithetic_variates,
        noise_level=args.noise_level,
        device=device,
    )

    model.to(device)
    model.eval()

    # Initialize wandb logging
    if args.wandb:
        print("Initializing weights and biases.")
        wandb.init(
            project="finetuning-scaling",
            entity="stanford-protein",
            name=args.run_name,
        )
        print("Weights and biases intialized.")

    finetune(
        model,
        megascale_train,
        megascale_valid,
        megascale_test,
        mgnify_train,
        fsd_thermo_train,
        args,
        batch_size=args.batch_size,
        device=device,
    )
