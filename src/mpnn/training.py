import argparse
import time, os
import numpy as np
import torch
import os.path
from tqdm import tqdm
import wandb
import gc
import random
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from mpnn.utils import worker_init_fn, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader, flattened_PDB_dataset
from mpnn.model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN
from mpnn.stability_eval import eval_pretrained_mpnn

def get_run_name(args):
    return f"{args.code_version}_h{args.hidden_dim}_l{args.num_encoder_layers}_n{args.num_neighbors}_lr{args.learning_rate}_e{args.num_epochs}_wd{args.weight_decay}_bs{args.batch_size}_bb{args.backbone_noise}_s{args.seed}"

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup_run(args):
    run_name, tags = None, None
    if args.run_name:
        run_name = args.run_name
        tags = args.tags if args.tags else [args.code_version]
    else:
        run_name = get_run_name(args)
        tags = args.tags
        tags.extend(run_name.split('_'))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    base_folder = os.path.join(args.output_dir, run_name)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    logfile = os.path.join(base_folder, 'log.txt')

    with open(logfile, 'w') as f:
        f.write('Epoch\tTrain\tValidation\n')

    return run_name, tags, base_folder, logfile

def load_pdb_data(data_path, args):
    params = {
        "LIST"    : os.path.join(data_path, "list.csv"), 
        "VAL"     : os.path.join(data_path, "valid_clusters.txt"),
        "TEST"    : os.path.join(data_path, "test_clusters.txt"),
        "DIR"     : data_path,
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }


    def collate_passthrough_filter(batch):  # batch_size=1
        if not batch[0]:
            return None
        return batch[0]

    excluded_pdbs = []
    if args.exclude_membrane:
        import pandas as pd
        excluded_pdbs = pd.read_csv('./data/excluded_PDBs.csv')['PDB_IDS'].tolist()
        
    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': False, # Arthur: fix data order as a baseline
                  'pin_memory': False,
                  'collate_fn': collate_passthrough_filter,
                  'persistent_workers': True,
                  'num_workers': min(args.num_workers, os.cpu_count())}

    print("building training clusters")
    train, valid, _ = build_training_clusters(params)

    print('loading datasets')

    train_clusters = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    print(f'number of training clusters: {len(train_clusters)}')
    train_cluster_loader = torch.utils.data.DataLoader(train_clusters, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    valid_clusters = flattened_PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    print(f'number of validation clusters: {len(valid_clusters)}')
    valid_cluster_loader = torch.utils.data.DataLoader(valid_clusters, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    if args.debug:
        first_100_train, first_100_valid = [], []
        i = 0
        for x, y in tqdm(zip(train_cluster_loader, valid_cluster_loader), total=100, desc="Loading training and validation data"):
            if i >= 100:
                break
            if x is not None:
                first_100_train.append(x)
            if y is not None:
                first_100_valid.append(y)
            i += 1

        train_cluster_loader = first_100_train
        valid_cluster_loader = first_100_valid

    pdb_dict_train = []
    skipped_excluded = 0
    for x in tqdm(train_cluster_loader, total=len(train_cluster_loader), desc="Loading training data"):
        if x is None: continue
        if x['name'].split('_')[0] in excluded_pdbs:
            skipped_excluded += 1
            continue
        pdb_dict_train.append(x)
    print(f"Skipped {skipped_excluded} excluded PDBs")

    pdb_dict_valid = []
    for x in tqdm(valid_cluster_loader, total=len(valid_cluster_loader), desc="Loading validation data"):
        if x is not None:
            pdb_dict_valid.append(x)

    if args.max_protein_length > args.batch_size:
        args.max_protein_length = args.batch_size
        print("max_protein_length must be less than batch_size. Reducing max_protein_length to batch_size.")

    pdb_dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length) 
    pdb_dataset_valid = StructureDataset(pdb_dict_valid, truncate=None, max_length=args.max_protein_length)
    
    pdb_loader_train = StructureLoader(pdb_dataset_train, batch_size=args.batch_size)
    pdb_loader_valid = StructureLoader(pdb_dataset_valid, batch_size=args.batch_size)

    return pdb_loader_train, pdb_loader_valid, train_cluster_loader, excluded_pdbs

def get_model_and_optimizer(args, device, total_steps):
    print("building model")
    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise)
    model.to(device)

    print('model parameter count: ', sum(p.numel() for p in model.parameters()))
        
    optimizer = None
    scheduler = None

    if args.previous_checkpoint:
        checkpoint = torch.load(args.previous_checkpoint)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.optimizer == "noam":
            optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
            optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.scheduler == "cosine":
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
                warmup_steps = 0.01 * total_steps
                warmup = LinearLR(optimizer, start_factor=1e-8, total_iters=warmup_steps)
                cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0.0)
                scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                raise ValueError(f"Invalid scheduler type: {args.scheduler}")
        else:
            raise ValueError(f"Invalid optimizer type: {args.optimizer}")
    else:
        total_step = 0
        epoch = 0
        if args.optimizer == "noam":
            optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
        elif args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            if args.scheduler == "cosine":
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
                warmup_steps = int(0.01 * total_steps)
                warmup = LinearLR(optimizer, start_factor=1e-8, total_iters=warmup_steps)
                cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0.0)
                scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        else:
            raise ValueError(f"Invalid optimizer type: {args.optimizer}")
    return model, optimizer, scheduler, epoch

def train(args):
    seed_everything(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    
    run_name, tags, base_folder, logfile = setup_run(args)
    if os.path.exists(os.path.join(base_folder, 'model_weights', 'last_epoch.pt')) and not args.force_rerun:
        print(f"Training run already exists for {run_name}. Use --force_rerun to rerun the model.")
        return
    
    pdb_loader_train, pdb_loader_valid, train_cluster_loader, excluded_pdbs = load_pdb_data(args.path_for_training_data, args)
    model, optimizer, scheduler, epoch = get_model_and_optimizer(args, device, total_steps=len(pdb_loader_train) * args.num_epochs)

    if args.wandb:
        wandb.init(entity=args.wandb_entity, 
            project=args.wandb_project, 
            name=run_name, 
            config=args.__dict__,
            tags=tags)
        wandb.log({
            "model_parameter_count": sum(p.numel() for p in model.parameters())
        })

    print("entering training loop")
    total_step = 0
    for e in tqdm(range(args.num_epochs), desc="Epoch"):
        t0 = time.time()
        e = epoch + e
        model.train()
        train_sum, train_weights = 0., 0.
        train_acc = 0.
        skipped_batches = 0

        for batch_idx, batch in tqdm(enumerate(pdb_loader_train), total=len(pdb_loader_train), desc="Training Batch"):
            try:
                model.train()
                optimizer.zero_grad()
                X, S, mask, _, chain_M, residue_idx, _, chain_encoding_all = featurize(batch, device)
                mask_for_loss = mask*chain_M

                if args.mixed_precision:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                else:
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)

                loss_av_smoothed.backward()

                if args.gradient_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                optimizer.step()
                if args.scheduler == "cosine":
                    scheduler.step()

                loss, _, true_false = loss_nll(S, log_probs, mask_for_loss)
            
                train_sum += torch.sum(loss * mask_for_loss).float().cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).float().cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).float().cpu().data.numpy()

                total_step += 1
                
            except RuntimeError as err:
                if "out of memory" in str(err).lower() or "CUDA out of memory" in str(err):
                    skipped_batches += 1
                    print(f"WARNING: OOM at step {total_step}, batch {batch_idx}. Skipping batch. ({skipped_batches} skipped so far)")
                    # Clear cache and free memory
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Zero gradients to free memory (safe even if no gradients exist)
                    try:
                        optimizer.zero_grad(set_to_none=True)
                    except:
                        pass
                    continue
                else:
                    # Re-raise if it's not an OOM error
                    skipped_batches += 1
                    print(f"WARNING: Error at step {total_step}, batch {batch_idx}. Skipping batch. ({skipped_batches} skipped so far)")
                    print(err)
                    continue


        train_loss = train_sum / train_weights
        train_accuracy = train_acc / train_weights
        train_perplexity = np.exp(train_loss)

        train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     

        train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
        
        if (e+1) == 1 or (e+1) % args.validate_every_n_epochs == 0 or e == args.num_epochs - 1:
            model.eval()
            validation_sum, validation_weights = 0., 0.
            validation_acc = 0.
            for _, batch in tqdm(enumerate(pdb_loader_valid), total=len(pdb_loader_valid), desc="Validation Batch"):
                X, S, mask, _, chain_M, residue_idx, _, chain_encoding_all = featurize(batch, device, dtype=torch.float32)
                mask_for_loss = mask*chain_M
                with torch.inference_mode():
                    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                
                loss, _, true_false = loss_nll(S, log_probs, mask_for_loss)
                
                validation_sum += torch.sum(loss * mask_for_loss).float().cpu().data.numpy()
                validation_acc += torch.sum(true_false * mask_for_loss).float().cpu().data.numpy()
                validation_weights += torch.sum(mask_for_loss).float().cpu().data.numpy()

            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)
        
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
            if args.wandb:
                wandb.log({
                    "validation_perplexity": float(validation_perplexity_),
                    "validation_accuracy": float(validation_accuracy_),
                }, step=total_step)
                if e == args.num_epochs - 1:
                    wandb.log({
                        "final_validation_perplexity": float(validation_perplexity_),
                        "final_validation_accuracy": float(validation_accuracy_),
                    }, step=total_step)
                    with torch.inference_mode():
                        train_metrics, valid_metrics, test_metrics, fsd_thermo_metrics = eval_pretrained_mpnn(model, batch_size=10000, device=device, mc_samples=20, backbone_noise=args.backbone_noise,
                            megascale_pdb_dir=args.megascale_pdb_dir,
                            megascale_csv=args.megascale_csv,
                            fsd_thermo_csv=args.fsd_thermo_csv,
                            fsd_thermo_pdb_dir=args.fsd_thermo_pdb_dir,
                            fsd_thermo_cache_path=args.fsd_thermo_cache_path
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
        dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
        with open(logfile, 'a') as f:
            f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}, skipped_batches: {skipped_batches}\n')
        if args.wandb:
            wandb.log({
                "epoch": e+1,
                "step": total_step,
                "time": float(dt),
                "train_perplexity": float(train_perplexity_),
                "train_accuracy": float(train_accuracy_),
                "learning_rate": optimizer.param_groups[0]['lr'] if args.optimizer == "adamw" else optimizer.rate(total_step),
                "skipped_batches": skipped_batches,
            }, step=total_step)
                
        print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, skipped_batches: {skipped_batches}')
        
        ### save model weights
        optimizer_state_dict = optimizer.optimizer.state_dict() if args.optimizer == "noam" else optimizer.state_dict()
        model_info = {
            'epoch': e+1,
            'step': total_step,
            'num_edges' : args.num_neighbors,
            'noise_level': args.backbone_noise,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_state_dict,
            'scheduler_state_dict': scheduler.state_dict() if args.scheduler == "cosine" else None,
        }
        
        if not os.path.exists(os.path.join(base_folder, 'model_weights')):
            os.makedirs(os.path.join(base_folder, 'model_weights'))
        checkpoint_filename_last = os.path.join(base_folder, 'model_weights', 'last_epoch.pt')
        torch.save(model_info, checkpoint_filename_last)

        if (e+1) % args.save_model_every_n_epochs == 0:
            checkpoint_filename = os.path.join(base_folder, 'model_weights', f'epoch_{e+1}_step_{total_step}.pt')
            torch.save(model_info, checkpoint_filename)

        if (e+1) % args.reload_data_every_n_epochs == 0:
            print(f"Reloading training data at epoch {e+1}...")
            pdb_dict_train = []
            skipped_excluded = 0
            for x in tqdm(train_cluster_loader, total=len(train_cluster_loader), desc="Loading training data"):
                if x is None: continue
                if x['name'].split('_')[0] in excluded_pdbs:
                    skipped_excluded += 1
                    continue
                pdb_dict_train.append(x)
            print(f"Skipped {skipped_excluded} excluded PDBs")
            pdb_dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=args.max_protein_length)   
            pdb_loader_train = StructureLoader(pdb_dataset_train, batch_size=args.batch_size)

        if e == args.num_epochs - 1:
            print("Training complete. Saving model weights...")
            checkpoint_filename = os.path.join(base_folder, 'model_weights', f'final_epoch.pt')
            torch.save(model_info, checkpoint_filename)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
    argparser.add_argument("--output_dir", type=str, default="./training_output", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--optimizer", type=str, default="adamw", help="optimizer to use")
    argparser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    argparser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    argparser.add_argument("--scheduler", type=str, default="", help="scheduler to use")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", action="store_true", help="train with mixed precision")
    argparser.add_argument("--wandb", action="store_true", help="use wandb for logging")
    argparser.add_argument("--wandb_entity", type=str, default="stanford-protein", help="wandb entity")
    argparser.add_argument("--run_name", type=str, default="", help="wandb run name")
    argparser.add_argument("--wandb_project", type=str, default="pretraining-scaling", help="wandb project")
    argparser.add_argument("--num_workers", type=int, default=12, help="number of workers for data loading")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=5, help="reload training data every n epochs")
    argparser.add_argument("--code_version", type=str, default="nov20", help="code version")
    argparser.add_argument("--force_rerun", action="store_true", help="force rerun of the model even if a training run already exists.")
    argparser.add_argument("--seed", type=int, default=42, help="random seed")
    argparser.add_argument("--validate_every_n_epochs", type=int, default=5, help="validate every n epochs")
    argparser.add_argument("--tags", type=lambda x: x.split(','), default=[], help="tags for wandb")
    argparser.add_argument("--exclude_membrane", action="store_true", help="exclude transmembrane proteins from training (soluble MPNN)")
    argparser.add_argument("--debug", action="store_true", help="debug mode")
    argparser.add_argument("--megascale_pdb_dir", type=str, default="/data/megascale/AlphaFold_model_PDBs", help="path for megascale PDBs")
    argparser.add_argument("--megascale_csv", type=str, default='/data/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv', help="path for megascale CSV")
    argparser.add_argument("--fsd_thermo_csv", type=str, default='data/FSD/fsd_thermo.csv', help="path for FSD thermo CSV")
    argparser.add_argument("--fsd_thermo_pdb_dir", type=str, default='data/FSD/PDBs', help="path for FSD thermo PDBs")
    argparser.add_argument("--fsd_thermo_cache_path", type=str, default='cache/fsd_thermo.pkl', help="path for FSD thermo cache")
    
    args = argparser.parse_args()    
    train(args)   
