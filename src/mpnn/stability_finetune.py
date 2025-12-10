import csv
import numpy as np
import torch
import os
import argparse
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import wandb
from mpnn.stabddg.folding_dataset import MegascaleDataset, MgnifyDataset, ThermoMutDBDataset
from mpnn.model_utils import ProteinMPNN
from mpnn.stabddg.model import StaBddG


def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

  alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
  states = len(alpha_1)
  alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
  
  aa_1_N = {a:n for n,a in enumerate(alpha_1)}
  aa_3_N = {a:n for n,a in enumerate(alpha_3)}
  aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
  aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
  aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
  
  def AA_to_N(x):
    # ["ARND"] -> [[0,1,2,3]]
    x = np.array(x)
    if x.ndim == 0: x = x[None]
    return [[aa_1_N.get(a, states-1) for a in y] for y in x]
  
  def N_to_AA(x):
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x)
    if x.ndim == 1: x = x[None]
    return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip()

    if line[:6] == "HETATM" and line[17:17+3] == "MSE":
      line = line.replace("HETATM","ATOM  ")
      line = line.replace("MSE","MET")

    if line[:4] == "ATOM":
      ch = line[21:22]
      if ch == chain or chain is None:
        atom = line[12:12+4].strip()
        resi = line[17:17+3]
        resn = line[22:22+5].strip()
        x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

        if resn[-1].isalpha(): 
            resa,resn = resn[-1],int(resn[:-1])-1
        else: 
            resa,resn = "",int(resn)-1

        if resn < min_resn: 
            min_resn = resn
        if resn > max_resn: 
            max_resn = resn
        if resn not in xyz: 
            xyz[resn] = {}
        if resa not in xyz[resn]: 
            xyz[resn][resa] = {}
        if resn not in seq: 
            seq[resn] = {}
        if resa not in seq[resn]: 
            seq[resn][resa] = resi

        if atom not in xyz[resn][resa]:
          xyz[resn][resa][atom] = np.array([x,y,z])

  # convert to numpy arrays, fill in missing values
  seq_,xyz_ = [],[]
  try:
      for resn in range(min_resn,max_resn+1):
        if resn in seq:
          for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
        else: seq_.append(20)
        if resn in xyz:
          for k in sorted(xyz[resn]):
            for atom in atoms:
              if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
              else: xyz_.append(np.full(3,np.nan))
        else:
          for atom in atoms: xyz_.append(np.full(3,np.nan))
      return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
  except TypeError:
      return 'no_chain', 'no_chain'

class StructureDataset():
    def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                #print(name, bad_chars, entry['seq'])
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False):
    c=0
    pdb_dict_list = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
     
    if input_chain_list:
        chain_alphabet = input_chain_list  
 

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ['CA']
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']
            xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_'+letter]=seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_'+letter]=xyz.tolist()
                else:
                    coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_'+letter]=coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict['name']=biounit[(fi+1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c+=1
    return pdb_dict_list

def validation_step(model, dataset_valid, batch_size=20000, name='val', mc_samples=20, device='cuda'):
    val_spearman=[]
    val_pearson=[]
    all_pred = []
    all_labels = []
    for sample in tqdm(dataset_valid):
        pdb_name = sample['name']
        ddG = sample['ddG'].to(device)
        mut_seqs = sample['complex_mut_seqs']
        N = mut_seqs.shape[0]
        M = batch_size // mut_seqs.shape[1] # convert number of tokens to number of sequences per batch

        mc_samples_pred = []
        for _ in range(mc_samples):
            sample_pred = []
            # Batching for mutants
            for batch_idx in range(0, N, M):
                B = min(N - batch_idx, M)
                # ddG prediction
                batch_pred = model.folding_ddG(sample['complex'], mut_seqs[batch_idx:batch_idx+B])
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
        f'{name}_spearman': np.format_float_positional(np.mean(val_spearman), unique=False, precision=3),
        f'{name}_pearson': np.format_float_positional(np.mean(val_pearson), unique=False, precision=3),
        f'{name}_all_spearman': np.format_float_positional(sp, unique=False, precision=3),
        f'{name}_all_pearson': np.format_float_positional(pr, unique=False, precision=3),
    }

def train_epoch(model, train_dataset, optimizer, ddG_loss_fn, name='megascale', batch_size=10000, device='cuda'):
    train_sum = []
    all_pred = []
    all_labels = []
    perm = torch.randperm(len(train_dataset))
    shuffled = [train_dataset[i] for i in perm]
    
    for sample in tqdm(shuffled, desc=f'{name} Epoch -- Complex'):
        pdb_name = sample['name']
        ddG = sample['ddG'].to(device)
        mut_seqs = sample['complex_mut_seqs']
        N = mut_seqs.shape[0]
        M = batch_size // mut_seqs.shape[1] # convert number of tokens to number of sequences per batch
        
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
            pred = model.folding_ddG(sample['complex'], mut_seqs[batch_idx:batch_idx+B])

            ddG_loss = ddG_loss_fn(pred, ddG[batch_idx:batch_idx+B])

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
        f'{name}_train_loss': np.mean(train_sum),
        f'{name}_train_spearman': sp,
        f'{name}_train_pearson': pr,
    }

def mgnify_train_epoch(model, mgnify_dataset, optimizer, ddG_loss_fn, batch_size=10000, device='cuda'):
    model.train()

    all_pred = []
    all_labels = []
    train_sum = []

    num_seqs = 0
    max_len = 0
    batch_complex_list = []
    batch_mut_seqs_list = []
    ddG_batched_list = []

    ### shuffle the dataset
    permutation = torch.randperm(len(mgnify_dataset))
    shuffled = [mgnify_dataset[i] for i in permutation]
    
    for sample in tqdm(shuffled, desc="Mgnify Epoch -- Complex"):
        if num_seqs * max_len < batch_size:
            batch_complex_list.append(sample['complex'])
            batch_mut_seqs_list.append(sample['complex_mut_seqs'])
            num_seqs += sample['complex_mut_seqs'].shape[0]
            max_len = max(max_len, sample['complex_mut_seqs'].shape[1])
            ddG_batched_list.append(sample['ddG'])
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
        'mgnify_train_spearman': sp,
        'mgnify_train_pearson': pr,
        'mgnify_train_loss': np.mean(train_sum),
    }

def finetune(model, megascale_train, megascale_valid, megascale_test, mgnify_train, fsd_thermo_train, args, batch_size=10000, device='cuda'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    ddG_loss_fn  = torch.nn.MSELoss()

    if args.wandb:
        print('Running zero-shot validation')
        model.eval()
        with torch.no_grad():
            valid_metrics = validation_step(model, megascale_valid, batch_size=10000, name='valid', mc_samples=args.mc_samples, device=device)
            test_metrics = validation_step(model, megascale_test, batch_size=10000, name='test', mc_samples=args.mc_samples, device=device)

        wandb.log(valid_metrics, step=0)
        wandb.log(test_metrics, step=0)

    for e in tqdm(range(args.num_epochs), desc="Epoch"):

        model.train()
        if args.mgnify_train:
            # Iterate through mgnify dataset
            mgnify_metrics = mgnify_train_epoch(model, mgnify_train, optimizer, ddG_loss_fn, batch_size=batch_size, device=device)

        # Iterate through megascale dataset
        if args.megascale_train:
            megascale_metrics = train_epoch(model, megascale_train, optimizer, ddG_loss_fn, name='megascale', batch_size=batch_size, device=device)

        if args.fsd_thermo_train:
            fsd_thermo_metrics = train_epoch(model, fsd_thermo_train, optimizer, ddG_loss_fn, name='fsd_thermo', batch_size=batch_size, device=device)

        if args.wandb:
            if e % args.val_freq == 0:
                model.eval()
                with torch.no_grad():
                    valid_metrics = validation_step(model, megascale_valid, batch_size=10000, name='valid', device=device, mc_samples=args.mc_samples)
                    test_metrics = validation_step(model, megascale_test, batch_size=10000, name='test', device=device, mc_samples=args.mc_samples)

                wandb.log(valid_metrics, step=e+1)
                wandb.log(test_metrics, step=e+1)

            if args.megascale_train:
                wandb.log(megascale_metrics, step=e+1)
            if args.mgnify_train:
                wandb.log(mgnify_metrics, step=e+1)
            if args.fsd_thermo_train:
                wandb.log(fsd_thermo_metrics, step=e+1)

            wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=e+1)
        scheduler.step()
        if (e + 1) % args.model_save_freq == 0:
            if not os.path.exists(args.model_save_dir):
                os.makedirs(args.model_save_dir)
            torch.save(model.pmpnn.state_dict(), f"{args.model_save_dir}/{args.run_name}_epoch{e}.pt")

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    torch.save(model.pmpnn.state_dict(), f"{args.model_save_dir}/{args.run_name}_final.pt")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--run_name", type=str, default="stability_finetune")
    argparser.add_argument("--checkpoint", type=str, default="model_ckpts/proteinmpnn.pt")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--num_epochs", type=int, default=80)
    argparser.add_argument("--batch_size", type=int, default=10000)
    argparser.add_argument("--model_save_freq", type=int, default=10)
    argparser.add_argument("--model_save_dir", type=str, default="cache/stability_finetuned")
    argparser.add_argument("--wandb", action='store_true')
    
    # Sample only one batch of mutants per domain during training.
    argparser.add_argument("--single_batch", action='store_true') 
    argparser.add_argument("--val_freq", type=int, default=10) # Train validation frequency
    argparser.add_argument("--noise_level", type=float, default=0.1) # Backbone noise.
    argparser.add_argument("--dropout", type=float, default=0.0) # Dropout during model training. 
    argparser.add_argument("--megascale_train", action='store_true')
    argparser.add_argument("--mgnify_train", action='store_true')
    argparser.add_argument("--fsd_thermo_train", action='store_true')
    argparser.add_argument("--mc_samples", type=int, default=20)

    # Do not permutation order between mutant and wildtype during decoding.
    argparser.add_argument("--no_antithetic_variates", action='store_true') 
    argparser.add_argument("--megascale_pdb_dir", type=str, default="/data/megascale/AlphaFold_model_PDBs")
    argparser.add_argument("--megascale_csv", type=str, default='/data/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv')
    argparser.add_argument("--mgnify_pdb_dir", type=str, default="/data/yehlin_mgnify/wt_structures")
    argparser.add_argument("--mgnify_csv", type=str, default='data/mgnify_yehlin/mgnify_processed_data.csv')
    argparser.add_argument("--mgnify_cache_path", type=str, default='cache/mgnify.pkl')
    argparser.add_argument("--fsd_thermo_csv", type=str, default='data/FSD/fsd_thermo.csv')
    argparser.add_argument("--fsd_thermo_pdb_dir", type=str, default='data/FSD/PDBs')
    argparser.add_argument("--fsd_thermo_cache_path", type=str, default='cache/fsd_thermo.pkl')
    argparser.add_argument("--lr", type=float, default=1e-6)
    argparser.add_argument("--weight_decay", type=float, default=1e-2)
    argparser.add_argument("--random_init", action='store_true')
    argparser.add_argument("--embedding_dim", type=int, default=128)
    argparser.add_argument("--num_layers", type=int, default=3)
    argparser.add_argument("--num_neighbors", type=int, default=48)

    args = argparser.parse_args()

    torch.manual_seed(args.seed)

    fsd_thermo_train = None
    mgnify_train = None
    megascale_train = None

    if args.fsd_thermo_train:
        fsd_thermo_train = ThermoMutDBDataset(csv_path=args.fsd_thermo_csv, pdb_dir=args.fsd_thermo_pdb_dir, pdb_dict_cache_path=args.fsd_thermo_cache_path, cif=False)

    if args.mgnify_train:
        mgnify_train = MgnifyDataset(csv_path=args.mgnify_csv, pdb_dir=args.mgnify_pdb_dir, pdb_dict_cache_path=args.mgnify_cache_path, cif=True)

    # Dataset preprocessing/loading
    if args.megascale_train:
        megascale_train = MegascaleDataset(csv_path=args.megascale_csv, pdb_dir=args.megascale_pdb_dir, split_path='data/rocklin/mega_splits.pkl', split='train')

    megascale_valid = MegascaleDataset(csv_path=args.megascale_csv, pdb_dir=args.megascale_pdb_dir, split_path='data/rocklin/mega_splits.pkl', split='val')
    megascale_test = MegascaleDataset(csv_path=args.megascale_csv, pdb_dir=args.megascale_pdb_dir, split_path='data/rocklin/mega_splits.pkl', split='test')
    # Load pre-trained ProteinMPNN
    device = torch.device("cuda")

    pmpnn = ProteinMPNN(node_features=128, 
                        edge_features=128, 
                        hidden_dim=args.embedding_dim,
                        num_encoder_layers=args.num_layers, 
                        num_decoder_layers=args.num_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=0.0,
                        augment_eps=0.0)
    
    mpnn_checkpoint = torch.load(args.checkpoint)
    if 'model_state_dict' in mpnn_checkpoint.keys():
        pmpnn.load_state_dict(mpnn_checkpoint['model_state_dict'])
    else:
        pmpnn.load_state_dict(mpnn_checkpoint)
    print('Successfully loaded model at', args.checkpoint)

    model = StaBddG(pmpnn=pmpnn, use_antithetic_variates=not args.no_antithetic_variates, 
                    noise_level=args.noise_level, device=device)
    
    model.to(device)
    model.eval()

    # Initialize wandb logging
    if args.wandb:
        print('Initializing weights and biases.')
        wandb.init(
            project="finetuning-scaling",
            entity='stanford-protein',
            name=args.run_name,
        )
        print('Weights and biases intialized.')

    finetune(model, megascale_train, megascale_valid, megascale_test, mgnify_train, fsd_thermo_train, args, batch_size=args.batch_size, device=device)
