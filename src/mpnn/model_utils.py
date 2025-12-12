import random

import numpy as np
import torch

from mpnn.constants import (
    AA_3_TO_N,
    AA_ALPHABET,
    BACKBONE_ATOMS,
    CA_ATOMS,
    CHAIN_ALPHABET,
    THREE_TO_ONE,
)
from mpnn.utils import N_to_AA


def featurize(batch, device):
    B = len(batch)
    lengths = np.array([len(b["seq"]) for b in batch], dtype=np.int32)  # sum of chain seq lengths
    L_max = max([len(b["seq"]) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)  # residue idx with jumps across chains
    chain_M = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones(
        [B, L_max, L_max], dtype=np.int32
    )  # for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros(
        [B, L_max], dtype=np.int32
    )  # integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    S = np.zeros([B, L_max], dtype=np.int32)  # sequence AAs integers
    for i, b in enumerate(batch):
        masked_chains = b["masked_list"]
        visible_chains = b["visible_list"]
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for letter in all_chains:
            chain_seq = b[f"seq_chain_{letter}"]
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for vm in masked_temp_dict.values():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)  # randomly shuffle chain order
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for letter in all_chains:
            if letter in visible_chains:
                chain_seq = b[f"seq_chain_{letter}"]
                chain_length = len(chain_seq)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                x_chain = np.stack(
                    [
                        chain_coords[c]
                        for c in [
                            f"N_chain_{letter}",
                            f"CA_chain_{letter}",
                            f"C_chain_{letter}",
                            f"O_chain_{letter}",
                        ]
                    ],
                    1,
                )  # [chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
            elif letter in masked_chains:
                chain_seq = b[f"seq_chain_{letter}"]
                chain_length = len(chain_seq)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.ones(chain_length)  # 0.0 for visible chains
                x_chain = np.stack(
                    [
                        chain_coords[c]
                        for c in [
                            f"N_chain_{letter}",
                            f"CA_chain_{letter}",
                            f"C_chain_{letter}",
                            f"O_chain_{letter}",
                        ]
                    ],
                    1,
                )  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
        x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list, 0)  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)

        l = len(all_sequence)
        x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], "constant", constant_values=(np.nan,))
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - l]], "constant", constant_values=(0.0,))
        chain_M[i, :] = m_pad

        chain_encoding_pad = np.pad(
            chain_encoding, [[0, L_max - l]], "constant", constant_values=(0.0,)
        )
        chain_encoding_all[i, :] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([AA_ALPHABET.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3)))
    X[isnan] = 0.0

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)

    return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all


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


def parse_PDB_biounits(x, atoms=["N", "CA", "C"], chain=None):
    """input:  x = PDB filename
            atoms = atoms to extract (optional)
    output: (length, atoms, coords=(x,y,z)), sequence
    """
    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    for line in open(x, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()

        if line[:6] == "HETATM" and line[17 : 17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12 : 12 + 4].strip()
                resi = line[17 : 17 + 3]
                resn = line[22 : 22 + 5].strip()
                x, y, z = [float(line[i : (i + 8)]) for i in [30, 38, 46]]

                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1
                else:
                    resa, resn = "", int(resn) - 1

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
                    xyz[resn][resa][atom] = np.array([x, y, z])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(AA_3_TO_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(np.full(3, np.nan))
            else:
                for atom in atoms:
                    xyz_.append(np.full(3, np.nan))
        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_))
    except TypeError:
        return "no_chain", "no_chain"


def parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False):
    c = 0
    pdb_dict_list = []
    chain_alphabet = list(CHAIN_ALPHABET)

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ""
        for letter in chain_alphabet:
            sidechain_atoms = CA_ATOMS if ca_only else BACKBONE_ATOMS
            xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict["seq_chain_" + letter] = seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain["CA_chain_" + letter] = xyz.tolist()
                else:
                    coords_dict_chain["N_chain_" + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain["CA_chain_" + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain["C_chain_" + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain["O_chain_" + letter] = xyz[:, 3, :].tolist()
                my_dict["coords_chain_" + letter] = coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict["name"] = biounit[(fi + 1) : -4]
        my_dict["num_of_chains"] = s
        my_dict["seq"] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c += 1
    return pdb_dict_list


def parse_CIF(path_to_cif):
    """Parse an mmCIF file and return a dict with these keys:
    - seq_chain_<X>    : one-letter sequence for chain X
    - coords_chain_<X> : dict with keys N_chain_X, CA_chain_X, C_chain_X, O_chain_X
                         each mapping to a list of [x,y,z]
    - name             : filename without extension
    - num_of_chains    : number of chains parsed
    - seq              : concatenation of all chain sequences in parsed order
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("S", path_to_cif)
    model = next(structure.get_models())  # first (and usually only) model

    out = {}
    seq_all = []
    chain_count = 0

    for chain in model:
        cid = chain.id
        # prepare per‐chain containers
        seq = []
        coords = {
            f"N_chain_{cid}": [],
            f"CA_chain_{cid}": [],
            f"C_chain_{cid}": [],
            f"O_chain_{cid}": [],
        }

        # walk residues in chain
        for res in chain:
            if not is_aa(res.get_resname(), standard=True):
                continue
            seq.append(THREE_TO_ONE[res.get_resname()])
            # for each backbone atom, either grab coords or fill nan
            for atom_name in ("N", "CA", "C", "O"):
                if atom_name in res:
                    coord = res[atom_name].get_coord().tolist()
                else:
                    coord = [float("nan")] * 3
                coords[f"{atom_name}_chain_{cid}"].append(coord)

        if not seq:
            # skip chains with zero standard residues
            continue

        # store into output
        out[f"seq_chain_{cid}"] = "".join(seq)
        out[f"coords_chain_{cid}"] = coords

        seq_all.append("".join(seq))
        chain_count += 1

    out["name"] = os.path.splitext(os.path.basename(path_to_cif))[0]
    out["num_of_chains"] = chain_count
    out["seq"] = "".join(seq_all)

    return out
