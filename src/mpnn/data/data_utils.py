"""Data utilities for the MPNN model."""

import csv
import os
import random

import numpy as np
import torch
from Bio.PDB.MMCIFParser import MMCIFParser
from dateutil import parser
from torch_geometric.data import Data

from mpnn.common.constants import (
    AA_1_TO_N,
    AA_3_TO_1,
    AA_3_TO_N,
    AA_ALPHABET,
    AA_N_TO_1,
    BACKBONE_ATOMS,
    BACKBONE_MAINCHAIN_ATOMS,
    CA_ATOMS,
    CHAIN_ALPHABET,
    VOCAB_SIZE,
)
from mpnn.typing_utils import StrPath


def AA_to_N(x):
    """Convert one-letter amino acid codes to numeric indices."""
    x = np.array(x)
    if x.ndim == 0:
        x = x[None]
    return [[AA_1_TO_N.get(a, VOCAB_SIZE - 1) for a in y] for y in x]


def N_to_AA(x):
    """Convert numeric amino acid indices back to one-letter codes."""
    x = np.array(x)
    if x.ndim == 1:
        x = x[None]
    return ["".join([AA_N_TO_1.get(a, "-") for a in y]) for y in x]


def is_aa(x: str) -> bool:
    """Check if a string is a valid amino acid."""
    return x in AA_ALPHABET


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


def entry_to_pyg_data(entry: dict) -> Data:
    """Convert a processed PDB entry dict to a PyG Data object."""
    X, S, mask, _, chain_M, residue_idx, _, chain_encoding_all = featurize(
        [entry], device=torch.device("cpu")
    )

    return Data(
        x=X[0],
        chain_seq_label=S[0],
        mask=mask[0],
        chain_mask_all=chain_M[0],
        residue_idx=residue_idx[0],
        chain_encoding_all=chain_encoding_all[0],
    )


def parse_pdb_biounits(x, atoms=BACKBONE_MAINCHAIN_ATOMS, chain=None):
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


def parse_pdb(pdb_path: StrPath, input_chain_list=None, ca_only=False):
    c = 0
    pdb_dict_list = []
    chain_alphabet = list(CHAIN_ALPHABET)

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [pdb_path]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ""
        for letter in chain_alphabet:
            sidechain_atoms = CA_ATOMS if ca_only else BACKBONE_ATOMS
            xyz, seq = parse_pdb_biounits(biounit, atoms=sidechain_atoms, chain=letter)
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


def parse_cif(cif_path: StrPath):
    """Parse an mmCIF file and return a dict with these keys:
    - seq_chain_<X>    : one-letter sequence for chain X
    - coords_chain_<X> : dict with keys N_chain_X, CA_chain_X, C_chain_X, O_chain_X
                         each mapping to a list of [x,y,z]
    - name             : filename without extension
    - num_of_chains    : number of chains parsed
    - seq              : concatenation of all chain sequences in parsed order
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("S", cif_path)
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
            if not is_aa(res.get_resname()):
                continue
            seq.append(AA_3_TO_1[res.get_resname()])
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

    out["name"] = os.path.splitext(os.path.basename(cif_path))[0]
    out["num_of_chains"] = chain_count
    out["seq"] = "".join(seq_all)

    return out


def process_pdb(t):
    my_dict = {}
    concat_seq = ""
    mask_list = []
    visible_list = []
    if len(list(np.unique(t["idx"]))) >= 352:
        return None
    for idx in list(np.unique(t["idx"])):
        letter = CHAIN_ALPHABET[idx]
        res = np.argwhere(t["idx"] == idx)
        initial_sequence = "".join(list(np.array(list(t["seq"]))[res][0,]))
        if initial_sequence[-6:] == "HHHHHH":
            res = res[:, :-6]
        if initial_sequence[0:6] == "HHHHHH":
            res = res[:, 6:]
        if initial_sequence[-7:-1] == "HHHHHH":
            res = res[:, :-7]
        if initial_sequence[-8:-2] == "HHHHHH":
            res = res[:, :-8]
        if initial_sequence[-9:-3] == "HHHHHH":
            res = res[:, :-9]
        if initial_sequence[-10:-4] == "HHHHHH":
            res = res[:, :-10]
        if initial_sequence[1:7] == "HHHHHH":
            res = res[:, 7:]
        if initial_sequence[2:8] == "HHHHHH":
            res = res[:, 8:]
        if initial_sequence[3:9] == "HHHHHH":
            res = res[:, 9:]
        if initial_sequence[4:10] == "HHHHHH":
            res = res[:, 10:]
        if res.shape[1] < 4:
            pass
        else:
            my_dict["seq_chain_" + letter] = "".join(list(np.array(list(t["seq"]))[res][0,]))
            concat_seq += my_dict["seq_chain_" + letter]
            if idx in t["masked"]:
                mask_list.append(letter)
            else:
                visible_list.append(letter)
            coords_dict_chain = {}
            all_atoms = np.array(t["xyz"][res,])[0,]  # [L, 14, 3]
            coords_dict_chain["N_chain_" + letter] = all_atoms[:, 0, :].tolist()
            coords_dict_chain["CA_chain_" + letter] = all_atoms[:, 1, :].tolist()
            coords_dict_chain["C_chain_" + letter] = all_atoms[:, 2, :].tolist()
            coords_dict_chain["O_chain_" + letter] = all_atoms[:, 3, :].tolist()
            my_dict["coords_chain_" + letter] = coords_dict_chain
    my_dict["name"] = t["label"]
    my_dict["masked_list"] = mask_list
    my_dict["visible_list"] = visible_list
    my_dict["num_of_chains"] = len(mask_list) + len(visible_list)
    my_dict["seq"] = concat_seq
    return my_dict


def loader_pdb(item, params):
    pdbid, chid = item[0].split("_")
    PREFIX = "%s/pdb/%s/%s" % (params["DIR"], pdbid[1:3], pdbid)

    # load metadata
    if not os.path.isfile(PREFIX + ".pt"):
        return {"seq": np.zeros(5)}
    meta = torch.load(PREFIX + ".pt")
    asmb_ids = meta["asmb_ids"]
    asmb_chains = meta["asmb_chains"]
    chids = np.array(meta["chains"])

    # find candidate assemblies which contain chid chain
    asmb_candidates = set([a for a, b in zip(asmb_ids, asmb_chains) if chid in b.split(",")])

    # if the chains is missing is missing from all the assemblies
    # then return this chain alone
    if len(asmb_candidates) < 1:
        chain = torch.load("%s_%s.pt" % (PREFIX, chid))
        L = len(chain["seq"])
        return {
            "seq": chain["seq"],
            "xyz": chain["xyz"],
            "idx": torch.zeros(L).int(),
            "masked": torch.Tensor([0]).int(),
            "label": item[0],
        }

    # randomly pick one assembly from candidates
    asmb_i = random.sample(list(asmb_candidates), 1)

    # indices of selected transforms
    idx = np.where(np.array(asmb_ids) == asmb_i)[0]

    # load relevant chains
    chains = {
        c: torch.load("%s_%s.pt" % (PREFIX, c))
        for i in idx
        for c in asmb_chains[i]
        if c in meta["chains"]
    }

    # generate assembly
    asmb = {}
    for k in idx:
        # pick k-th xform
        xform = meta["asmb_xform%d" % k]
        u = xform[:, :3, :3]
        r = xform[:, :3, 3]

        # select chains which k-th xform should be applied to
        s1 = set(meta["chains"])
        s2 = set(asmb_chains[k].split(","))
        chains_k = s1 & s2

        # transform selected chains
        for c in chains_k:
            try:
                xyz = chains[c]["xyz"]
                xyz_ru = torch.einsum("bij,raj->brai", u, xyz) + r[:, None, None, :]
                asmb.update({(c, k, i): xyz_i for i, xyz_i in enumerate(xyz_ru)})
            except KeyError:
                return {"seq": np.zeros(5)}

    # select chains which share considerable similarity to chid
    seqid = meta["tm"][chids == chid][0, :, 1]
    homo = set([ch_j for seqid_j, ch_j in zip(seqid, chids) if seqid_j > params["HOMO"]])
    # stack all chains in the assembly together
    seq, xyz, idx, masked = "", [], [], []
    seq_list = []
    for counter, (k, v) in enumerate(asmb.items()):
        seq += chains[k[0]]["seq"]
        seq_list.append(chains[k[0]]["seq"])
        xyz.append(v)
        idx.append(torch.full((v.shape[0],), counter))
        if k[0] in homo:
            masked.append(counter)

    return {
        "seq": seq,
        "xyz": torch.cat(xyz, dim=0),
        "idx": torch.cat(idx, dim=0),
        "masked": torch.Tensor(masked).int(),
        "label": item[0],
    }


def build_training_clusters(params):
    val_ids = set([int(l) for l in open(params["VAL"]).readlines()])
    test_ids = set([int(l) for l in open(params["TEST"]).readlines()])

    # read & clean list.csv
    with open(params["LIST"], encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        rows = [
            [r[0], r[3], int(r[4])]
            for r in reader
            if float(r[2]) <= params["RESCUT"]
            and parser.parse(r[1]) <= parser.parse(params["DATCUT"])
        ]

    # compile training and validation sets
    train = {}
    valid = {}
    test = {}

    for r in rows:
        if r[2] in val_ids:
            if r[2] in valid:
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        elif r[2] in test_ids:
            if r[2] in test:
                test[r[2]].append(r[:2])
            else:
                test[r[2]] = [r[:2]]
        else:
            if r[2] in train:
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]

    return train, valid, test
