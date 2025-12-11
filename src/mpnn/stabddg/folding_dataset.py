import os
import pickle

import numpy as np
import pandas as pd
import torch
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from mpnn.typing_utils import StrPath
from mpnn.utils import StructureDataset

three_to_one = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}


def parse_PDB_biounits(x, atoms=["N", "CA", "C"], chain=None):
    """input:  x = PDB filename
            atoms = atoms to extract (optional)
    output: (length, atoms, coords=(x,y,z)), sequence
    """
    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "GAP",
    ]

    aa_1_N = {a: n for n, a in enumerate(alpha_1)}
    aa_3_N = {a: n for n, a in enumerate(alpha_3)}
    aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
    aa_1_3 = {a: b for a, b in zip(alpha_1, alpha_3)}
    aa_3_1 = {b: a for a, b in zip(alpha_1, alpha_3)}

    def AA_to_N(x):
        # ["ARND"] -> [[0,1,2,3]]
        x = np.array(x)
        if x.ndim == 0:
            x = x[None]
        return [[aa_1_N.get(a, states - 1) for a in y] for y in x]

    def N_to_AA(x):
        # [[0,1,2,3]] -> ["ARND"]
        x = np.array(x)
        if x.ndim == 1:
            x = x[None]
        return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

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
                    seq_.append(aa_3_N.get(seq[resn][k], 20))
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
    init_alphabet = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ""
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ["CA"]
            else:
                sidechain_atoms = ["N", "CA", "C", "O"]
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
            seq.append(three_to_one[res.get_resname()])
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


class FoldingDataset(Dataset):
    def __init__(
        self,
        csv_path: StrPath,
        split_path: StrPath = "",
        pdb_dir: StrPath = "",
        pdb_dict_cache_path: StrPath = "",
        cif: bool = False,  # whether to use cif files instead of pdb files
        alphabet: str = "ACDEFGHIKLMNPQRSTVWYX",
    ):

        self.alphabet = alphabet
        self.data = []
        self.num_mutants = 0

        # Process skempi csv, keep only relevant split
        ddG_df = self.preprocess_df(csv_path)

        # Keep only split clusters
        if split_path:
            with open(split_path, "rb") as f:
                split_pdbs = pickle.load(f)
            ddG_df = ddG_df[ddG_df["#Pdb"].isin(split_pdbs)]

        # Get PDB file names
        self.pdb_names = set(ddG_df["#Pdb"].to_list())

        # Load cached structure dictionary
        structure_dict = {}
        if os.path.exists(pdb_dict_cache_path):
            print("Found cached structure dictionary at", pdb_dict_cache_path)
            with open(pdb_dict_cache_path, "rb") as f:
                structure_dict = pickle.load(f)
            print(f"Loaded {len(structure_dict)} structures from cache.")
        else:
            print("Did not find a cached structure dictionary, processing structures now.")
            structure_dict = self.preprocess_structures(pdb_dir, pdb_dict_cache_path, cif=cif)
            print(f"Processed {len(structure_dict)} structures.")

        # Process mutants
        grouped_ddG_df = ddG_df.groupby("#Pdb")

        structure_dict = [x for x in structure_dict if x["name"] in self.pdb_names]

        self.name_to_struct = {x["name"]: x for x in structure_dict}
        for name, group in tqdm(
            grouped_ddG_df, total=len(grouped_ddG_df), desc="Processing mutations"
        ):
            complex = self.name_to_struct[name]

            mutations_list = group["mutations"].to_list()

            complex_mut_seqs = self.mutations_to_seq(complex, mutations_list)

            ddG = group["ddG"].to_numpy()

            self.data.append({
                "name": name,
                "complex": complex,
                "complex_mut_seqs": torch.from_numpy(complex_mut_seqs),
                "ddG": torch.from_numpy(ddG),
            })

            self.num_mutants += len(mutations_list)

        print(
            f"Finished loading dataset with {len(self.data)} complexes and {self.num_mutants} mutants."
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def preprocess_structures(
        self,
        pdb_dir: StrPath = "",
        pdb_dict_cache_path: StrPath = "cache/pdb_dict.pkl",
        cif: bool = False,
    ) -> dict:

        file_paths = []
        if cif:
            file_paths = [os.path.join(pdb_dir, f"{name}.cif") for name in self.pdb_names]
        else:
            file_paths = [os.path.join(pdb_dir, f"{name}.pdb") for name in self.pdb_names]

        pdb_dict = []

        for path in tqdm(file_paths, desc="Reading in structures"):
            if cif:
                pdb_dict.append(parse_CIF(path))
            else:
                pdb_dict.append(parse_PDB(path)[0])

        for dict in pdb_dict:
            all_chains = []
            for key in dict.keys():
                if key.startswith("seq_chain_"):
                    all_chains.append(key.split("_")[-1])
                    dict[key] = "".join([x if x != "-" else "X" for x in dict[key]])
            dict["seq"] = "".join([x if x != "-" else "X" for x in dict["seq"]])
            dict["masked_list"] = all_chains
            dict["visible_list"] = []

        cache_dir = os.path.dirname(pdb_dict_cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        with open(pdb_dict_cache_path, "wb") as f:
            pickle.dump(pdb_dict, f)

        return pdb_dict

    def mutations_to_seq(
        self,
        processed_struct: dict,
        mutations_list: list,
    ) -> np.ndarray:

        chain_offset = {}
        offset = 0
        for key in processed_struct:
            if key.startswith("seq_chain_"):
                chain_offset[key.split("_")[-1]] = offset
                offset += len(processed_struct[key])

        index_matrix = []
        if "-" in list(processed_struct["seq"]) or "X" in list(processed_struct["seq"]):
            print(
                f"Warning: Sequence {processed_struct['name']} contains gaps or unknown residues."
            )

        for mutations in mutations_list:
            mut_seq = list(processed_struct["seq"])
            for mut in set(mutations):
                # print(mut)
                # print(mutations)
                # print(mutations_list)
                mut_chain = mut[1]
                wt_aa = mut[0]
                mut_aa = mut[-1]

                if mut_chain not in chain_offset:
                    continue

                mut_chain_offset = chain_offset[mut_chain]
                mut_pos = int(mut[2:-1]) + mut_chain_offset - 1

                assert mut_seq[mut_pos] == wt_aa, (
                    f"{mut}: Expected {wt_aa} at position {mut_pos} in sequence, but found {mut_seq[mut_pos]} in {processed_struct['name']}."
                )

                mut_seq[mut_pos] = mut_aa

            indices = [
                self.alphabet.index(a) if a != "-" else self.alphabet.index("X") for a in mut_seq
            ]
            indices = np.asarray(indices, dtype=np.int64)
            index_matrix.append(indices)

        index_matrix = np.vstack(index_matrix)

        return index_matrix

    def preprocess_df(self, csv_path) -> pd.DataFrame:

        raise NotImplementedError("")


class ThermoMutDBDataset(FoldingDataset):
    def __init__(
        self,
        csv_path: StrPath,
        split_path: StrPath = "",
        pdb_dir: StrPath = "",
        pdb_dict_cache_path: StrPath = "",
        cif: bool = False,
        alphabet: str = "ACDEFGHIKLMNPQRSTVWYX",
    ):
        super().__init__(csv_path, split_path, pdb_dir, pdb_dict_cache_path, cif, alphabet)

    def preprocess_df(self, csv_path):
        df = pd.read_csv(csv_path)
        df["mutations"] = df["Mutation(s)_cleaned"].astype(str)
        df["mutations"] = df["mutations"].str.split(",")
        df = df[["#Pdb", "mutations", "ddG"]]
        # convert ddG to float32
        df["ddG"] = df["ddG"].astype(np.float32)
        return df


class MgnifyDataset(FoldingDataset):
    def __init__(
        self,
        csv_path: StrPath,
        split_path: StrPath = "",
        pdb_dir: StrPath = "",
        pdb_dict_cache_path: StrPath = "",
        cif: bool = False,
        alphabet: str = "ACDEFGHIKLMNPQRSTVWYX",
    ):
        super().__init__(csv_path, split_path, pdb_dir, pdb_dict_cache_path, cif, alphabet)

    def preprocess_df(self, csv_path):
        df = pd.read_csv(csv_path)
        df["mutations"] = df["mutations"].astype(str)
        df["mutations"] = df["mutations"].str.split(",")
        df = df[["#Pdb", "mutations", "ddG"]]
        # convert ddG to float32
        df["ddG"] = -df[
            "ddG"
        ].astype(
            np.float32
        )  # Mgnify uses the convention that a negative value (ddG < 0) represents a destabilizing mutation. Flip it here.
        return df


class MgnifyBatchedDatset(Dataset):
    def __init__(self, mgnify_dataset, batch_size=10000):
        self.mgnify_dataset = mgnify_dataset
        self.batch_size = batch_size

    def __len__(self):
        return len(self.mgnify_dataset)

    def __getitem__(self, idx):
        return self.mgnify_dataset[idx]


class MegascaleDataset(Dataset):
    """Note that the Megascale dataset is handled differently from other datasets due to naming issues.
    This is non-standard and therefore does not inherit from the base FoldingDataset class.
    """

    def __init__(
        self,
        csv_path: StrPath = "",
        pdb_dir: StrPath = "",
        split_path: StrPath = "",
        split: str = "",
    ):
        # Dataset preprocessing/loading

        # Read split files
        complex_names = None
        with open(split_path, "rb") as f:
            splits = pickle.load(f)
            train_names = splits["train"]
            val_names = splits["val"].tolist()
            test_names = splits["test"].tolist()
            if split == "train":
                complex_names = train_names
            elif split == "val":
                complex_names = val_names
            elif split == "test":
                complex_names = test_names
            else:
                complex_names = train_names + val_names + test_names

        pdb_dict = []
        for name in tqdm(complex_names, desc="Loading dataset"):
            name = name.split(".pdb", 1)[0] + ".pdb"
            name = name.replace("|", ":")
            path = os.path.join(pdb_dir, name)
            pdb_dict.append(parse_PDB(path)[0])

        # Read ddG data
        self.ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
        self.ddG_data = {}

        df_2 = pd.read_csv(csv_path, low_memory=False)
        dataset_3 = df_2[df_2["ddG_ML"] != "-"]
        dataset_3_noindel = dataset_3.loc[
            ~dataset_3.mut_type.str.contains("ins") & ~dataset_3.mut_type.str.contains("del"), :
        ].reset_index(drop=True)

        for name in complex_names:
            cleaned_name = name.split(".pdb", 1)[0] + ".pdb"
            cleaned_name = cleaned_name.replace("|", ":")
            self.ddG_data[cleaned_name] = dataset_3_noindel[
                (dataset_3_noindel["WT_name"] == name) & (dataset_3_noindel["mut_type"] != "wt")
            ]

        for name, mut_df in self.ddG_data.items():
            self.ddG_data[name] = {
                "mut_seqs": mut_df["aa_seq"].to_list(),
                "ddG": -mut_df["ddG_ML"].to_numpy(
                    dtype=np.float32
                ),  # Megascale uses the convention that a negative value (ddG < 0) represents a destabilizing mutation. Flip it here.
            }

        # Featurize mutations as sequences
        for name, mut_df in self.ddG_data.items():
            index_matrix = []
            for s in mut_df["mut_seqs"]:
                indices = np.asarray([self.ALPHABET.index(a) for a in s], dtype=np.int64)
                index_matrix.append(indices)
            index_matrix = np.vstack(index_matrix)
            self.ddG_data[name]["mut_seqs"] = torch.from_numpy(index_matrix)
            self.ddG_data[name]["ddG"] = torch.tensor(mut_df["ddG"])

        # Mask all input chains
        for dict in pdb_dict:
            dict["masked_list"] = ["A"]
            dict["visible_list"] = []

        self.structure_data = StructureDataset(pdb_dict)

    def __len__(self):
        return len(self.structure_data)

    def __getitem__(self, idx):

        complex = self.structure_data[idx]
        pdb_name = complex["name"]
        complex_mut_seqs = self.ddG_data[f"{pdb_name}.pdb"]["mut_seqs"]
        ddG = self.ddG_data[f"{pdb_name}.pdb"]["ddG"]

        return {
            "name": pdb_name,
            "complex": complex,
            "complex_mut_seqs": complex_mut_seqs,
            "ddG": ddG,
        }
