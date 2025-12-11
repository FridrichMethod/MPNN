import csv
import functools
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import numpy as np
import torch
import yaml
from dateutil import parser

from mpnn.typing_utils import StrPath

P = ParamSpec("P")
T = TypeVar("T")


_conda_not_installed_errmsg = "conda not installed"


def get_conda_prefix() -> str:
    """Attempts to find the root Conda folder. Works with miniforge3/miniconda3"""
    conda_root = os.getenv("CONDA_ROOT", None)
    if conda_root is None:
        # Attempt $CONDA_PREFIX_1 or $CONDA_PREFIX, depending
        # on whether the `base` environment is activated.
        default_env_name = os.getenv("CONDA_DEFAULT_ENV", None)
        assert default_env_name is not None, _conda_not_installed_errmsg
        conda_prefix_env_name = "CONDA_PREFIX" if default_env_name == "base" else "CONDA_PREFIX_1"
        conda_root = os.getenv(conda_prefix_env_name, None)
    assert conda_root is not None, _conda_not_installed_errmsg
    return conda_root


def clean_gpu_cache[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to clean GPU memory cache after the decorated function is executed."""

    counter = 0

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal counter
        try:
            result = func(*args, **kwargs)
        finally:
            # gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                counter += 1
        return result

    return wrapper


def norm_path(
    path: StrPath,
    expandvars: bool = True,
    expanduser: bool = True,
    resolve: bool = True,
) -> Path:
    """Normalize a file path.

    Args:
        path (StrPath): The file path to normalize.
        expandvars (bool, optional): Whether to expand environment variables. Defaults to True.
        expanduser (bool, optional): Whether to expand the user directory. Defaults to True.
        resolve (bool, optional): Whether to resolve the path. Defaults to True.

    Returns:
        Path: The normalized file path.
    """

    p = Path(path)
    if expandvars:
        p = Path(os.path.expandvars(p))
    if expanduser:
        p = p.expanduser()
    if resolve:
        p = p.resolve()

    return p


def seed_everything(seed: int = 0, freeze_cuda: bool = False) -> None:
    """Set the seed for all random number generators.

    Adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/seed.py
    Freeze CUDA for reproducibility if needed.

    Args:
        seed (int, optional): The seed value. Defaults to 0.
        freeze_cuda (bool, optional): Whether to freeze CUDA for reproducibility. Defaults to False.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if freeze_cuda:
        # nonrandom CUDNN convolution algo, maybe slower
        torch.backends.cudnn.deterministic = True
        # nonrandom selection of CUDNN convolution, maybe slower
        torch.backends.cudnn.benchmark = False


def load_config(config_path: StrPath) -> dict[str, Any]:
    """Load a config file."""
    config_path = norm_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file {config_path} not found")

    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def enable_tf32_if_available(precision: str = "high") -> bool:
    """Enable TF32 on Ampere+ CUDA GPUs for matmul (cuBLAS) and conv (cuDNN).

    On non-Ampere or non-CUDA setups it safely does nothing (and disables flags).

    Returns:
        bool: True if TF32 is enabled, False otherwise.
    """

    enabled = False

    # Default to strict FP32 unless we detect Ampere+
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Highest = strict FP32; High = allow TF32 on Ampere for matmul
    torch.set_float32_matmul_precision("highest")

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:  # Ampere(8.x) / Hopper(9.x)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision(precision)

            enabled = True

    return enabled


def parse_amp_dtype(amp_dtype: str | None) -> torch.dtype | None:
    """Convert a user string (or None) to a torch.dtype (or None) for autocast.

    Args:
        amp_dtype (str | None): The string to parse.

    Returns:
        torch.dtype | None: The parsed torch.dtype.

    Note:
        Accepted strings (case-insensitive):
        - 'fp16', 'float16', 'half'
        - 'bf16', 'bfloat16'
        - 'fp32', 'float32'

    """
    if amp_dtype is None:
        return None

    amp_dtype = amp_dtype.strip().lower()

    table = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }

    try:
        return table[amp_dtype]
    except KeyError as e:
        raise ValueError(
            f"Unsupported amp dtype string: {amp_dtype!r}. "
            "Use one of: fp16/float16/half, bf16/bfloat16, fp32/float32/float."
        ) from e


class StructureDataset:
    def __init__(
        self, pdb_dict_list, truncate=None, max_length=100, alphabet="ACDEFGHIKLMNPQRSTVWYX"
    ):
        alphabet_set = set([a for a in alphabet])
        discard_count = {"bad_chars": 0, "too_long": 0, "bad_seq_length": 0}

        self.data = []

        for _, entry in enumerate(pdb_dict_list):
            seq = entry["seq"]

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry["seq"]) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count["too_long"] += 1
            else:
                discard_count["bad_chars"] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

        print("Discarded: ", discard_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader:
    def __init__(
        self, dataset, batch_size=100, shuffle=True, collate_fn=lambda x: x, drop_last=False
    ):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]["seq"]) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
            else:
                clusters.append(batch)
                batch = []
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def worker_init_fn(worker_id):
    np.random.seed()


def process_pdb(t):
    init_alphabet = list("ACDEFGHIKLMNPQRSTVWYXabcdefghijklmnopqrstuvwxyz")
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    my_dict = {}
    concat_seq = ""
    mask_list = []
    visible_list = []
    if len(list(np.unique(t["idx"]))) >= 352:
        return None
    for idx in list(np.unique(t["idx"])):
        letter = chain_alphabet[idx]
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


class PDB_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, dict, params, max_length=10000):
        self.IDs = IDs
        self.dict = dict
        self.loader = loader
        self.params = params
        self.max_length = max_length

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.dict[ID]))
        out = self.loader(self.dict[ID][sel_idx], self.params)
        if "label" not in out:
            return None
        out = process_pdb(out)
        if not out or len(out["seq"]) > self.max_length:
            return None
        return out


class flattened_PDB_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, dict, params, max_length=10000):
        self.IDs = IDs
        self.dict = dict
        self.loader = loader
        self.params = params
        self.max_length = max_length
        self.flattened_ids = []
        for ID in self.IDs:
            for idx in range(len(self.dict[ID])):
                self.flattened_ids.append(self.dict[ID][idx])

    def __len__(self):
        return len(self.flattened_ids)

    def __getitem__(self, index):
        out = self.loader(self.flattened_ids[index], self.params)
        if "label" not in out:
            return None
        out = process_pdb(out)
        if not out or len(out["seq"]) > self.max_length:
            return None
        return out


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


def build_training_clusters(params, debug=False):
    val_ids = set([int(l) for l in open(params["VAL"]).readlines()])
    test_ids = set([int(l) for l in open(params["TEST"]).readlines()])

    if debug:
        val_ids = []
        test_ids = []

    # read & clean list.csv
    with open(params["LIST"]) as f:
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

    if debug:
        rows = rows[:20]
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
    if debug:
        valid = train
    return train, valid, test
