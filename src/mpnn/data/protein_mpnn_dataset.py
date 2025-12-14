"""ProteinMPNN dataset."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch

from mpnn.common.constants import AA_ALPHABET
from mpnn.data.data_utils import entry_to_pyg_data, process_pdb


class StructureDataset:
    def __init__(
        self,
        pdb_dict_list,
        max_length=100,
    ):
        alphabet_set = set([a for a in AA_ALPHABET])
        discard_count = {"bad_chars": 0, "too_long": 0, "bad_seq_length": 0}

        self.data = []

        for entry in pdb_dict_list:
            seq = entry["seq"]

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry["seq"]) <= max_length:
                    self.data.append(entry_to_pyg_data(entry))
                else:
                    discard_count["too_long"] += 1
            else:
                discard_count["bad_chars"] += 1

        print("Discarded: ", discard_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader:
    def __init__(
        self,
        dataset,
        batch_size=100,
        shuffle=True,
        drop_last=False,
    ):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i].chain_seq_label) for i in range(self.size)]
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

            yield Batch.from_data_list(batch)


class PDBDataset(torch.utils.data.Dataset):
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


class PDBDatasetFlattened(torch.utils.data.Dataset):
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
