"""ProteinMPNN dataset."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from itertools import chain

import numpy as np
from torch.utils.data import Dataset, Sampler
from torch_geometric.loader import DataLoader

from mpnn.common.constants import AA_ALPHABET
from mpnn.data.data_utils import entry_to_pyg_data, process_pdb
from mpnn.utils import get_logger

logger = get_logger(__name__)


class PDBDataset(Dataset):
    """Wrap raw PDB entries; optionally flatten all entries for deterministic access."""

    def __init__(
        self,
        ids: Sequence[str],
        loader: Callable[[dict, dict], dict],
        pdb_dict: dict,
        params: dict,
        max_length: int = 10000,
        flattened: bool = False,
    ):
        """Store references and optionally build a flattened entry list."""
        self.ids = ids
        self.pdb_dict = pdb_dict
        self.loader = loader
        self.params = params
        self.max_length = max_length
        self.flattened = flattened

        if self.flattened:
            self.flat_entries = list(
                chain.from_iterable(self.pdb_dict[pdb_id] for pdb_id in self.ids)
            )

    def __len__(self) -> int:
        """Return number of IDs or flattened entries."""
        if self.flattened:
            return len(self.flat_entries)
        return len(self.ids)

    def __getitem__(self, index: int) -> dict | None:
        """Load, process, and filter a single PDB entry."""
        if self.flattened:
            entry = self.flat_entries[index]
        else:
            idx = self.ids[index]
            sel_idx = np.random.randint(0, len(self.pdb_dict[idx]))
            entry = self.pdb_dict[idx][sel_idx]

        out = self.loader(entry, self.params)
        if "label" not in out:
            return None
        out = process_pdb(out)
        if not out or len(out["seq"]) > self.max_length:
            return None
        return out


class StructureDataset(Dataset):
    """Filter PDB dicts by length/alphabet and expose PyG Data items."""

    def __init__(
        self,
        entry_list: Iterable[dict],
        max_length: int = 10000,
    ):
        """Filter invalid/long sequences and cache lengths for batching."""
        alphabet_set = set([a for a in AA_ALPHABET])
        discard_count = {"bad_chars": 0, "too_long": 0, "bad_seq_length": 0}

        self.data = []
        self.lengths = []

        for entry in entry_list:
            seq = entry["seq"]

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry["seq"]) <= max_length:
                    self.lengths.append(len(entry["seq"]))
                    self.data.append(entry_to_pyg_data(entry))
                else:
                    discard_count["too_long"] += 1
            else:
                discard_count["bad_chars"] += 1

        logger.info("Discarded: %s", discard_count)

    def __len__(self) -> int:
        """Return number of retained structures."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Return a pre-converted PyG Data item."""
        return self.data[idx]


class LengthBatchSampler(Sampler[list[int]]):
    """Length-aware batch sampler that mimics the original clustering logic."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int = 100,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """Pre-compute length-clustered batches."""
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.clusters = self._build_clusters()

    def _build_clusters(self) -> list[list[int]]:
        sorted_ix = np.argsort(self.lengths)
        clusters: list[list[int]] = []
        batch: list[int] = []
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(int(ix))
            else:
                clusters.append(batch)
                batch = [int(ix)]
        if len(batch) > 0:
            clusters.append(batch)
        return clusters

    def __iter__(self):
        """Yield index batches, shuffling batch order each epoch."""
        clusters = list(self.clusters)
        if self.shuffle:
            np.random.shuffle(clusters)
        yield from clusters

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.clusters)


class StructureLoader(DataLoader):
    """DataLoader wrapper that enforces a custom batch sampler for PyG Data."""

    def __init__(
        self,
        dataset: Dataset,
        batch_sampler: Sampler[list[int]],
        **dataloader_kwargs,
    ):
        """Initialize with a user-provided batch sampler and safe defaults."""
        # Prevent incompatible overrides that would break custom batch_sampler logic
        forbidden = {"batch_sampler", "sampler", "shuffle", "drop_last", "batch_size"}
        bad_keys = forbidden.intersection(dataloader_kwargs.keys())
        if bad_keys:
            raise ValueError(f"StructureLoader manages {bad_keys}; please remove them from kwargs.")

        super().__init__(
            dataset,
            batch_size=1,  # PyTorch requires batch_size=1 when batch_sampler is set
            shuffle=False,
            sampler=None,
            batch_sampler=batch_sampler,
            drop_last=False,
            **dataloader_kwargs,
        )
