"""Model utilities."""

import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from dateutil import parser


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


# The following gather functions
def gather_edges(edges, neighbor_idx):
    """Gather edges."""
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    """Gather nodes."""
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(*neighbor_idx.shape[:3], -1)
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    """Gather nodes t."""
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """Concatenate neighbors and nodes."""
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    """Encoder layer."""

    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        """Initialize the encoder layer."""
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer."""
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    """Decoder layer."""

    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        """Initialize the decoder layer."""
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer."""
        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed forward network."""

    def __init__(self, num_hidden, num_ff):
        """Initialize."""
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        """Forward pass."""
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(nn.Module):
    """Positional encodings."""

    def __init__(self, num_embeddings, max_relative_feature=32):
        """Initialize."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        """Forward pass."""
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1).to(
            self.linear.weight.dtype
        )
        E = self.linear(d_onehot)
        return E


class ProteinFeatures(nn.Module):
    """Extract protein features."""

    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        num_chain_embeddings=16,
    ):
        """Extract protein features."""
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels, backbone_noise=None):
        """Forward pass."""
        if backbone_noise is not None:
            X = X + backbone_noise
        elif self.training:
            # randomly sample backbone noise
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class ProteinMPNN(nn.Module):
    """Protein MPNN model."""

    def __init__(
        self,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=32,
        augment_eps=0.1,
        dropout=0.1,
    ):
        """Initialize."""
        super().__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(
            node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps
        )

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        fix_order=None,
        fix_backbone_noise=None,
    ):
        """Graph-conditioned sequence model."""
        device = X.device
        # Prepare node and edge embeddings
        # Checkpoint the expensive feature extraction during training
        if self.training:
            E, E_idx = torch.utils.checkpoint.checkpoint(
                self.features,
                X,
                mask,
                residue_idx,
                chain_encoding_all,
                fix_backbone_noise,
                use_reentrant=False,
            )
        else:
            E, E_idx = self.features(
                X, mask, residue_idx, chain_encoding_all, backbone_noise=fix_backbone_noise
            )

        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device).to(
            self.W_out.weight.dtype
        )
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(
                layer, h_V, h_E, E_idx, mask, mask_attend, use_reentrant=False
            )

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask  # update chain_M to include missing regions
        if fix_order is not None:
            decoding_order = fix_order  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        else:
            decoding_order = torch.argsort(
                (chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=device)))
            )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).to(self.W_out.weight.dtype)
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (
                1
                - torch.triu(
                    torch.ones(mask_size, mask_size, device=device, dtype=self.W_out.weight.dtype)
                )
            ),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask, use_reentrant=False)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
