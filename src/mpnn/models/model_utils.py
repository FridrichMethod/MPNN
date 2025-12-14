"""Model utilities."""

from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


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
        edge_cutoff=None,
        use_virtual_center=False,
        occupancy_cutoff=None,
    ):
        """Extract protein features."""
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.edge_cutoff = edge_cutoff
        self.use_virtual_center = use_virtual_center
        self.occupancy_cutoff = occupancy_cutoff

        self.embeddings = PositionalEncodings(num_positional_embeddings)

        num_atoms = 6 if use_virtual_center else 5
        edge_in = num_positional_embeddings + num_rbf * (num_atoms**2)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _get_virtual_center(self, Ca, Cb, N):
        """Calculate virtual center coordinates."""
        l_virtual = 3.06  # 2 * 1.53 Angstrom

        u_bc = Cb - Ca
        u_bc = u_bc / (torch.norm(u_bc, dim=-1, keepdim=True) + 1e-6)

        u_nc = N - Ca
        u_nc = u_nc / (torch.norm(u_nc, dim=-1, keepdim=True) + 1e-6)

        n_plane = torch.cross(u_nc, u_bc, dim=-1)
        n_plane = n_plane / (torch.norm(n_plane, dim=-1, keepdim=True) + 1e-6)

        v_dir = -torch.cross(n_plane, u_bc, dim=-1)
        V = Ca + l_virtual * v_dir
        return V

    def _compute_occupancy(self, V, mask):
        """Compute occupancy feature."""
        if self.occupancy_cutoff is None:
            return None

        # V: [B, L, 3]
        # mask: [B, L]

        # Calculate pairwise distances
        # diff: [B, L, L, 3]
        diff = V.unsqueeze(2) - V.unsqueeze(1)
        d_sq = torch.sum(diff**2, dim=-1)  # [B, L, L]

        # Mask
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, L, L]

        # We also need to mask out self-loops (j != i)
        diag_mask = 1.0 - torch.eye(V.size(1), device=V.device).unsqueeze(0)  # [1, L, L]

        valid_mask = mask_2d * diag_mask

        # Cutoff check
        cutoff_mask = (d_sq < (self.occupancy_cutoff**2)).float()

        # Neighbors
        # weights w_j = 1.0
        sigma = self.occupancy_cutoff / 3.0

        occ = torch.exp(-d_sq / (2 * sigma**2))
        occ = occ * valid_mask * cutoff_mask

        occupancy = torch.sum(occ, dim=2, keepdim=True)  # [B, L, 1]

        # Normalization (log1p)
        occupancy = torch.log1p(occupancy)

        return occupancy

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max

        if self.edge_cutoff is not None:
            cutoff_mask = (self.edge_cutoff > D).float()
            D_adjust = D_adjust + (1.0 - cutoff_mask) * 1e6

        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )

        edge_mask = None
        if self.edge_cutoff is not None:
            edge_mask = (D_neighbors < self.edge_cutoff).float()

        return D_neighbors, E_idx, edge_mask

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

        atoms_list = [N, Ca, C, O, Cb]

        if self.use_virtual_center:
            V = self._get_virtual_center(Ca, Cb, N)
            atoms_list.append(V)

        D_neighbors, E_idx, edge_mask = self._dist(Ca, mask)

        occupancy = None
        if self.occupancy_cutoff is not None:
            # We need V for occupancy. If not already computed:
            V_occ = (
                atoms_list[-1] if self.use_virtual_center else self._get_virtual_center(Ca, Cb, N)
            )
            occupancy = self._compute_occupancy(V_occ, mask)

        RBF_all = []
        # Use loop for all pairs
        for atom1, atom2 in product(atoms_list, repeat=2):
            RBF_all.append(self._get_rbf(atom1, atom2, E_idx))

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
        return E, E_idx, occupancy, edge_mask


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
        edge_cutoff=None,
        use_virtual_center=False,
        occupancy_cutoff=None,
    ):
        """Initialize."""
        super().__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(
            node_features,
            edge_features,
            top_k=k_neighbors,
            augment_eps=augment_eps,
            edge_cutoff=edge_cutoff,
            use_virtual_center=use_virtual_center,
            occupancy_cutoff=occupancy_cutoff,
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

        self.occupancy_mlp = None
        if occupancy_cutoff is not None:
            self.occupancy_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

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
            E, E_idx, occupancy, edge_mask = torch.utils.checkpoint.checkpoint(
                self.features,
                X,
                mask,
                residue_idx,
                chain_encoding_all,
                fix_backbone_noise,
                use_reentrant=False,
            )
        else:
            E, E_idx, occupancy, edge_mask = self.features(
                X, mask, residue_idx, chain_encoding_all, backbone_noise=fix_backbone_noise
            )

        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device).to(
            self.W_out.weight.dtype
        )

        if self.occupancy_mlp is not None and occupancy is not None:
            h_V = h_V + self.occupancy_mlp(occupancy)

        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        if edge_mask is not None:
            mask_attend = mask_attend * edge_mask

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

        if edge_mask is not None:
            # Apply edge_mask to autoregressive masks too?
            # mask_attend above for encoder was (mask_V_neighbor & edge_mask).
            # Here mask_attend is "visible based on order".
            # We should also AND it with edge_mask to respect connectivity.
            edge_mask_expanded = edge_mask.unsqueeze(-1)
            mask_bw = mask_bw * edge_mask_expanded
            mask_fw = mask_fw * edge_mask_expanded

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask, use_reentrant=False)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
