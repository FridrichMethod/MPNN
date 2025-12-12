"""Regression test for autoregressive mask construction."""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch

from mpnn.protein_mpnn import build_autoregressive_mask


def build_autoregressive_mask_legacy(
    chain_seq_label: torch.Tensor,
    chain_mask_all: torch.Tensor,
    mask: torch.Tensor,
    batch: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Ground-truth dense autoregressive mask used for regression checks."""
    device = chain_seq_label.device
    batch_chain_mask_all, _ = to_dense_batch(chain_mask_all * mask, batch)  # [B, N]

    # 0 - visible - encoder, 1 - masked - decoder
    noise = torch.abs(torch.randn(batch_chain_mask_all.shape, device=device))
    decoding_order = torch.argsort((batch_chain_mask_all + 1e-4) * noise)
    mask_size = batch_chain_mask_all.size(1)
    permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=mask_size).float()
    order_mask_backward = torch.einsum(
        "ij, biq, bjp->bqp",
        1 - torch.triu(torch.ones(mask_size, mask_size, device=device)),
        permutation_matrix_reverse,
        permutation_matrix_reverse,
    )
    adj = to_dense_adj(edge_index, batch)
    mask_attend = order_mask_backward[adj.bool()].unsqueeze(-1)

    return mask_attend


def test_autoregressive_mask_matches_legacy() -> None:
    """Ensure new sparse mask matches legacy dense construction."""
    chain_seq_label = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    mask = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float)
    chain_mask_all = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 0, 2, 1, 3, 3, 4, 4],
            [0, 1, 0, 1, 2, 0, 2, 1, 2, 3, 4, 3, 4],
        ],
        dtype=torch.long,
    )

    torch.manual_seed(42)

    mask_new = build_autoregressive_mask(
        chain_seq_label,
        chain_mask_all,
        mask,
        batch,
        edge_index,
    )
    mask_legacy = build_autoregressive_mask_legacy(
        chain_seq_label,
        chain_mask_all,
        mask,
        batch,
        edge_index,
    )

    torch.testing.assert_close(mask_new, mask_legacy)


def test_autoregressive_mask_uses_less_memory_than_legacy() -> None:
    """New sparse mask avoids the dense adjacency allocation from the legacy code."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for reliable memory measurements.")

    device = torch.device("cuda")
    batch_size = 2
    nodes_per_graph = 1024
    total_nodes = batch_size * nodes_per_graph

    chain_seq_label = torch.zeros(total_nodes, dtype=torch.long, device=device)
    mask = torch.ones(total_nodes, dtype=torch.float, device=device)
    chain_mask_all = torch.zeros(total_nodes, dtype=torch.float, device=device)
    batch = torch.repeat_interleave(torch.arange(batch_size, device=device), nodes_per_graph)

    # fully connected (including self-loops) within each graph
    idx = torch.arange(nodes_per_graph, device=device)
    row_list = []
    col_list = []
    for g in range(batch_size):
        base = g * nodes_per_graph
        row_block, col_block = torch.meshgrid(idx + base, idx + base, indexing="ij")
        row_list.append(row_block.reshape(-1))
        col_list.append(col_block.reshape(-1))
    row = torch.cat(row_list)
    col = torch.cat(col_list)
    edge_index = torch.stack([row, col], dim=0)

    def legacy():
        return build_autoregressive_mask_legacy(
            chain_seq_label, chain_mask_all, mask, batch, edge_index
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        legacy()
    torch.cuda.synchronize()
    legacy_peak = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        build_autoregressive_mask(chain_seq_label, chain_mask_all, mask, batch, edge_index)
    torch.cuda.synchronize()
    new_peak = torch.cuda.max_memory_allocated()

    # Expect no regression; allow small overhead tolerance
    assert new_peak <= legacy_peak * 1.1, f"New mask used {new_peak} vs legacy {legacy_peak} bytes"


def test_autoregressive_mask_random_parity_multiple_graphs() -> None:
    """Randomized parity across multiple graphs with varying masks."""
    seeds = [0, 1, 2]
    graph_sizes = [3, 4, 5]

    for seed in seeds:
        torch.manual_seed(seed)
        # build batch
        node_counts = torch.tensor(graph_sizes)
        total_nodes = int(node_counts.sum().item())
        batch = torch.cat([
            torch.full((n,), i, dtype=torch.long) for i, n in enumerate(graph_sizes)
        ])

        chain_seq_label = torch.arange(total_nodes, dtype=torch.long) % 4
        mask = torch.ones(total_nodes, dtype=torch.float)
        chain_mask_all = torch.randint(0, 2, (total_nodes,), dtype=torch.float)

        # fully connected per graph (including self edges)
        rows, cols = [], []
        offset = 0
        for n in graph_sizes:
            idx = torch.arange(n)
            r, c = torch.meshgrid(idx + offset, idx + offset, indexing="ij")
            rows.append(r.reshape(-1))
            cols.append(c.reshape(-1))
            offset += n
        edge_index = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)

        torch.manual_seed(seed)
        mask_new = build_autoregressive_mask(
            chain_seq_label, chain_mask_all, mask, batch, edge_index
        )
        torch.manual_seed(seed)
        mask_legacy = build_autoregressive_mask_legacy(
            chain_seq_label, chain_mask_all, mask, batch, edge_index
        )

        torch.testing.assert_close(mask_new, mask_legacy)
