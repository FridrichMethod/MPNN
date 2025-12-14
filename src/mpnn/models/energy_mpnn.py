"""Energy MPNN model module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device

from mpnn.data.data_utils import featurize
from mpnn.models.model_utils import ProteinMPNN


class EnergyMPNN(nn.Module):
    """Energy-based MPNN model."""

    def __init__(
        self,
        protein_mpnn: ProteinMPNN,
        use_antithetic_variates: bool = True,
        device: Device = "cuda",
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.protein_mpnn = protein_mpnn
        self.use_antithetic_variates = (
            use_antithetic_variates  # TODO: Implement antithetic variates
        )
        self.device = device

    def folding_dG(self, domain: dict, seqs: torch.Tensor) -> torch.Tensor:
        """Predicts the folding stability (dG) for a list of sequences."""
        B = seqs.shape[0]

        # Featurize the domain (single structure)
        # featurize expects a list of dicts. We can featurize one and repeat, or list of one.
        X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
            [domain], device=self.device
        )

        # X: [1, L, 4, 3] -> Repeat for B
        X = X.repeat(B, 1, 1, 1)
        # mask: [1, L] -> Repeat for B
        mask = mask.repeat(B, 1)
        # chain_M: [1, L] -> Repeat for B
        chain_M = chain_M.repeat(B, 1)
        # residue_idx: [1, L] -> Repeat for B
        residue_idx = residue_idx.repeat(B, 1)
        # chain_encoding_all: [1, L] -> Repeat for B
        chain_encoding_all = chain_encoding_all.repeat(B, 1)

        # seqs is [B, L] containing the mutant sequences
        S_ = seqs.to(self.device)

        log_probs = self.protein_mpnn(
            X=X,
            S=S_,
            mask=mask,
            chain_M=chain_M,
            residue_idx=residue_idx,
            chain_encoding_all=chain_encoding_all,
        )

        # log_probs: [B, L, 21]
        seq_oh = F.one_hot(S_, 21).float()
        dG = torch.sum(seq_oh * log_probs, dim=(1, 2))

        return dG

    def folding_ddG(
        self, domain: dict, mut_seqs: torch.Tensor, wt_seq: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predicts the folding ddG."""
        if wt_seq is None:
            # Domain dict should have the WT sequence as "seq" or we can featurize to get it?
            # featurize returns S which is the WT sequence indices if provided in dict
            _, S_wt, _, _, _, _, _, _ = featurize([domain], device=self.device)
            wt_seq = S_wt  # [1, L]

        wt_dG = self.folding_dG(domain, wt_seq)
        mut_dG = self.folding_dG(domain, mut_seqs)

        ddG = mut_dG - wt_dG

        # use the convention that a negative value (ddG < 0) represents a stabilizing mutation.
        return -ddG

    def folding_dG_batched(
        self, domain_list: list[dict], seqs_list: list[torch.Tensor]
    ) -> torch.Tensor:
        """Batched version of the folding_dG function, with multiple complexes and mutated sequences.

        domain_list: list of dicts
        seqs_list: list of tensors [Mi, Li]
        """
        # We need to process each domain and its mutants.
        # Since ProteinMPNN is dense, we can't easily batch different length proteins unless we pad them.
        # But featurize handles padding for a batch of structures.

        # However, here we have multiple mutants per structure.
        # domain_i has M_i mutants.
        # We can flatten everything into a single large batch of [Sum(M_i), L_max, ...].
        # Or loop over domains if they are too different/large.

        # For efficiency with featurize, we construct a large list of (structure, sequence) pairs?
        # featurize takes a list of structures (dicts). It extracts sequence from dict["seq"].
        # But we have mutant sequences in seqs_list.

        # Strategy:
        # 1. Expand domain_list to match seqs_list counts.
        # 2. Call featurize on the expanded list of domains.
        # 3. Replace the 'S' returned by featurize with our mutant sequences.

        expanded_domains = []
        expanded_seqs = []
        batch_indices = []  # to aggregate back if needed (not needed for dG calculation per seq)

        for i, (domain, seqs) in enumerate(zip(domain_list, seqs_list)):
            k = seqs.shape[0]
            expanded_domains.extend([domain] * k)
            expanded_seqs.append(seqs)

        S_mutants = torch.cat(expanded_seqs, dim=0).to(self.device)  # [Total_M, L_?]
        # Problem: sequences might have different lengths if domains vary.
        # But dense packing requires same length (padding).
        # featurize handles padding.
        # S_mutants from caller (mgnify_train_epoch) is a list of [Mi, Li].
        # If we cat them, they must have same dimension or we pad?
        # mgnify_train_epoch collects batches such that:
        # num_seqs * max_len < batch_size
        # But it doesn't guarantee same length.
        # Actually mgnify_train_epoch logic:
        # batch_mut_seqs_list.append(sample["complex_mut_seqs"])
        # sample["complex_mut_seqs"] is [M, L].
        # If L differs between samples, we cannot torch.cat them directly without padding.

        # If we rely on featurize, it pads X, S, mask, etc. to L_max of the batch.
        # But we need to supply the mutant sequences S aligned to that padding.

        # Let's trust featurize to pad X, mask etc.
        # We need to pad S_mutants manually to match L_max determined by featurize?
        # OR, we construct 'dummy' dicts with the mutant sequences and pass to featurize?
        # But featurize extracts coordinates. We want coords from WT, seq from mutant.

        # Simpler approach: Process one domain at a time (or group by domain) if batching heterogeneous proteins is hard.
        # But we want to utilize GPU parallelism.

        # Correct approach with featurize:
        X, S_wt, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
            expanded_domains, device=self.device
        )

        # S_wt is [Total_M, L_max].
        # S_mutants is what we want to use.
        # We need to pad S_mutants to [Total_M, L_max] and put it into the graph.
        # Since S_mutants comes from list of tensors, we pad them.

        L_max = X.shape[1]
        padded_seqs = []
        for seqs in seqs_list:
            # seqs: [Mi, Li]
            # Pad dim 1 to L_max
            p = L_max - seqs.shape[1]
            if p > 0:
                seqs = F.pad(seqs, (0, p), value=20)  # 20 is 'X' or gap? typically 20 in vocab
            elif p < 0:
                # Should not happen if featurize determined L_max from domains and seqs are same length
                pass
            padded_seqs.append(seqs)

        S_in = torch.cat(padded_seqs, dim=0).to(self.device)  # [Total_M, L_max]

        log_probs = self.protein_mpnn(
            X=X,
            S=S_in,
            mask=mask,
            chain_M=chain_M,
            residue_idx=residue_idx,
            chain_encoding_all=chain_encoding_all,
        )

        seq_oh = F.one_hot(S_in, 21).float()
        per_node_score = torch.sum(seq_oh * log_probs, dim=-1)  # [Total_M, L_max]

        # Sum over length (masked by mask which handles padding)
        dG = torch.sum(per_node_score * mask, dim=-1)  # [Total_M]

        # But wait, mgnify_train_epoch expects a single scalar or tensor?
        # mgnify_train_epoch calls: pred = model.folding_ddG_batched(batch_complex_list, batch_mut_seqs_list)
        # And compares to ddG (concatenated).
        # So we return the flat dG vector.

        return dG

    def folding_ddG_batched(
        self, domain_list: list[dict], mut_seqs_list: list[torch.Tensor]
    ) -> torch.Tensor:
        """Batched version of the folding_ddG function, with multiple complexes and mutated sequences."""
        # Calculate WT dG
        # For each domain, we need its WT sequence.
        # We can extract it via featurize or from the dict if we trust it matches the coords length.

        # We can use folding_dG_batched but we need to pass WT sequences.
        # Each domain has 1 WT sequence. We have M mutants.
        # We need to repeat WT sequence M times to subtract.

        # Or efficiently: calculate WT dG once per domain.

        # 1. Calculate WT dG for each domain (batch size = num_domains)
        # We pass dummy sequences to folding_dG_batched just to get X?
        # No, we can pass [domain] and [wt_seq] for each.

        # Extract WT seqs (as tensors)
        # featurize(domain_list) gives us S_wt [N_domains, L_max]
        X, S_wt, mask, lengths, chain_M, residue_idx, _, chain_encoding_all = featurize(
            domain_list, device=self.device
        )

        log_probs_wt = self.protein_mpnn(
            X=X,
            S=S_wt,
            mask=mask,
            chain_M=chain_M,
            residue_idx=residue_idx,
            chain_encoding_all=chain_encoding_all,
        )
        seq_oh_wt = F.one_hot(S_wt, 21).float()
        dG_wt = torch.sum(torch.sum(seq_oh_wt * log_probs_wt, dim=-1) * mask, dim=-1)  # [N_domains]

        # 2. Calculate Mutant dG
        dG_mut = self.folding_dG_batched(domain_list, mut_seqs_list)  # [Total_Mutants]

        # 3. Expand WT dG to match mutants
        num_muts = torch.tensor([s.shape[0] for s in mut_seqs_list], device=self.device)
        dG_wt_expanded = dG_wt.repeat_interleave(num_muts)

        ddG = dG_mut - dG_wt_expanded

        # use the convention that a negative value (ddG < 0) represents a stabilizing mutation.
        return -ddG

    def binding_ddG(
        self,
        protein_complex: dict,
        binder1: dict,
        binder2: dict,
        complex_mut_seqs: torch.Tensor,
        binder1_mut_seqs: torch.Tensor,
        binder2_mut_seqs: torch.Tensor,
    ) -> torch.Tensor:
        """We calculate the binding ddG by decomposing it into three folding ddG terms.

        Corresponding to the entire complex and each individual binders.
        """
        complex_ddG_fold = self.folding_ddG(protein_complex, complex_mut_seqs)
        binder1_ddG_fold = self.folding_ddG(binder1, binder1_mut_seqs)
        binder2_ddG_fold = self.folding_ddG(binder2, binder2_mut_seqs)

        ddG = complex_ddG_fold - (binder1_ddG_fold + binder2_ddG_fold)

        return ddG

    forward = binding_ddG
