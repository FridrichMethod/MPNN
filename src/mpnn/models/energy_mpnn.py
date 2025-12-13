import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device
from torch_geometric.data import Batch


class EnergyMPNN(nn.Module):
    def __init__(
        self,
        protein_mpnn,
        use_antithetic_variates=True,
        device: Device = "cuda",
    ):
        super().__init__()
        self.protein_mpnn = protein_mpnn
        self.use_antithetic_variates = (
            use_antithetic_variates  # TODO: Implement antithetic variates
        )
        self.device = device

    def folding_dG(self, domain, seqs):
        """Predicts the folding stability (dG) for a list of sequences."""
        B = seqs.shape[0]

        # domain is Data object. Repeat it B times.
        batch_data = Batch.from_data_list([domain] * B).to(self.device)
        S_ = seqs.to(self.device)
        S_flat = S_.view(-1)

        log_probs = self.protein_mpnn(
            x=batch_data.x,
            chain_seq_label=S_flat,
            mask=batch_data.mask,
            chain_mask_all=batch_data.chain_mask_all,
            residue_idx=batch_data.residue_idx,
            chain_encoding_all=batch_data.chain_encoding_all,
            batch=batch_data.batch,
        )

        # log_probs: [B*L, 21]. Reshape to [B, L, 21] (assuming equal lengths)
        L = seqs.shape[1]
        log_probs = log_probs.view(B, L, 21)
        seq_oh = F.one_hot(S_, 21).float()
        dG = torch.sum(seq_oh * log_probs, dim=(1, 2))

        return dG

    def folding_ddG(self, domain, mut_seqs, wt_seq=None):
        """Predicts the folding ddG."""
        if wt_seq is None:
            wt_seq = domain.chain_seq_label.unsqueeze(0)
        wt_dG = self.folding_dG(domain, wt_seq)
        mut_dG = self.folding_dG(domain, mut_seqs)

        ddG = mut_dG - wt_dG

        # use the convention that a negative value (ddG < 0) represents a stabilizing mutation.
        return -ddG

    def folding_dG_batched(self, domain_list, seqs_list):
        """Batched version of the folding_dG function, with multiple complexes and mutated sequences.
        domain_list: list of Data objects
        seqs_list: list of tensors [Mi, Li]
        decoding_order: list of tensors [Li]
        backbone_noise: list of tensors [Li, 4, 3]
        """
        repeated_domains = []
        for domain, seqs in zip(domain_list, seqs_list):
            repeated_domains.extend([domain] * seqs.shape[0])

        batch_data = Batch.from_data_list(repeated_domains).to(self.device)
        S_flat = torch.cat([s.view(-1) for s in seqs_list]).to(self.device)

        log_probs = self.protein_mpnn(
            x=batch_data.x,
            chain_seq_label=S_flat,
            mask=batch_data.mask,
            chain_mask_all=batch_data.chain_mask_all,
            residue_idx=batch_data.residue_idx,
            chain_encoding_all=batch_data.chain_encoding_all,
            batch=batch_data.batch,
        )

        seq_oh = F.one_hot(S_flat, 21).float()
        per_node_score = torch.sum(seq_oh * log_probs, dim=-1)

        dG = torch.zeros(len(repeated_domains), device=self.device)
        dG.scatter_add_(0, batch_data.batch, per_node_score)

        return dG

    def folding_ddG_batched(self, domain_list, mut_seqs_list):
        """Batched version of the folding_ddG function, with multiple complexes and mutated sequences."""
        wt_seq_list = [domain.chain_seq_label.unsqueeze(0) for domain in domain_list]

        wt_dG = self.folding_dG_batched(domain_list, wt_seq_list)
        mut_dG = self.folding_dG_batched(domain_list, mut_seqs_list)

        num_muts = torch.tensor([s.shape[0] for s in mut_seqs_list], device=self.device)
        wt_dG_expanded = wt_dG.repeat_interleave(num_muts)

        ddG = mut_dG - wt_dG_expanded

        # use the convention that a negative value (ddG < 0) represents a stabilizing mutation.
        return -ddG

    def binding_ddG(
        self,
        protein_complex,
        binder1,
        binder2,
        complex_mut_seqs,
        binder1_mut_seqs,
        binder2_mut_seqs,
    ):
        """We calculate the binding ddG by decomposing it into three folding ddG terms,
        corresponding to the entire complex and each individual binders.
        """
        complex_ddG_fold = self.folding_ddG(protein_complex, complex_mut_seqs)
        binder1_ddG_fold = self.folding_ddG(binder1, binder1_mut_seqs)
        binder2_ddG_fold = self.folding_ddG(binder2, binder2_mut_seqs)

        ddG = complex_ddG_fold - (binder1_ddG_fold + binder2_ddG_fold)

        return ddG

    forward = binding_ddG
