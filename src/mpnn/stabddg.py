import torch
import torch.nn as nn
import torch.nn.functional as F

from mpnn.model_utils import featurize


class StaBddG(nn.Module):
    def __init__(
        self, pmpnn, use_antithetic_variates=True, noise_level=0.1, l_to_r=False, device="cuda"
    ):
        super().__init__()
        self.pmpnn = pmpnn
        self.use_antithetic_variates = use_antithetic_variates
        self.noise_level = noise_level
        self.device = device
        self.l_to_r = l_to_r

    def get_wt_seq(self, domain):
        """Returns the wild type sequence of a protein."""
        _, wt_seq, *_ = featurize([domain], self.device)
        return wt_seq

    def folding_dG(self, domain, seqs, decoding_order=None, backbone_noise=None):
        """Predicts the folding stability (dG) for a list of sequences."""
        B = seqs.shape[0]

        X_, _, mask_, _, chain_M_, residue_idx_, _, chain_encoding_all_ = featurize(
            [domain], self.device
        )
        X_, S_, mask_ = X_.repeat(B, 1, 1, 1), seqs.to(self.device), mask_.repeat(B, 1)
        chain_M_ = chain_M_.repeat(B, 1)
        residue_idx_, chain_encoding_all_ = (
            residue_idx_.repeat(B, 1),
            chain_encoding_all_.repeat(B, 1),
        )

        order = decoding_order.repeat(B, 1) if decoding_order is not None else None
        backbone_noise = backbone_noise.repeat(B, 1, 1, 1) if backbone_noise is not None else None

        log_probs = self.pmpnn(
            X_,
            S_,
            mask_,
            chain_M_,
            residue_idx_,
            chain_encoding_all_,
            fix_order=order,
            fix_backbone_noise=backbone_noise,
        )

        seq_oh = torch.nn.functional.one_hot(seqs, 21).to(self.device)
        dG = torch.sum(seq_oh * log_probs, dim=(1, 2))

        return dG

    def folding_ddG(self, domain, mut_seqs, set_wt_seq=None):
        """Predicts the folding ddG."""
        X, wt_seq, _, _, chain_M, _, _, _ = featurize([domain], self.device)

        if set_wt_seq is not None:
            wt_seq = set_wt_seq

        decoding_order = self._get_decoding_order(chain_M) if self.use_antithetic_variates else None
        backbone_noise = self._get_backbone_noise(X) if self.use_antithetic_variates else None

        wt_dG = self.folding_dG(
            domain, wt_seq, decoding_order=decoding_order, backbone_noise=backbone_noise
        )
        mut_dG = self.folding_dG(
            domain, mut_seqs, decoding_order=decoding_order, backbone_noise=backbone_noise
        )

        ddG = mut_dG - wt_dG
        ddG *= (
            -1
        )  # use the convention that a negative value (ddG < 0) represents a stabilizing mutation.
        return ddG

    def folding_dG_batched(self, domain_list, seqs_list, decoding_order=None, backbone_noise=None):
        """Batched version of the folding_dG function, with multiple complexes and mutated sequences.
        domain_list: list of domains
        seqs_list: list of sequences
        decoding_order: list of decoding orders
        backbone_noise: list of backbone noises
        """
        B = len(domain_list)
        X_, _, mask_, _, chain_M_, residue_idx_, _, chain_encoding_all_ = featurize(
            domain_list, self.device
        )
        num_mut_seqs = torch.tensor([seqs.shape[0] for seqs in seqs_list]).to(self.device)

        # repeat the features to account for multiple mutated sequences
        X_, mask_, chain_M_ = (
            X_.repeat_interleave(num_mut_seqs, dim=0),
            mask_.repeat_interleave(num_mut_seqs, dim=0),
            chain_M_.repeat_interleave(num_mut_seqs, dim=0),
        )
        residue_idx_, chain_encoding_all_ = (
            residue_idx_.repeat_interleave(num_mut_seqs, dim=0),
            chain_encoding_all_.repeat_interleave(num_mut_seqs, dim=0),
        )

        # pad the mutated sequences to the same length
        max_len = max(t.size(1) for t in seqs_list)
        padded = []
        S_mask = []
        for t in seqs_list:
            n_i, L_i = t.shape
            pad_len = max_len - L_i

            # mask: ones on real positions, zeros on padded
            mask = torch.ones(n_i, max_len, dtype=torch.bool, device=t.device)
            if pad_len > 0:
                mask[:, L_i:] = 0
                t = F.pad(t, (0, pad_len))

            padded.append(t)
            S_mask.append(mask)

        S_mask = torch.cat(S_mask, dim=0).to(self.device)
        S_ = torch.cat(padded, dim=0).to(self.device)

        if self.use_antithetic_variates:
            backbone_noise = backbone_noise.repeat_interleave(num_mut_seqs, dim=0)
            decoding_order = decoding_order.repeat_interleave(num_mut_seqs, dim=0)
            backbone_noise = (
                backbone_noise * S_mask[..., None, None]
            )  # this line is to make sure that padded positions have no noise

        log_probs = self.pmpnn(
            X_,
            S_,
            mask_,
            chain_M_,
            residue_idx_,
            chain_encoding_all_,
            fix_order=decoding_order,
            fix_backbone_noise=backbone_noise,
        )

        seq_oh = torch.nn.functional.one_hot(S_, 21).to(self.device)
        dG_batched = torch.sum(seq_oh * log_probs * S_mask.unsqueeze(-1), dim=(1, 2))

        return dG_batched

    def folding_ddG_batched(self, domain_list, mut_seqs_list):
        """Batched version of the folding_ddG function, with multiple complexes and mutated sequences."""
        X, wt_seq, _, _, chain_M, _, _, _ = featurize(domain_list, self.device)

        decoding_order = self._get_decoding_order(chain_M) if self.use_antithetic_variates else None
        backbone_noise = self._get_backbone_noise(X) if self.use_antithetic_variates else None
        wt_seq = [self.get_wt_seq(domain) for domain in domain_list]

        wt_dG = self.folding_dG_batched(
            domain_list, wt_seq, decoding_order=decoding_order, backbone_noise=backbone_noise
        )
        mut_dG = self.folding_dG_batched(
            domain_list, mut_seqs_list, decoding_order=decoding_order, backbone_noise=backbone_noise
        )
        num_mut_seqs = torch.tensor([seqs.shape[0] for seqs in mut_seqs_list]).to(self.device)
        ddG_batched = mut_dG - wt_dG.repeat_interleave(num_mut_seqs, dim=0)
        ddG_batched *= (
            -1
        )  # use the convention that a negative value (ddG < 0) represents a stabilizing mutation.
        return ddG_batched

    def binding_ddG(
        self, complex, binder1, binder2, complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs
    ):
        """We calculate the binding ddG by decomposing it into three folding ddG terms,
        corresponding to the entire complex and each individual binders.
        """
        complex_ddG_fold = self.folding_ddG(complex, complex_mut_seqs)
        binder1_ddG_fold = self.folding_ddG(binder1, binder1_mut_seqs)
        binder2_ddG_fold = self.folding_ddG(binder2, binder2_mut_seqs)

        ddG = complex_ddG_fold - (binder1_ddG_fold + binder2_ddG_fold)

        return ddG

    def forward(
        self, complex, binder1, binder2, complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs
    ):
        return self.binding_ddG(
            complex, binder1, binder2, complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs
        )

    def _get_decoding_order(self, chain_M):
        """Generate a random decoding order with the same shape as chain_M."""
        if self.l_to_r:
            # decoding order is from left to right
            return torch.arange(chain_M.shape[1], device=self.device)
        # decoding order is random
        return torch.argsort(torch.abs(torch.randn(chain_M.shape, device=self.device)))

    def _get_backbone_noise(self, X):
        """Generate random backbone noise. Defaults to 0.1A."""
        return self.noise_level * torch.randn_like(X, device=self.device)
