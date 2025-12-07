from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard Losses


def cosine_sim_matrix(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two embedding batches.
    Args: z1, z2: (N, d)
    Returns: sim: (N, N)
    """
    return torch.matmul(z1, z2.T)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Standard SimCLR/InfoNCE contrastive loss.
    """
    assert z1.shape == z2.shape
    N = z1.size(0)

    # Pairwise similarity
    sim = cosine_sim_matrix(z1, z2)
    sim = sim / temperature

    # Positive similarities are on diagonal
    pos_sim = torch.diag(sim)

    # Denominator: sum over all columns
    exp_sim = torch.exp(sim)
    denom = exp_sim.sum(dim=1)

    loss = -torch.log(torch.exp(pos_sim) / denom)
    return loss.mean()


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    NT-Xent contrastive loss (SimCLR / GraphCL standard).
    """
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    sim = cosine_sim_matrix(z, z)
    sim = sim / temperature

    mask = torch.eye(2*N, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)

    pos_idx = torch.arange(N, device=z.device)
    pos_idx = torch.cat([pos_idx + N, pos_idx])

    log_prob = torch.log_softmax(sim, dim=1)

    loss = -log_prob[torch.arange(2*N), pos_idx]
    return loss.mean()


# MoCHi class

class ExtendedMoCHILoss(nn.Module):
    def __init__(self, temperature=0.1, num_negatives=128):
        """
        Extended InfoNCE Loss with MoCHi hard negative synthesis.
        Args:
            temperature (float): Scaling factor for logits.
            num_negatives (int): Number of synthetic negatives to generate.
        """
        super(ExtendedMoCHILoss, self).__init__()
        self.tau = temperature
        self.num_negatives = num_negatives

    def forward(self, anchor, positives, hard_negatives):
        """
        Computes the loss for a SINGLE anchor node.
        Args:
            anchor (Tensor): [1, D]
            positives (Tensor): [N_pos, D]
            hard_negatives (Tensor): [N_hard, D]
        """
        # 1. Normalize everything
        anchor = F.normalize(anchor, dim=1)
        positives = F.normalize(positives, dim=1)
        
        # If no hard negatives exist, fall back to standard negatives
        if hard_negatives.size(0) > 0:
            hard_negatives = F.normalize(hard_negatives, dim=1)
            
            # 2. MoCHi Synthesis
            synthetic_negs = self.mochi_generation(anchor, hard_negatives)
            all_negatives = torch.cat([hard_negatives, synthetic_negs], dim=0)
        else:
            all_negatives = hard_negatives

        # 3. Calculate Logits
        # Positive logits: s(q, k+) / tau
        pos_logits = torch.matmul(positives, anchor.t()).squeeze(-1) / self.tau
        
        # Negative logits: s(q, negs) / tau
        if all_negatives.size(0) > 0:
            neg_logits = torch.matmul(all_negatives, anchor.t()).squeeze(-1) / self.tau
        else:
            neg_logits = torch.tensor([-1e9]).to(anchor.device)

        # 4. Compute Extended InfoNCE
        neg_sum_exp = torch.sum(torch.exp(neg_logits))
        
        loss = 0
        n_pos = pos_logits.size(0)
        
        if n_pos > 0:
            for i in range(n_pos):
                pos_val = torch.exp(pos_logits[i])
                denom = pos_val + neg_sum_exp
                loss += -torch.log(pos_val / (denom + 1e-8))
            
            loss = loss / n_pos
            
        return loss

    def mochi_generation(self, anchor, hard_negs):
        """Synthesizes harder negatives by interpolating features."""
        N = hard_negs.size(0)
        if N < 2: return hard_negs 
        
        num_mix_anchor = self.num_negatives // 2
        num_mix_negs = self.num_negatives - num_mix_anchor
        
        # Type 1: Hardest {h'-} (Mix Anchor + Negs)
        indices = torch.randint(0, N, (num_mix_anchor,)).to(anchor.device)
        selected_negs = hard_negs[indices]
        alpha = torch.rand(num_mix_anchor, 1).to(anchor.device) * 0.4 + 0.1 
        hardest_negs = (1 - alpha) * selected_negs + alpha * anchor
        hardest_negs = F.normalize(hardest_negs, dim=1)
        
        # Type 2: Harder {h-} (Mix Negs + Negs)
        idx_a = torch.randint(0, N, (num_mix_negs,)).to(anchor.device)
        idx_b = torch.randint(0, N, (num_mix_negs,)).to(anchor.device)
        negs_a = hard_negs[idx_a]
        negs_b = hard_negs[idx_b]
        beta = torch.rand(num_mix_negs, 1).to(anchor.device) * 0.4 + 0.3 
        harder_negs = beta * negs_a + (1 - beta) * negs_b
        harder_negs = F.normalize(harder_negs, dim=1)
        
        return torch.cat([hardest_negs, harder_negs], dim=0)


if __name__ == '__main__':

    """
    Code snippet to visualize the effect of temperature
    The higher the temperature, the more uniform the probabilities
    """
    #Toy dataset
    torch.manual_seed(0)

    N, d = 4, 8

    # Random embeddings
    z1 = F.normalize(torch.randn(N, d), dim=1)

    # Positives = noisy copies
    z2 = F.normalize(0.7 * z1 + 0.3 * torch.randn(N, d), dim=1)

    temperatures = [2.0, 0.5, 0.2, 0.1, 0.05]

    for tau in temperatures:
        # Inspect how probabilities change
        sim = cosine_sim_matrix(z1, z2) / tau
        probs = F.softmax(sim, dim=1)   # (N, N)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1).mean()

        print(f"\nTemperature τ = {tau:.3f}")
        print("Softmax row probabilities:")
        print(probs)
        print("Entropy: ", entropy)
